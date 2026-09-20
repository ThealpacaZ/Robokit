"""LeRobot flow-matching（PI0 / PI0.5）权重的 robokit 适配器。

一个类同时覆盖以前四份几乎逐行重复的实现：

    family        pi0 | pi05
    权重形态      全量微调目录 | PEFT adapter 目录 | HF repo id
    动作空间      eef_delta（局部 SE(3) delta + 绝对夹爪）| joint（绝对关节角 + 夹爪）
    推理方式      sync（一次一块）| rtc（ΠGDM 前缀引导，arXiv:2506.07339）

动作契约由**数据集**定义，不由 policy 定义：``use_relative_actions`` 必须为 false，
否则 LeRobot 会再叠一层自己的相对化，和 RLDS 转换生成的标签不是同一个量。

单位约定：米 / 真弧度 / xyz 外旋欧拉序（见 robokit/pose.py）。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FAMILIES = ("pi0", "pi05")
ACTION_SPACES = ("eef_delta", "joint")


class LeRobotDiTPolicy:
    """加载 PI0/PI0.5 权重，产出 ``(N, 7)`` 的 robokit 动作块。"""

    def __init__(
        self,
        checkpoint: str,
        family: str = "pi05",
        camera: str = "cam_high",
        arm: str | None = None,
        instruction: str | None = None,
        horizon: int | None = None,
        action_space: str = "eef_delta",
        expected_chunk_size: int | None = None,
        device: str = "cuda",
        compile_model: bool = True,
        rtc: bool = False,
        rtc_max_guidance_weight: float = 5.0,
        rtc_schedule: str = "EXP",
        rtc_num_inference_steps: int = 5,
        view=None,
    ):
        import torch
        # lerobot 版本兼容:0.4.4 起这些符号搬了家/不再 re-export,双路径 fallback
        try:
            from lerobot.configs import RTCAttentionSchedule
        except ImportError:
            from lerobot.configs.types import RTCAttentionSchedule
        from lerobot.configs.policies import PreTrainedConfig
        try:
            from lerobot.policies import get_policy_class, make_pre_post_processors
        except ImportError:
            from lerobot.policies.factory import get_policy_class, make_pre_post_processors
        try:
            from lerobot.policies.rtc import RTCConfig
        except ImportError:
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

        self.torch = torch
        self.device = torch.device(device)
        self.family = str(family)
        self.camera = camera
        self.arm = arm
        self.instruction = instruction
        self.horizon = None if horizon is None else int(horizon)
        self.action_space = str(action_space)
        self.rtc = bool(rtc)
        self.view = view
        self.infer_count = 0
        if self.family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {self.family!r}")
        if self.action_space not in ACTION_SPACES:
            raise ValueError(
                f"action_space must be one of {ACTION_SPACES}, got {self.action_space!r}"
            )

        checkpoint_path = Path(checkpoint).expanduser()
        pretrained_path = str(checkpoint_path) if checkpoint_path.exists() else checkpoint
        is_adapter = checkpoint_path.is_dir() and (
            checkpoint_path / "adapter_config.json"
        ).is_file()
        peft_config = None
        if is_adapter:
            from peft import PeftConfig

            peft_config = PeftConfig.from_pretrained(pretrained_path)
            base_path = peft_config.base_model_name_or_path
            if not base_path:
                raise RuntimeError(
                    f"{pretrained_path}: adapter_config.json has no base_model_name_or_path"
                )
        else:
            base_path = pretrained_path

        policy_config = PreTrainedConfig.from_pretrained(pretrained_path)
        if policy_config.type != self.family:
            raise RuntimeError(
                f"登记表说这是 {self.family!r}，checkpoint 里却是 "
                f"{policy_config.type!r}：{pretrained_path}"
            )
        if policy_config.use_relative_actions:
            raise RuntimeError(
                "The dataset already defines the physical action contract; "
                "policy.use_relative_actions must remain false"
            )
        self.chunk_size = int(policy_config.chunk_size)
        if expected_chunk_size is not None and self.chunk_size != int(expected_chunk_size):
            raise RuntimeError(
                f"登记表写的 chunk_size={int(expected_chunk_size)}，checkpoint 实际是 "
                f"{self.chunk_size}；请改 configs/models.yaml 而不是让两边对不上"
            )
        if self.horizon is not None and not 1 <= self.horizon <= self.chunk_size:
            raise ValueError(
                f"horizon 必须落在 [1, {self.chunk_size}]，got {self.horizon}"
            )
        policy_config.device = str(self.device)
        policy_config.compile_model = bool(compile_model)
        if self.rtc:
            if int(rtc_num_inference_steps) <= 0:
                raise ValueError("rtc_num_inference_steps must be positive")
            policy_config.num_inference_steps = int(rtc_num_inference_steps)
            try:
                schedule = RTCAttentionSchedule[str(rtc_schedule).upper()]
            except KeyError as exc:
                choices = ", ".join(item.name for item in RTCAttentionSchedule)
                raise ValueError(f"rtc_schedule must be one of {choices}") from exc
            policy_config.rtc_config = RTCConfig(
                enabled=True,
                # 每个请求会给出真正的 overlap end；这里只是 config 期必填的兜底值。
                execution_horizon=policy_config.chunk_size // 2,
                max_guidance_weight=float(rtc_max_guidance_weight),
                prefix_attention_schedule=schedule,
            )
        if len(policy_config.image_features) != 1:
            raise RuntimeError(
                "This robokit adapter requires the fine-tuned policy config to contain exactly "
                f"one image feature, got {list(policy_config.image_features)}. Train with "
                "--policy.input_features=null on the single-camera dataset."
            )
        self.image_key = next(iter(policy_config.image_features))
        expected_image_key = f"observation.images.{camera}"
        if self.image_key != expected_image_key:
            raise RuntimeError(
                f"Checkpoint expects {self.image_key!r}, but camera={camera!r} maps to "
                f"{expected_image_key!r}"
            )
        state_feature = policy_config.input_features.get("observation.state")
        action_feature = policy_config.output_features.get("action")
        if state_feature is None or tuple(state_feature.shape) != (7,):
            raise RuntimeError(f"Expected 7-D observation.state, got {state_feature}")
        if action_feature is None or tuple(action_feature.shape) != (7,):
            raise RuntimeError(f"Expected 7-D action, got {action_feature}")

        policy_class = get_policy_class(self.family)
        if is_adapter:
            from peft import PeftModel

            base_policy = policy_class.from_pretrained(
                base_path, config=policy_config, revision=peft_config.revision
            )
            self.policy = PeftModel.from_pretrained(
                base_policy, pretrained_path, config=peft_config, is_trainable=False
            ).to(self.device).eval()
        else:
            self.policy = policy_class.from_pretrained(
                pretrained_path, config=policy_config
            ).to(self.device).eval()
        self.policy_config = policy_config
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=pretrained_path,
            preprocessor_overrides={"device_processor": {"device": str(self.device)}},
        )

    def describe(self) -> str:
        return (
            f"family={self.family}, action_space={self.action_space}, "
            f"H={self.chunk_size}, horizon={self.horizon or self.chunk_size}, "
            f"camera={self.camera}, rtc={'on' if self.rtc else 'off'}"
        )

    def reset(self) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset()

    def _select_arm(self, state: dict) -> str:
        if self.arm is not None:
            if self.arm not in state:
                raise RuntimeError(f"Arm {self.arm!r} is absent; available={sorted(state)}")
            return self.arm
        if len(state) != 1:
            raise RuntimeError(
                f"Expected exactly one arm or explicit arm=..., got {sorted(state)}"
            )
        return next(iter(state))

    def _prepare(self, obs: dict):
        from lerobot.policies.utils import prepare_observation_for_inference

        from robokit.comm import decode_image_maybe_jpeg

        if self.camera not in obs["images"]:
            raise RuntimeError(
                f"Camera {self.camera!r} is absent; available={sorted(obs['images'])}"
            )
        arm = self._select_arm(obs["state"])
        arm_state = obs["state"][arm]
        if self.action_space == "joint":
            joint = np.asarray(arm_state.get("joint"), dtype=np.float32)
            if joint.shape != (6,):
                raise RuntimeError(
                    f"Arm {arm!r} joint state must have shape (6,), got {joint.shape}"
                )
            state = np.concatenate(
                [joint, np.asarray([arm_state["gripper"]], dtype=np.float32)]
            )
        else:
            eef_pose = arm_state.get("eef_pose")
            if eef_pose is None:
                raise RuntimeError(f"Arm {arm!r} does not provide eef_pose")
            state = np.concatenate(
                [
                    np.asarray(eef_pose, dtype=np.float32).reshape(6),
                    np.asarray([arm_state["gripper"]], dtype=np.float32),
                ]
            )
        instruction = self.instruction or obs.get("instruction") or ""
        if not instruction:
            raise RuntimeError("instruction is empty")
        received = np.asarray(
            decode_image_maybe_jpeg(obs["images"][self.camera]), dtype=np.uint8
        )
        frame = {self.image_key: received, "observation.state": state}
        prepared = prepare_observation_for_inference(
            frame, self.device, task=instruction, robot_type="piper"
        )
        prepared = self.preprocessor(prepared)
        if self.view is not None and self.view.enabled:
            # 收到的原始帧 + 预处理之后真正进模型的那张，一起落盘/显示。
            self.view.images({self.camera: received}, self.infer_count, kind="recv")
            tensor = prepared.get(self.image_key) if hasattr(prepared, "get") else None
            if tensor is not None:
                self.view.tensor(tensor, self.infer_count, name=self.camera, kind="model")
        self.infer_count += 1
        return prepared

    def _as_action_array(self, chunk, expected_rows: int | None = None) -> np.ndarray:
        if hasattr(chunk, "detach"):
            chunk = chunk.detach().cpu().numpy()
        result = np.asarray(chunk, dtype=np.float32)
        if result.ndim == 3 and result.shape[0] == 1:
            result = result[0]
        if result.ndim != 2 or result.shape[1] != 7:
            raise RuntimeError(
                f"{self.family} action chunk must have shape (N, 7), got {result.shape}"
            )
        if not np.isfinite(result).all():
            raise RuntimeError(f"{self.family} action chunk contains non-finite values")
        if expected_rows is not None and result.shape[0] != int(expected_rows):
            raise RuntimeError(
                f"{self.family} must return {int(expected_rows)} actions, got {result.shape[0]}"
            )
        return result

    def infer(self, obs: dict) -> np.ndarray:
        prepared = self._prepare(obs)
        with self.torch.inference_mode():
            chunk = self.policy.predict_action_chunk(prepared)
            chunk = self.postprocessor(chunk)
        result = self._as_action_array(chunk)
        return result if self.horizon is None else result[: self.horizon]

    def infer_rtc(
        self,
        obs: dict,
        *,
        previous_chunk=None,
        executed_at_start: int = 0,
        inference_delay: int = 0,
    ) -> tuple[np.ndarray, object]:
        """按 arXiv:2506.07339 用 RTC ΠGDM 前缀引导生成完整 H 步 chunk。

        ``previous_chunk`` 必须留在**归一化/模型空间**：LeRobot 的 postprocessor 才
        把它映射到机器人物理单位，而 RTC 的去噪与引导都发生在归一化空间。
        """
        if not self.rtc:
            raise RuntimeError("infer_rtc requires rtc=True when loading the policy")

        prepared = self._prepare(obs)
        horizon = self.chunk_size
        executed = int(executed_at_start)
        delay = int(inference_delay)
        if not 0 <= executed < horizon:
            raise ValueError(f"executed_at_start must be in [0, {horizon}), got {executed}")
        if delay < 0:
            raise ValueError(f"inference_delay must be non-negative, got {delay}")

        kwargs = {}
        if previous_chunk is not None:
            if not self.torch.is_tensor(previous_chunk):
                previous_chunk = self.torch.as_tensor(previous_chunk, device=self.device)
            previous_chunk = previous_chunk.to(self.device)
            if previous_chunk.ndim == 3 and previous_chunk.shape[0] == 1:
                previous_chunk = previous_chunk[0]
            if previous_chunk.ndim != 2 or previous_chunk.shape[0] != horizon:
                raise ValueError(
                    "previous normalized chunk must have shape "
                    f"({horizon}, action_dim), got {tuple(previous_chunk.shape)}"
                )
            overlap_end = horizon - executed
            if delay > overlap_end:
                raise ValueError(
                    f"RTC requires inference_delay <= H-s ({overlap_end}), got {delay}"
                )
            kwargs = {
                "prev_chunk_left_over": previous_chunk[executed:],
                "inference_delay": delay,
                # LeRobot names this field execution_horizon, but RTCProcessor
                # uses it as the exclusive end of the non-zero prefix mask.
                # Equation (5) says that end is H-s.
                "execution_horizon": overlap_end,
            }

        # Do not use torch.inference_mode here. RTCProcessor temporarily enables
        # gradients to compute the vector-Jacobian product at each denoising step.
        with self.torch.no_grad():
            normalized = self.policy.predict_action_chunk(prepared, **kwargs)
        cached = normalized.detach().clone()
        physical = self.postprocessor(normalized)
        return self._as_action_array(physical, expected_rows=horizon), cached
