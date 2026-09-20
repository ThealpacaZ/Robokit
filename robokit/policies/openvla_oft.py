"""OpenVLA-OFT LoRA + L1 action-head adapter for the generic sync server."""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np


class OpenVLAOFTPolicy:
    """Load a frozen latest-only OFT checkpoint and return a ``(30, 7)`` chunk."""

    def __init__(self, checkpoint, base, codebase, camera="cam_high", instruction=None,
                 unnorm_key="robokit_stackcups_all", center_crop=True, horizon=None,
                 device="cuda:0", robot_platform="piper", action_space="eef_delta"):
        if str(robot_platform).lower() != "piper":
            raise ValueError(f"OpenVLA-OFT 当前 checkpoint 要求 robot_platform=piper，得到 {robot_platform!r}")
        checkpoint, base, codebase = Path(checkpoint), Path(base), Path(codebase)
        required = (
            checkpoint / "lora_adapter" / "adapter_model.safetensors",
            checkpoint / "lora_adapter" / "adapter_config.json",
            checkpoint / "action_head--latest_checkpoint.pt",
            checkpoint / "dataset_statistics.json",
            base / "config.json",
        )
        missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"OpenVLA-OFT checkpoint 不完整: {missing}")
        if str(codebase) not in sys.path:
            sys.path.insert(0, str(codebase))
        # openvla-oft 的 prismatic/vla/constants.py 在 import 时扫 sys.argv 里的关键词
        # 决定 NUM_ACTIONS_CHUNK / ACTION_DIM / 归一化方式，认不出就静默退回 LIBERO
        # （chunk=8）。训练命令行里带 robokit_* 所以选中 PIPER（chunk=30），推理命令行
        # 里没有，于是同一份权重在服务端被当成 8 步——不报错，只是每次少给 22 行。
        # 这里在 import 之前把平台写进 argv，让两端拿到同一套常量。
        if str(robot_platform).lower() not in " ".join(sys.argv).lower():
            sys.argv = list(sys.argv) + [f"--robot-platform={robot_platform}"]

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        import torch
        from peft import PeftModel
        from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor
        from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
        from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
        from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
        from prismatic.models.action_heads import L1RegressionActionHead
        from experiments.robot.openvla_utils import get_vla_action

        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

        print(f"[openvla_oft] load base {base}", flush=True)
        vla = AutoModelForVision2Seq.from_pretrained(
            str(base), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        )
        vla.vision_backbone.set_num_images_in_input(1)
        print(f"[openvla_oft] load adapter {checkpoint / 'lora_adapter'}", flush=True)
        vla = PeftModel.from_pretrained(vla, str(checkpoint / "lora_adapter"))
        vla = vla.to(device).eval()

        action_head = L1RegressionActionHead(
            input_dim=vla.llm_dim, hidden_dim=vla.llm_dim, action_dim=7
        ).to(torch.bfloat16).to(device)
        state = torch.load(checkpoint / "action_head--latest_checkpoint.pt",
                           weights_only=True, map_location="cpu")
        state = {key.removeprefix("module."): value for key, value in state.items()}
        action_head.load_state_dict(state)
        self.action_head = action_head.eval()

        stats = json.loads((checkpoint / "dataset_statistics.json").read_text(encoding="utf-8"))
        if unnorm_key not in stats:
            raise KeyError(f"unnorm_key={unnorm_key!r} 不在 dataset_statistics: {list(stats)}")
        vla.norm_stats = stats
        vla.base_model.model.norm_stats = stats

        self.vla = vla
        self.processor = AutoProcessor.from_pretrained(str(base), trust_remote_code=True)
        self.camera, self.instruction = str(camera), instruction
        self.unnorm_key = str(unnorm_key)
        self.horizon = None if horizon is None else int(horizon)
        # 服务端回包要带 action_space（真机端逐条校验，接错端口直接拒绝执行），
        # chunk_size 则由 openvla-oft 的 NUM_ACTIONS_CHUNK 常量决定，登记表要对得上。
        self.action_space = str(action_space)
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK
        self.chunk_size = int(NUM_ACTIONS_CHUNK)
        self._torch, self._get_vla_action = torch, get_vla_action
        self.cfg = SimpleNamespace(num_images_in_input=1, center_crop=bool(center_crop),
                                   use_proprio=False, unnorm_key=self.unnorm_key)
        print(f"[openvla_oft] ready checkpoint={checkpoint} unnorm_key={self.unnorm_key}",
              flush=True)

    def describe(self) -> str:
        return (
            f"family=openvla_oft, action_space={self.action_space}, H={self.chunk_size}, "
            f"horizon={self.horizon or self.chunk_size}, camera={self.camera}, "
            f"unnorm_key={self.unnorm_key}, rtc=off"
        )

    def reset(self):
        return None

    def infer(self, obs):
        images = obs.get("images") or {}
        if self.camera not in images:
            raise RuntimeError(f"观测里没有相机 {self.camera!r}，实际有 {list(images)}")
        instruction = self.instruction or obs.get("instruction") or ""
        if not instruction:
            raise RuntimeError("instruction 为空")
        # 客户端发的是 JPEG dict（900KB→70KB，隧道往返从 1s 降到 0.16s），旧客户端发
        # 原始数组；decode_image_maybe_jpeg 两种都接。这个适配器写在 JPEG 协议之前，
        # 少了这一步会在第一次推理时报 "int() argument must be ... not 'dict'"。
        from robokit.comm import decode_image_maybe_jpeg

        frame = decode_image_maybe_jpeg(images[self.camera])
        observation = {"full_image": np.asarray(frame, dtype=np.uint8)}
        with self._torch.inference_mode():
            action = self._get_vla_action(
                self.cfg, self.vla, self.processor, observation, instruction,
                action_head=self.action_head, use_film=False,
            )
        action = np.asarray(action, dtype=np.float32)
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]
        if action.ndim != 2 or action.shape[1] != 7:
            raise RuntimeError(f"OpenVLA-OFT 动作块应为 (N,7)，实际 {action.shape}")
        return action if self.horizon is None else action[:self.horizon]
