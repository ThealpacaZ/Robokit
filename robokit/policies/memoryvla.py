"""MemoryVLA 全量 checkpoint（laMem-VLA 系列 33.5 GB `.pt`）的 policy 适配器。

和 memvla_lora.py 的区别：那边加载的是「CogACT-Large 底座 + 仅可训练参数」的 LoRA
包，这边加载的是上游 `train.py` 直接存下的完整 VLA checkpoint，`load_vla` 一步到位。
两者的图像预处理、夹爪门控、动作契约完全一致。

服务端（GPU）：

    python scripts/serve_policy.py --model lamem-v2 --mode sync        # 走登记表
    python scripts/deploy_server.py --port 8080 --policy memoryvla \\
        --policy-arg checkpoint=/path/RUN/checkpoints/xxx.pt \\
        --policy-arg codebase=/path/MemoryVLA-openvla-codebase \\
        --policy-arg unnorm_key=Stack_one_cup_on_top_of_another_cup_b0

checkpoint 路径必须长成 `<RUN>/checkpoints/<name>.pt`，且 `<RUN>/` 下有 config.json 与
dataset_statistics.json —— 这是上游 `vla/load.py` 的硬断言，不是本文件的要求。
（quhongyu123/laMem-VLA 的 zip 里目录拼成了 `checkpints/`，落盘时要改名。）

unnorm_key 必须是 dataset_statistics.json 里的键，决定动作反归一化的尺度；写错会
KeyError 或尺度全错。laMem-VLA v2 可选：Stack_one_cup_on_top_of_another_cup_b0 /
Cover_the_building_block_with_a_cup,_then_lift_up_the_cup_covering_the_block._b0 / _b1。

动作语义：每次 predict_action 输出 (future_action_window_size+1, 7) 动作块，每行
[局部 delta 位姿(6), 夹爪]，米 / 真弧度 / xyz 外旋。夹爪是连续绝对开度
（dataset_statistics 里该维 mask=False，不反归一化），默认通过 MEMVLA_BINARIZE_GRIPPER=0
关掉上游 predict_action 内的 0.5 硬二值化，并拒绝没有该门控补丁的 codebase。

图像预处理默认 train_aligned（整帧 resize 224 → 0.9 中心裁），理由见
memvla_lora._preprocess；preprocess="deploy" 复刻上游 deploy.py 的中心方形裁剪，仅供对照。

记忆机制：每个新客户端连接（= 新 episode）第一帧传 episode_first_frame='True'，
重置 MemoryVLA 的记忆库；所以真机端每一轮都要重新连接，不能复用连接跨 episode。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from robokit.comm import decode_image_maybe_jpeg
from robokit.image import lememory_preprocess
from robokit.policies.memvla_lora import _configure_gripper_contract

ACTION_DIM = 7
DEFAULT_PREPROCESS = "train_aligned"
# config.yaml 里这些键不是模型构造参数（路径 / 训练时的 token），不能透传给 load_vla
NON_MODEL_KEYS = ("model_id_or_path", "saved_model_path", "pretrained_checkpoint", "hf_token")


def _load_load_vla(codebase):
    """取出 vla.load_vla，但不执行 vla/__init__.py。

    `vla/__init__.py` 最后一行会导入 .materialize，那条链一路拉到 vla.datasets.rlds，
    需要 tensorflow / tensorflow_datasets / dlimp —— 全是训练侧依赖，推理用不上。
    做法是先把一个只有 __path__ 的 `vla` 包对象放进 sys.modules，Python 就不会再去跑
    真正的 __init__，之后按子模块逐个导入。vla/load.py 里有 `from vla import MemoryVLA`，
    所以 memory_vla 必须先导入并挂到包上。
    """
    import importlib
    import types

    vla_dir = Path(codebase) / "vla"
    if "vla" not in sys.modules:
        pkg = types.ModuleType("vla")
        pkg.__path__ = [str(vla_dir)]
        sys.modules["vla"] = pkg
        memory_vla = importlib.import_module("vla.memory_vla")
        pkg.MemoryVLA = memory_vla.MemoryVLA
    return importlib.import_module("vla.load").load_vla


def _load_adaptive_ensembler(codebase):
    """按文件路径加载 AdaptiveEnsembler，绕开 evaluation.simpler_env 包的 __init__
    （它会连带导入需要 transforms3d 等仿真评测依赖的模块）。"""
    import importlib.util

    path = Path(codebase) / "evaluation" / "simpler_env" / "adaptive_ensemble.py"
    spec = importlib.util.spec_from_file_location("_robokit_adaptive_ensemble", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.AdaptiveEnsembler


def checkpoint_layout(checkpoint):
    """返回 (checkpoint, run_dir)，并按上游 load_vla 的断言先在这里把路径问题说清楚。"""
    checkpoint = Path(checkpoint).expanduser()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint 不存在: {checkpoint}")
    if checkpoint.suffix != ".pt" or checkpoint.parent.name != "checkpoints":
        raise ValueError(
            f"MemoryVLA checkpoint 必须是 <RUN>/checkpoints/<name>.pt，实际 {checkpoint}"
        )
    run_dir = checkpoint.parents[1]
    for name in ("config.json", "dataset_statistics.json"):
        path = run_dir / name
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"{run_dir} 缺 {name}（上游 load_vla 需要它）")
    return checkpoint, run_dir


class MemoryVLAPolicy:
    def __init__(self, checkpoint, codebase, camera="cam_high", instruction=None,
                 unnorm_key="Stack_one_cup_on_top_of_another_cup_b0", image_size=224,
                 cfg_scale=1.5, num_ddim_steps=10, use_ddim=True, use_bf16=True,
                 action_ensemble=False, action_ensemble_horizon=2,
                 adaptive_ensemble_alpha=0.1, action_chunking_window=None,
                 horizon=None, action_space="eef_delta", preprocess=DEFAULT_PREPROCESS,
                 binarize_gripper=False, device="cuda", view=None,
                 llama_hf_path=None, timm_pretrained=True):
        if action_chunking_window is not None and action_ensemble:
            raise ValueError("action_chunking_window 与 action_ensemble 互斥")
        if preprocess not in ("deploy", "train_aligned"):
            raise ValueError(f"preprocess must be deploy|train_aligned, got {preprocess!r}")
        if action_space != "eef_delta":
            raise ValueError(
                f"laMem-VLA 训练标签是局部 EEF delta，action_space 只能是 eef_delta，got {action_space!r}"
            )
        checkpoint, run_dir = checkpoint_layout(checkpoint)
        codebase = str(Path(codebase).expanduser())
        if not (Path(codebase) / "vla" / "load.py").is_file():
            raise FileNotFoundError(f"codebase 不像 MemoryVLA-openvla-codebase: {codebase}")
        if codebase not in sys.path:
            sys.path.insert(0, codebase)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        # 下面三个环境变量都要在 import vla/prismatic 之前设好：
        #   llama_hf_path   Llama-2 config+tokenizer 的本地目录（推理只建空模型，不下 13 GB 权重）
        #   timm_pretrained False = 不下 DINOv2/SigLIP 预训练权重，checkpoint 自带 vision_backbone
        if llama_hf_path:
            if not (Path(llama_hf_path) / "tokenizer_config.json").is_file():
                raise FileNotFoundError(f"llama_hf_path 缺 tokenizer_config.json: {llama_hf_path}")
            os.environ["MEMVLA_LLAMA2_HF_PATH"] = str(llama_hf_path)
        os.environ["MEMVLA_TIMM_PRETRAINED"] = "1" if timm_pretrained else "0"
        # 夹爪门控：上游在 predict_action 内读这个环境变量
        self.binarize_gripper = _configure_gripper_contract(codebase, binarize_gripper)

        import torch
        import yaml

        load_vla = _load_load_vla(codebase)

        self.torch = torch
        self.camera = camera
        self.instruction = instruction
        self.unnorm_key = unnorm_key
        self.image_size = (int(image_size), int(image_size))
        self.cfg_scale = float(cfg_scale)
        self.num_ddim_steps = int(num_ddim_steps)
        self.use_ddim = bool(use_ddim)
        self.action_chunking_window = None if action_chunking_window is None else int(action_chunking_window)
        self.horizon = None if horizon is None else int(horizon)
        self.action_space = action_space
        self.preprocess = preprocess
        self.device = device
        self.view = view
        self.infer_count = 0
        self.checkpoint = str(checkpoint)

        # 与上游 deploy.py 一致：RUN 目录下的 config.yaml 作为 load_vla 的 kwargs
        kwargs = {}
        config_yaml = run_dir / "config.yaml"
        if config_yaml.exists():
            kwargs = yaml.safe_load(config_yaml.read_text()) or {}
        for key in NON_MODEL_KEYS:
            kwargs.pop(key, None)

        with open(run_dir / "dataset_statistics.json") as f:
            stats = json.load(f)
        if unnorm_key not in stats:
            raise RuntimeError(
                f"unnorm_key={unnorm_key!r} 不在 {run_dir / 'dataset_statistics.json'}，"
                f"可选: {sorted(stats)}"
            )
        act = stats[unnorm_key]["action"]
        if len(act.get("q01", [])) != ACTION_DIM:
            raise RuntimeError(f"dataset_statistics[{unnorm_key}] 不是 {ACTION_DIM} 维动作")

        print(f"[memoryvla] load {checkpoint}", flush=True)
        self.vla = load_vla(model_id_or_path=str(checkpoint), load_for_training=False, **kwargs)
        self.vla = self.vla.to(device).eval()
        self.vla = self.vla.to(torch.bfloat16 if use_bf16 else torch.float32)
        self.chunk_size = int(getattr(self.vla, "future_action_window_size", 15)) + 1
        print(f"[memoryvla] unnorm_key={unnorm_key} mask={act.get('mask')} "
              f"chunk={self.chunk_size} preprocess={preprocess} "
              f"binarize_gripper={self.binarize_gripper}", flush=True)

        self.ensembler = None
        if action_ensemble:
            ensembler_cls = _load_adaptive_ensembler(codebase)
            self.ensembler = ensembler_cls(action_ensemble_horizon, adaptive_ensemble_alpha)
        self._first_frame = True

    def describe(self) -> str:
        return (
            f"family=memoryvla, action_space={self.action_space}, H={self.chunk_size}, "
            f"horizon={self.horizon or self.chunk_size}, camera={self.camera}, "
            f"unnorm_key={self.unnorm_key}, preprocess={self.preprocess}, rtc=off"
        )

    def reset(self):
        """新客户端连接 => 下一帧是 episode 首帧，重置 MemoryVLA 的记忆库。"""
        if self.ensembler is not None:
            self.ensembler.reset()
        self._first_frame = True

    def _image(self, obs):
        from PIL import Image

        images = obs["images"]
        if self.camera not in images:
            raise RuntimeError(f"观测里没有相机 {self.camera!r}，实际有: {sorted(images)}")
        received = np.asarray(decode_image_maybe_jpeg(images[self.camera]), dtype=np.uint8)
        if received.ndim != 3 or received.shape[-1] != 3:
            raise RuntimeError(f"相机帧应为 (H,W,3) uint8，实际 {received.shape}")
        image = lememory_preprocess(Image.fromarray(received), self.image_size, self.preprocess)
        if self.view is not None and self.view.enabled:
            self.view.images({self.camera: received}, self.infer_count, kind="recv")
            self.view.images({self.camera: np.asarray(image)}, self.infer_count, kind="model")
        return image

    def infer(self, obs):
        instruction = self.instruction or obs.get("instruction") or ""
        if not instruction:
            raise RuntimeError("instruction 为空；VLA 动作完全由语言条件决定")
        image = self._image(obs)
        self.infer_count += 1

        first = "True" if self._first_frame else "False"
        self._first_frame = False
        with self.torch.inference_mode():
            out = self.vla.predict_action(
                image=image, instruction=instruction, unnorm_key=self.unnorm_key,
                cfg_scale=self.cfg_scale, use_ddim=self.use_ddim,
                num_ddim_steps=self.num_ddim_steps, episode_first_frame=first,
            )
        # predict_action 返回 (unnormed_actions, normalized_actions) 二元组
        unnormed = out[0] if isinstance(out, (tuple, list)) else out
        chunk = np.asarray(unnormed, dtype=np.float32)
        if chunk.ndim == 3 and chunk.shape[0] == 1:
            chunk = chunk[0]
        if chunk.ndim != 2 or chunk.shape[1] != ACTION_DIM:
            raise RuntimeError(f"动作块应为 (N,{ACTION_DIM})，实际 {chunk.shape}")

        if self.ensembler is not None:
            action = np.asarray(self.ensembler.ensemble_action(chunk), dtype=np.float32)
            if self.binarize_gripper:
                action[6] = float(action[6] > 0.5)
            return action[None, :]
        if self.action_chunking_window is not None:
            chunk = chunk[: self.action_chunking_window]
        return chunk if self.horizon is None else chunk[: self.horizon]
