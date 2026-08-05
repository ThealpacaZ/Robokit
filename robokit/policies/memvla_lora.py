"""MemoryVLA + LoRA 微调权重的 policy 适配器（CogACT-Large 底座）。

LoRA 训练存下来的是「仅可训练参数」的 3.5 GB 包（不是完整 checkpoint，不能 load_vla
一步到位），必须三步拼起来：

    1. load_vla(CogACT-Large)，kwargs 里剔除 use_ema（memory_vla.py:658 断言）
    2. freeze_backbones(stage) + 挂上**与训练时逐字一致**的 LoRA 配置
    3. 把 trainable_state_dict 灌进去

第 2 步的 r / alpha / target_modules 任何一处对不上，都会得到一个「结构对、权重错位」
的模型——能跑、不报错、但输出是垃圾。所以构造时会断言所有可训练参数都被 ckpt 覆盖到。

启动：
    python scripts/deploy_server.py --port 8080 --policy memvla_lora \\
      --policy-arg checkpoint=/root/autodl-tmp/runs/memvla-lora-r32/step-010000.pt \\
      --policy-arg base=/root/autodl-tmp/CogACT-Large/checkpoints/CogACT-Large.pt \\
      --policy-arg codebase=/root/autodl-tmp/MemoryVLA-openvla-codebase \\
      --policy-arg dataset_statistics=/root/autodl-tmp/runs/memvla-lora-r32/dataset_statistics.json

动作语义：7 维 [局部 delta 位姿(6), 夹爪]，米 / 真弧度 / xyz 外旋。MemoryVLA 输出
动作块 (future_action_window_size+1, 7)，所以客户端的 --horizon 可以 >1（这是它相对
OpenVLA base 单步策略的优势）。

Piper 的第 7 维训练标签是连续绝对开度，不是二值开/关。上游 MemoryVLA 默认会在
``predict_action`` 内按 0.5 把它硬二值化，因此本适配器默认在导入上游代码前设置
``MEMVLA_BINARIZE_GRIPPER=0``，并拒绝没有该门控补丁的 codebase。这样不会出现
EEF 六维正常、夹爪却每步在 0/70mm 间跳变的隐蔽部署错误。
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

from robokit.image import lememory_preprocess

ACTION_DIM = 7

# 图像预处理路径，见 _preprocess。默认对齐**训练** loader 的几何。
DEFAULT_PREPROCESS = "train_aligned"


def _preprocess(image, size=(224, 224), mode=DEFAULT_PREPROCESS):
    """把原始相机帧变成训练时 image_transform 见到的那种图。

    必须有这一步，否则训练和推理喂给视觉骨干的东西根本不是一回事：

        训练  RLDS 图 → dlimp.resize_image = tf.image.resize(img,(224,224),"lanczos3")
                        **整帧压扁，不裁剪、不保长宽比**
              → image_aug（train.py:54 默认 True）
                random_resized_crop(scale=[0.9,0.9], ratio=[1.0,1.0])
              → 得到 224×224 → 才进 vlm.vision_backbone.image_transform

        推理  原始 320×240 直接进 image_transform

    也就是说训练侧进 image_transform 的已经是 224×224 方图，推理侧却是 4:3 原图。
    无论 image_resize_strategy 是 resize-naive / resize-crop / letterbox，这两者都不
    相等（分别是"少了 0.9 放大"、"多裁掉 25% 宽"、"多出灰边"）。所以在调用
    predict_action 之前必须自己把图做成训练那样。

    mode="train_aligned"（默认，正确的那条）
        整帧 resize 到 224×224（4:3 会被横向压扁）→ 0.9 中心裁剪。横向视野 ≈ 95%。

    mode="deploy"（复刻上游 deploy.py:156，保留仅供对照）
        先中心裁方形（4:3 下左右各丢 12.5%）→ resize 224 → 0.9 中心裁剪。视野 ≈ 71%。

    上游仓库自己这两条就不一致：它的训练 loader 不裁方形，它的 deploy.py 裁。0.9 那步
    两边对得上（部署取中心 = 随机位置的期望值），多出来的只有那行方形裁剪。**权重的
    输入域由训练决定，所以取 train_aligned。** 实测两条路径像素 MAE 17.6。
    """
    return lememory_preprocess(image, size=size, mode=mode)


def _configure_gripper_contract(codebase, binarize_gripper):
    """Configure and verify the upstream gripper behavior before importing VLA."""
    codebase = Path(codebase)
    enabled = bool(binarize_gripper)
    os.environ["MEMVLA_BINARIZE_GRIPPER"] = "1" if enabled else "0"
    if not enabled:
        memory_vla = codebase / "vla" / "memory_vla.py"
        try:
            source = memory_vla.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                "binarize_gripper=False 时无法检查上游夹爪门控："
                f"{memory_vla}: {exc}"
            ) from exc
        if "MEMVLA_BINARIZE_GRIPPER" not in source:
            raise RuntimeError(
                f"binarize_gripper=False 需要 {memory_vla} 含 "
                "MEMVLA_BINARIZE_GRIPPER 门控补丁；否则上游会把连续 Piper "
                "夹爪标签静默变成 0/1"
            )
    return enabled


class MemVLALoRAPolicy:
    def __init__(self, checkpoint, base, codebase, dataset_statistics=None,
                 camera="cam_high", instruction=None, unnorm_key="robokit_dataset",
                 stage="align", lora_rank=32, lora_alpha=None,
                 cfg_scale=1.5, use_ddim=True, num_ddim_steps=10,
                 horizon=None, binarize_gripper=False,
                 preprocess=DEFAULT_PREPROCESS, image_size=224,
                 device="cuda:0"):
        codebase = str(codebase)
        if codebase not in sys.path:
            sys.path.insert(0, codebase)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        # This must happen before importing ``vla``.  The upstream module
        # reads the environment variable inside predict_action, and a missing
        # source gate would otherwise fail only after the GPU model is loaded.
        self.binarize_gripper = _configure_gripper_contract(
            codebase, binarize_gripper
        )

        import torch
        import yaml
        from peft import LoraConfig, get_peft_model
        from vla import load_vla

        self._torch = torch
        self.camera = camera
        self.instruction = instruction
        if preprocess not in ("deploy", "train_aligned"):
            raise ValueError(
                f"preprocess must be deploy|train_aligned, got {preprocess!r}"
            )
        self.preprocess = preprocess
        self.image_size = int(image_size)
        self.unnorm_key = unnorm_key
        self.cfg_scale = float(cfg_scale)
        self.use_ddim = bool(use_ddim)
        self.num_ddim_steps = int(num_ddim_steps)
        self.horizon = None if horizon is None else int(horizon)
        self.device = device
        self._first = True

        base = Path(base)
        kwargs = {}
        cfg_yaml = base.parent.parent / "config.yaml"
        if cfg_yaml.exists():
            kwargs = yaml.safe_load(cfg_yaml.read_text()) or {}
            for k in ("model_id_or_path", "saved_model_path", "pretrained_checkpoint",
                      "use_ema"):
                kwargs.pop(k, None)

        print(f"[memvla_lora] load base {base}", flush=True)
        vla = load_vla(model_id_or_path=str(base), load_for_training=True, **kwargs)

        vla.freeze_backbones(stage)
        alpha = int(lora_alpha) if lora_alpha is not None else int(lora_rank) * 2
        vla.vlm.llm_backbone.llm = get_peft_model(
            vla.vlm.llm_backbone.llm,
            LoraConfig(r=int(lora_rank), lora_alpha=alpha, lora_dropout=0.0,
                       target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                       init_lora_weights="gaussian"))

        print(f"[memvla_lora] load trainable weights {checkpoint}", flush=True)
        blob = torch.load(str(checkpoint), map_location="cpu", weights_only=False)
        sd = blob["trainable_state_dict"]
        _, unexpected = vla.load_state_dict(sd, strict=False)
        tr = {n for n, p in vla.named_parameters() if p.requires_grad}
        missing_tr = sorted(tr - set(sd.keys()))
        if missing_tr:
            raise RuntimeError(
                f"{len(missing_tr)} 个可训练参数没能从 ckpt 加载（LoRA 配置与训练时不一致？）"
                f" 例: {missing_tr[:3]}")
        if unexpected:
            raise RuntimeError(f"ckpt 里有模型不认的键: {list(unexpected)[:3]}")
        print(f"[memvla_lora] step={blob.get('step')} stage={blob.get('stage')} "
              f"lora_rank={blob.get('lora_rank')} 键数={len(sd)}", flush=True)

        for p in vla.parameters():
            if p.dtype == torch.float32:
                p.data = p.data.to(torch.bfloat16)
        self.vla = vla.to(device).eval()

        stats = Path(dataset_statistics) if dataset_statistics else \
            Path(checkpoint).parent / "dataset_statistics.json"
        if not stats.exists():
            raise FileNotFoundError(f"找不到 {stats}；缺了它动作反归一化尺度会全错")
        with open(stats) as f:
            self.vla.norm_stats = json.load(f)
        if unnorm_key not in self.vla.norm_stats:
            raise RuntimeError(f"unnorm_key='{unnorm_key}' 不在 {stats}，"
                               f"可选: {list(self.vla.norm_stats)}")
        act = self.vla.norm_stats[unnorm_key]["action"]
        print(f"[memvla_lora] norm_stats={stats} mask={act.get('mask')}", flush=True)

    def reset(self):
        """新客户端连接 => 下一帧作为 episode 首帧，重置 MemoryVLA 的记忆库。"""
        self._first = True

    def infer(self, obs):
        from PIL import Image

        images = obs["images"]
        if self.camera not in images:
            raise RuntimeError(f"观测里没有相机 '{self.camera}'，实际有: {list(images)}")
        instruction = self.instruction or obs.get("instruction") or ""
        if not instruction:
            raise RuntimeError("instruction 为空；VLA 动作完全由语言条件决定")

        img = Image.fromarray(np.asarray(images[self.camera], dtype=np.uint8))
        # 做成训练时 image_transform 见到的 224×224，见 _preprocess。不能把原始 4:3
        # 帧直接丢进 predict_action —— 那样视觉骨干拿到的图和训练时不是一回事。
        img = _preprocess(img, (self.image_size, self.image_size), self.preprocess)
        with self._torch.inference_mode():
            out = self.vla.predict_action(
                image=img, instruction=instruction, unnorm_key=self.unnorm_key,
                cfg_scale=self.cfg_scale, use_ddim=self.use_ddim,
                num_ddim_steps=self.num_ddim_steps,
                episode_first_frame='True' if self._first else 'False')
        self._first = False

        # predict_action 返回的是 **二元 tuple** `(unnormed_actions, normalized_actions)`
        # （见上游 deploy.py:116 的解包写法），不是单个数组。直接 np.asarray 会得到
        # (2, 16, 7) —— 那个前导的 2 是 tuple 长度而不是 batch 维，客户端会报
        # 「服务端返回 shape=[2,16,7]，期望单臂 (N,7)」。这里取反归一化后的物理量。
        if isinstance(out, (tuple, list)):
            out = out[0]
        act = np.asarray(out, dtype=np.float32)
        if act.ndim == 3 and act.shape[0] == 1:      # 可能带 batch=1
            act = act[0]
        if act.ndim == 1:
            act = act[None, :]
        if act.ndim != 2:
            raise RuntimeError(f"动作块应为 (N,{ACTION_DIM})，实际 {act.shape}")
        if act.shape[-1] != ACTION_DIM:
            raise RuntimeError(f"模型输出 {act.shape[-1]} 维，期望 {ACTION_DIM}")
        return act if self.horizon is None else act[:self.horizon]
