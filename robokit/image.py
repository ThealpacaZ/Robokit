"""训练与推理共用的图像几何变换。"""

from __future__ import annotations

import math

import numpy as np


def lememory_preprocess(image, size=(224, 224), mode="train_aligned"):
    """复现 leMemory 训练 loader 送入视觉骨干前的图像几何。

    ``train_aligned`` 先把整帧直接 resize 到目标方图（不保长宽比），再用中心位置
    复现训练期 ``random_resized_crop(scale=[0.9, 0.9], ratio=[1, 1])`` 的期望视野。
    ``deploy`` 仅保留用于复刻上游部署代码多出来的中心方形裁剪。

    参数 ``size`` 使用 PIL 的 ``(width, height)`` 顺序；返回 PIL RGB 图像。
    """
    from PIL import Image

    if mode not in ("deploy", "train_aligned"):
        raise ValueError(f"preprocess mode must be deploy|train_aligned, got {mode!r}")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    else:
        image = image.convert("RGB")

    width, height = (int(value) for value in size)
    if width <= 0 or height <= 0:
        raise ValueError(f"target size must be positive, got {(width, height)}")
    if mode == "deploy":
        source_width, source_height = image.size
        left = min(
            max((source_width - source_height) // 2, 0),
            max(source_width - source_height, 0),
        )
        image = image.crop((left, 0, left + source_height, source_height))

    image = image.resize((width, height), resample=Image.LANCZOS)
    crop_width = int(width * math.sqrt(0.9))
    crop_height = int(height * math.sqrt(0.9))
    margin_width = (width - crop_width) // 2
    margin_height = (height - crop_height) // 2
    image = image.crop(
        (
            margin_width,
            margin_height,
            margin_width + crop_width,
            margin_height + crop_height,
        )
    )
    return image.resize((width, height), resample=Image.LANCZOS)
