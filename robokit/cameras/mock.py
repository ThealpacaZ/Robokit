"""无硬件测试用相机：生成随时间变化的合成图像。"""
import time

import numpy as np

from robokit.cameras.base import Camera


class MockCamera(Camera):
    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.height = int(cfg.get("height", 480))
        self.width = int(cfg.get("width", 640))
        self._t0 = None

    def connect(self):
        self._t0 = time.time()

    def read(self):
        now = time.time()
        t = now - self._t0
        img = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # 随时间平移的渐变背景 + 移动方块，保证相邻帧不同（供冻结帧检查测试）
        gradient = (np.linspace(0, 255, self.width) + t * 40) % 256
        img[:, :, 0] = gradient.astype(np.uint8)[None, :]
        img[:, :, 1] = 128
        x = int((0.5 + 0.4 * np.sin(t)) * (self.width - 60))
        y = int((0.5 + 0.4 * np.cos(t)) * (self.height - 60))
        img[y:y + 60, x:x + 60] = (255, 255, 255)
        return {"image": img, "capture_ts": now, "receive_ts": now}
