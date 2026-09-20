"""看「模型端拿到的图像」。

部署链路上图像要经过三道手：相机取帧 → 网络送到服务端 → policy 预处理后进模型。
排查「动作诡异」时，第一个要排除的就是模型看到的画面本身不对（视角变了、亮度变了、
裁剪把物体切掉了、送错相机）。本模块把这三道手上的画面显示出来或落盘。

两端都能用：
  * 真机端（有显示器）：``show=True`` 开窗实时看自己发出去的帧。
  * 服务端（AutoDL 无显示器）：``save_dir=...`` 落 PNG，事后 scp 回来看。

服务端能落两种图：
  ``recv``   网络上收到的原始帧，和真机发出的逐字节相同。
  ``model``  交给 ``policy.predict_action_chunk`` 的那个张量反解成图，即 LeRobot
             preprocessor 之后的结果（归一化已生效）。注意 PI0/PI0.5 的 resize 发生
             在模型 forward 内部，所以这张图的分辨率通常仍等于收到的帧；它回答的是
             「送进模型的像素内容对不对」，不是「模型内部最后那层 224×224 长什么样」。
"""
from __future__ import annotations

from pathlib import Path
from threading import Event, Lock, Thread

import numpy as np

from robokit.utils import log


def tensor_to_uint8(tensor) -> tuple[np.ndarray, tuple[float, float]]:
    """把预处理后的图像张量反解成可看的 RGB uint8，并回报原始数值范围。

    归一化区间不做假设，按实测数值范围判断；原始 (min, max) 一并返回交给调用方记
    日志 —— 猜错了画面会明显偏色/偏暗，比静默输出一张“看起来对”的图更容易发现。
    """
    array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    array = np.asarray(array, dtype=np.float32)
    while array.ndim > 3:                      # 去掉 batch 维
        array = array[0]
    if array.ndim != 3:
        raise ValueError(f"expected a 3-D image tensor, got shape {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))  # CHW → HWC
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    low, high = float(np.nanmin(array)), float(np.nanmax(array))
    if low < -0.01:                            # [-1, 1]
        array = (array + 1.0) * 127.5
    elif high <= 1.01:                         # [0, 1]
        array = array * 255.0
    return np.clip(array, 0, 255).astype(np.uint8), (low, high)


class ObsView:
    """显示 / 落盘部署链路上的图像。``every`` 控制抽样，避免拖慢控制频率。

    看图是诊断功能，**绝不允许占用控制线程的时间预算**：RTC 下调用方每多花 10ms
    就少 0.3 步的推理余量，实测 imshow+waitKey 在 XWayland 上要 170-400ms，足以
    把 d 顶出 s_min 触发 deadline abort。因此 ``images()`` 只把帧放进一个槽位就
    返回，真正的 cvtColor/imwrite/imshow 全部在后台线程做；来不及显示的帧直接丢
    弃（要的是「现在长什么样」，不是每帧都看到）。cv2 的 GUI 调用全部只发生在这
    一个后台线程里。
    """

    def __init__(self, show=False, save_dir=None, every=1, tag="obs"):
        self.show = bool(show)
        self.save_dir = None if save_dir in (None, "") else Path(save_dir)
        self.every = max(1, int(every))
        self.tag = str(tag)
        self._cv2 = None
        self._windows = set()
        self._saved = 0
        self._dropped = 0
        self._slot = None
        self._slot_lock = Lock()
        self._wake = Event()
        self._stop = Event()
        self._worker = None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            log(self.tag, f"图像落盘 → {self.save_dir}（每 {self.every} 次一张）", "INFO")
        if self.show:
            log(self.tag, f"图像实时显示：开（每 {self.every} 次一帧，后台线程）", "INFO")

    @property
    def enabled(self) -> bool:
        return self.show or self.save_dir is not None

    def _lazy_cv2(self):
        if self._cv2 is None:
            import cv2

            self._cv2 = cv2
        return self._cv2

    def _wants(self, index: int) -> bool:
        return self.enabled and int(index) % self.every == 0

    def images(self, images: dict, index: int, kind="recv") -> None:
        """处理一组 {相机名: RGB uint8 图}。调用方线程只做一次拷贝就返回。"""
        if not self._wants(index):
            return
        frames = {}
        for name, image in images.items():
            frame = np.asarray(image)
            if frame.ndim != 3 or frame.shape[-1] != 3:
                continue
            # 拷贝：调用方（相机/观测字典）随时可能复用这块缓冲。
            frames[name] = np.array(frame, dtype=np.uint8, copy=True)
        if not frames:
            return
        self._ensure_worker()
        with self._slot_lock:
            if self._slot is not None:
                self._dropped += 1
            self._slot = (frames, int(index), str(kind))
        self._wake.set()

    def _ensure_worker(self) -> None:
        if self._worker is None:
            self._worker = Thread(target=self._pump, name=f"{self.tag}-view", daemon=True)
            self._worker.start()

    def _pump(self) -> None:
        while not self._stop.is_set():
            if not self._wake.wait(timeout=0.2):
                # 没有新帧也要给 GUI 一次事件处理，否则窗口拖动/重绘会卡住。
                self._drain_gui()
                continue
            self._wake.clear()
            with self._slot_lock:
                item, self._slot = self._slot, None
            if item is None:
                continue
            try:
                self._render(*item)
            except Exception as exc:  # 看图绝不能打断控制循环
                log(self.tag, f"显示失败：{type(exc).__name__}: {exc}", "WARNING")

    def _render(self, frames: dict, index: int, kind: str) -> None:
        cv2 = self._lazy_cv2()
        for name, frame in frames.items():
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.save_dir is not None:
                path = self.save_dir / f"{kind}-{index:06d}-{name}.png"
                cv2.imwrite(str(path), bgr)
                self._saved += 1
            if self.show:
                title = f"{kind}:{name}"
                self._windows.add(title)
                cv2.imshow(title, bgr)
        self._drain_gui()

    def _drain_gui(self) -> None:
        if self.show and self._windows and self._cv2 is not None:
            self._cv2.waitKey(1)

    def tensor(self, tensor, index: int, name="cam", kind="model") -> None:
        """处理一张预处理后的图像张量（真正进模型的那张）。"""
        if not self._wants(index):
            return
        try:
            image, (low, high) = tensor_to_uint8(tensor)
        except Exception as exc:  # 看图是诊断功能，绝不能因此打断推理
            log(self.tag, f"图像张量反解失败：{type(exc).__name__}: {exc}", "WARNING")
            return
        if int(index) % (self.every * 20) == 0:
            log(self.tag, f"{kind}:{name} 张量数值范围 [{low:.3f}, {high:.3f}]", "DEBUG")
        self.images({name: image}, index, kind=kind)

    def close(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)
            self._worker = None
        if self.save_dir is not None and self._saved:
            log(self.tag, f"共落盘 {self._saved} 张图 → {self.save_dir}", "INFO")
        if self._dropped:
            log(self.tag, f"显示跟不上，丢弃 {self._dropped} 帧（不影响控制）", "INFO")
        if self._windows and self._cv2 is not None:
            try:
                self._cv2.destroyAllWindows()
            except Exception:
                pass
            self._windows.clear()
