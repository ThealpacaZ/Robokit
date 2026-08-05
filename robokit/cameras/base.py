"""相机抽象基类。"""
import time
from abc import ABC, abstractmethod


class Camera(ABC):
    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg

    @abstractmethod
    def connect(self):
        ...

    @abstractmethod
    def read(self) -> dict | None:
        """返回最新帧，尚未收到任何帧时返回 None。

        {"image": (H,W,3) uint8 RGB,
         "capture_ts": float,   # 相机侧采集时间（ROS header stamp），epoch 秒
         "receive_ts": float}   # 本机收到该帧的时间，epoch 秒
        """
        ...

    def supports_frame_clock(self) -> bool:
        """本后端能否把采集节拍锁到相机自己的帧到达事件上（见 wait_for_frame）。"""
        return False

    def wait_for_frame(self, after_seq: int, timeout: float) -> dict | None:
        """阻塞到出现序号 > after_seq 的新帧并返回它（帧内含 "seq"）；超时/断流返回 None。

        只有 supports_frame_clock() 为真的后端需要实现。
        """
        raise NotImplementedError(
            f"{type(self).__name__} 不支持帧时钟，采集需回退到主机时钟节拍")

    def last_receive_ts(self) -> float:
        frame = self.read()
        return 0.0 if frame is None else frame["receive_ts"]

    def alive(self, timeout: float) -> bool:
        """timeout 秒内收到过新帧则认为相机存活。"""
        return time.time() - self.last_receive_ts() < timeout

    def disconnect(self):
        pass
