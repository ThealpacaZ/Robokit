"""Intel RealSense 彩色相机直连后端（pyrealsense2）。

后台线程持续读取硬件帧，主循环的 ``read()`` 只返回最新缓存。相比经 ROS2
发布/订阅 ``sensor_msgs/Image``，直连路径避免 DDS 序列化和 cv_bridge 转换，
同时保留相机的全局采集时间戳。
"""
import threading
import time

import numpy as np

from robokit.cameras.base import Camera
from robokit.utils import log


class RealSenseCamera(Camera):
    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.serial = str(cfg.get("serial", ""))
        self.width = int(cfg.get("width", 640))
        self.height = int(cfg.get("height", 480))
        self.fps = int(cfg.get("fps", 60))
        self.enable_auto_exposure = bool(cfg.get("enable_auto_exposure", True))
        self.enable_auto_white_balance = bool(
            cfg.get("enable_auto_white_balance", True)
        )
        self.auto_exposure_priority = bool(cfg.get("auto_exposure_priority", False))

        self._rs = None
        self._pipeline = None
        self._profile = None
        self._thread = None
        self._stop = threading.Event()
        # Condition 而不是 Lock：采集线程每落一帧要唤醒 wait_for_frame 的等待者。
        # `with self._lock:` 的语义不变，多出来的只是 wait/notify。
        self._lock = threading.Condition()
        self._latest = None
        self._error = None
        self._clock_offset = None
        self._seq = -1            # 已发布帧的序号，-1 = 还没有任何帧

    def connect(self):
        import pyrealsense2 as rs

        with self._lock:
            self._latest = None
            self._error = None
            self._seq = -1
        self._clock_offset = None
        self._stop.clear()
        self._rs = rs
        self._pipeline = rs.pipeline()
        config = rs.config()
        if self.serial:
            config.enable_device(self.serial)
        config.enable_stream(
            rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps
        )
        self._profile = self._pipeline.start(config)

        # global_time 让 frame.get_timestamp() 直接使用 epoch 毫秒，能够与机械臂的
        # time.time() 时间戳比较。关闭 auto_exposure_priority，避免暗光下主动降 FPS。
        sensor = self._profile.get_device().first_color_sensor()
        if sensor.supports(rs.option.global_time_enabled):
            sensor.set_option(rs.option.global_time_enabled, 1.0)
        if sensor.supports(rs.option.enable_auto_exposure):
            sensor.set_option(
                rs.option.enable_auto_exposure,
                1.0 if self.enable_auto_exposure else 0.0,
            )
        if sensor.supports(rs.option.enable_auto_white_balance):
            sensor.set_option(
                rs.option.enable_auto_white_balance,
                1.0 if self.enable_auto_white_balance else 0.0,
            )
        if sensor.supports(rs.option.auto_exposure_priority):
            sensor.set_option(
                rs.option.auto_exposure_priority,
                1.0 if self.auto_exposure_priority else 0.0,
            )

        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        serial = f", serial={self.serial}" if self.serial else ""
        log(
            self.name,
            f"RealSense direct {self.width}x{self.height}@{self.fps} RGB8{serial}",
            "INFO",
        )

    def _capture_loop(self):
        rs = self._rs
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(timeout_ms=1000)
                frame = frames.get_color_frame()
                if not frame:
                    continue

                # RealSense 的 frame buffer 会被后续采集复用，必须复制后再发布到缓存。
                image = np.asanyarray(frame.get_data()).copy()
                receive_ts = time.time()
                device_ts = frame.get_timestamp() * 1e-3
                if frame.get_frame_timestamp_domain() == rs.timestamp_domain.global_time:
                    capture_ts = device_ts
                else:
                    # 个别固件不支持 global_time：用首帧建立硬件时钟到 epoch 的映射。
                    if self._clock_offset is None:
                        self._clock_offset = receive_ts - device_ts
                    capture_ts = device_ts + self._clock_offset

                with self._lock:
                    self._seq += 1
                    self._latest = {
                        "image": image,
                        "capture_ts": capture_ts,
                        "receive_ts": receive_ts,
                        "seq": self._seq,
                    }
                    self._lock.notify_all()
            except RuntimeError as exc:
                if not self._stop.is_set():
                    with self._lock:
                        self._error = exc
                        # 采集线程要退出了，必须唤醒等待者，否则它们会一直阻塞到超时
                        self._lock.notify_all()
                    log(self.name, f"RealSense capture failed: {exc}", "ERROR")
                return

    def read(self):
        with self._lock:
            return self._latest

    def supports_frame_clock(self):
        return True

    def wait_for_frame(self, after_seq, timeout):
        """阻塞到出现序号 > ``after_seq`` 的帧，返回该帧（已含 seq）；超时或采集线程已死返回 None。

        采集循环用它把节拍锁到相机自己的晶振上。主机时钟和相机晶振是两个独立时钟
        （实测相机 59.53Hz、主机节拍 30.00Hz，比值 1.985 不是整数），按主机时钟睡固定
        周期取「当前最新帧」，每个 tick 前进的传感器帧数就会在 1/2/3 之间跳，图像的真实
        间隔随之在 16.8/33.6/50.4ms 之间跳，而 frame_ts 却一律声称 33.3ms。等新帧而不是
        等时钟，间隔才是严格的传感器周期整数倍。
        """
        deadline = time.time() + timeout
        with self._lock:
            while self._seq <= after_seq:
                if self._error is not None or self._stop.is_set():
                    return None
                if not self._lock.wait(timeout=max(0.0, deadline - time.time())):
                    return None
            return self._latest

    @property
    def error(self):
        """后台捕获线程最近一次异常；正常运行或未连接时为 ``None``。"""
        with self._lock:
            return self._error

    def disconnect(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError as exc:
                log(self.name, f"RealSense stop warning: {exc}", "WARNING")
            self._pipeline = None
        self._profile = None
        with self._lock:
            self._latest = None
            self._error = None
            self._lock.notify_all()
