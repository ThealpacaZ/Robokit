"""ROS2 相机：订阅 sensor_msgs/Image 话题，后台 executor 线程持续 spin。

相比旧实现（采集循环里 spin_once(0.001) 拉取）的关键改进：
- 独立 spin 线程，回调即时解码，read() 永远拿到已解码的最新帧，主循环零阻塞；
- 记录 ROS header stamp（相机侧采集时间）与本机接收时间，供清洗阶段做图像-动作对齐检查。
"""
import re
import threading
import time

import rclpy
from cv_bridge import CvBridge
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image

from robokit.cameras.base import Camera
from robokit.utils import log

_rclpy_lock = threading.Lock()
_rclpy_inited = False


def _ensure_rclpy():
    global _rclpy_inited
    with _rclpy_lock:
        if not _rclpy_inited:
            rclpy.init()
            _rclpy_inited = True


class ROS2Camera(Camera):
    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self.topic = cfg["topic"]
        self._bridge = CvBridge()
        self._lock = threading.Lock()
        self._latest = None
        self._node = None
        self._executor = None
        self._spin_thread = None

    def connect(self):
        _ensure_rclpy()
        node_name = "robokit_" + re.sub(r"\W", "_", self.name)
        self._node = Node(node_name)
        qos = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                         history=HistoryPolicy.KEEP_LAST, depth=1)
        self._node.create_subscription(Image, self.topic, self._on_image, qos)

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._node)
        self._spin_thread = threading.Thread(target=self._executor.spin, daemon=True)
        self._spin_thread.start()
        log(self.name, f"subscribed to {self.topic}", "INFO")

    def _on_image(self, msg):
        image = self._bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        capture_ts = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        frame = {"image": image, "capture_ts": capture_ts, "receive_ts": time.time()}
        with self._lock:
            self._latest = frame

    def read(self):
        with self._lock:
            return self._latest

    def disconnect(self):
        if self._executor is not None:
            self._executor.shutdown()
        if self._node is not None:
            self._node.destroy_node()
