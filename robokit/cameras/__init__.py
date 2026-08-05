"""相机类型注册表。工厂函数按需导入：未配置 ros2 相机时不需要 rclpy/cv_bridge。"""
from robokit.cameras.base import Camera


def _ros2(name, cfg):
    from robokit.cameras.ros2_camera import ROS2Camera
    return ROS2Camera(name, cfg)


def _realsense(name, cfg):
    from robokit.cameras.realsense_camera import RealSenseCamera
    return RealSenseCamera(name, cfg)


def _mock(name, cfg):
    from robokit.cameras.mock import MockCamera
    return MockCamera(name, cfg)


CAM_TYPES = {
    "ros2": _ros2,
    "realsense": _realsense,
    "mock": _mock,
}


def create_camera(name, cfg) -> Camera:
    cam_type = cfg["type"]
    if cam_type not in CAM_TYPES:
        raise ValueError(f"unknown camera type '{cam_type}', available: {list(CAM_TYPES)}")
    return CAM_TYPES[cam_type](name, cfg)
