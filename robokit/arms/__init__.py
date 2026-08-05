"""机械臂类型注册表。新增机械臂：实现 Arm 子类后在此注册。

工厂函数按需导入，未用到的臂类型不需要安装其 SDK。
"""
from robokit.arms.base import Arm


def _piper(name, cfg):
    from robokit.arms.piper import PiperArm
    return PiperArm(name, cfg)


def _mock(name, cfg):
    from robokit.arms.mock import MockArm
    return MockArm(name, cfg)


ARM_TYPES = {
    "piper": _piper,
    "mock": _mock,
}


def create_arm(name, cfg) -> Arm:
    arm_type = cfg["type"]
    if arm_type not in ARM_TYPES:
        raise ValueError(f"unknown arm type '{arm_type}', available: {list(ARM_TYPES)}")
    return ARM_TYPES[arm_type](name, cfg)
