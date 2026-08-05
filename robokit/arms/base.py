"""机械臂抽象基类：任意自由度，关节 + 可选 EEF 位姿 + 夹爪。"""
import time
from abc import ABC, abstractmethod

import numpy as np


class ArmCommandRejected(RuntimeError):
    """机械臂在发送运动帧前拒绝一条不安全或不可执行的命令。"""

    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = str(kind)


class Arm(ABC):
    """所有单位统一：关节弧度、位姿米/弧度([x,y,z,roll,pitch,yaw])、夹爪开合度 [0,1]。"""

    def __init__(self, name, cfg):
        self.name = name
        self.cfg = cfg
        self.dof = int(cfg["dof"])

    @abstractmethod
    def connect(self, read_only=False):
        """建立连接。

        read_only=True 时实现方不得向硬件发送任何控制帧（遥操作采集：主臂在驱动从臂，
        本进程只是旁路读取，抢总线会把从臂从联动里拽出来）。
        """
        ...

    @abstractmethod
    def get_state(self) -> dict:
        """返回 {"joint": (dof,), "eef_pose": (6,) 或 None, "gripper": float, "ts": float}。

        ts 为读取时刻的 time.time()，用于清洗阶段的图像-动作对齐检查。
        """
        ...

    @abstractmethod
    def move_joint(self, joint, gripper=None):
        """关节位置控制。joint: (dof,) 弧度；gripper: [0,1] 或 None（不动夹爪）。"""
        ...

    def move_eef(self, pose, gripper=None):
        """EEF 位姿控制（可选实现）。pose: (6,) 米/弧度。"""
        raise NotImplementedError(f"{type(self).__name__} does not support EEF control")

    def disconnect(self):
        pass

    def _stamped(self, joint, eef_pose, gripper) -> dict:
        return {
            "joint": np.asarray(joint, dtype=np.float64),
            "eef_pose": None if eef_pose is None else np.asarray(eef_pose, dtype=np.float64),
            "gripper": float(gripper),
            "ts": time.time(),
        }
