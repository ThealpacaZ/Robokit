"""无硬件测试用机械臂：平滑正弦运动，支持任意自由度。"""
import time

import numpy as np

from robokit.arms.base import Arm


class MockArm(Arm):
    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        self._t0 = None

    def connect(self, read_only=False):
        self._t0 = time.time()

    def get_state(self):
        t = time.time() - self._t0
        phase = np.arange(self.dof) * 0.7
        joint = 0.3 * np.sin(0.5 * t + phase)
        eef_pose = np.array([0.3 + 0.05 * np.sin(0.5 * t), 0.05 * np.cos(0.5 * t), 0.25,
                             0.0, 0.1 * np.sin(0.3 * t), 0.0])
        gripper = 0.5 + 0.5 * np.sin(0.8 * t)
        return self._stamped(joint, eef_pose, gripper)

    def move_joint(self, joint, gripper=None):
        pass

    def move_eef(self, pose, gripper=None):
        pass
