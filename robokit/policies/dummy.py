"""联调用 policy：回发当前状态（joint 模式下机械臂保持不动）。"""
import numpy as np


class DummyPolicy:
    def __init__(self, horizon=10):
        self.horizon = int(horizon)

    def reset(self):
        pass

    def infer(self, obs):
        state = []
        for arm in sorted(obs["state"]):
            s = obs["state"][arm]
            state.extend(list(np.asarray(s["joint"]).ravel()) + [s["gripper"]])
        return np.tile(np.array(state, dtype=np.float32), (self.horizon, 1))
