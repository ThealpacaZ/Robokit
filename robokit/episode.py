"""robokit 格式 HDF5 episode 的读取辅助（清洗/可视化共用）。"""
import json

import h5py
import numpy as np


class EpisodeFile:
    """惰性读取：小数据（关节/时间戳）直接取，图像保留 h5py dataset 按需切片。"""

    def __init__(self, path):
        self.path = path
        self.f = h5py.File(path, "r")
        obs = self.f["observations"]
        self.cams = sorted(obs["images"].keys()) if "images" in obs else []
        self.arms = sorted(k for k in obs.keys() if k != "images")
        self.length = self.f["timestamps/frame"].shape[0]

    @property
    def attrs(self):
        return dict(self.f.attrs)

    def has(self, key):
        """dataset 是否存在。清洗阶段要先判存在再读，缺 dataset 不能抛 KeyError。"""
        return key in self.f

    @property
    def freq(self):
        return float(self.f.attrs.get("freq", 30))

    def config(self):
        return json.loads(self.f.attrs.get("config_json", "{}"))

    def joint(self, arm):
        return self.f[f"observations/{arm}/joint"][:]

    def gripper(self, arm):
        """始终返回 1-D (T,)。recorder 写的是 (T,1)，但别处产生的文件可能是 (T,)。"""
        arr = self.f[f"observations/{arm}/gripper"][:]
        return arr[:, 0] if arr.ndim == 2 else np.ravel(arr)

    def eef_pose(self, arm):
        key = f"observations/{arm}/eef_pose"
        return self.f[key][:] if key in self.f else None

    def images(self, cam):
        """返回 h5py dataset (T,H,W,3)，支持切片，不整体载入内存。"""
        return self.f[f"observations/images/{cam}"]

    def frame_ts(self):
        return self.f["timestamps/frame"][:]

    def cam_ts(self, cam, kind="capture"):
        return self.f[f"timestamps/cams/{cam}/{kind}"][:]

    def arm_ts(self, arm):
        return self.f[f"timestamps/arms/{arm}"][:]

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
