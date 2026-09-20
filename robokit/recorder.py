"""HDF5 episode 录制器：后台线程增量写盘，1 个 episode = 1 个文件。

写入期间使用 <path>.tmp，close() 成功后重命名为最终文件名，
因此目录里凡是 *.hdf5 都是完整落盘的 episode，中途崩溃只会留下 .tmp。

文件格式:
    observations/images/{cam}    (T, H, W, 3) uint8
    observations/{arm}/joint     (T, dof)     float32
    observations/{arm}/eef_pose  (T, 6)       float32   # 臂不提供 EEF 位姿时缺省
    observations/{arm}/eef_rotvec (T, 3)      float32   # 同一姿态的旋转向量（弧度），无欧拉 ±180° 翻转
    observations/{arm}/gripper   (T, 1)       float32
    timestamps/frame             (T,)         float64   # 采集主循环 tick 时间 (epoch 秒)
    timestamps/cams/{cam}/capture   (T,)      float64   # 相机侧采集时间 (ROS header stamp)
    timestamps/cams/{cam}/receive   (T,)      float64   # 本机收到该帧的时间
    timestamps/arms/{arm}        (T,)         float64   # 臂状态读取时间
    attrs: task_name, freq, config_json, version
"""
import json
import os
import queue
import threading

import h5py
import numpy as np

from scipy.spatial.transform import Rotation

from robokit.pose import EULER_SEQ
from robokit.utils import log

FORMAT_VERSION = "robokit-1.0"


class EpisodeRecorder:
    def __init__(self, path, task_name, freq, config, flush_every=64):
        self.path = path
        self.tmp_path = path + ".tmp"
        self.attrs = {
            "task_name": task_name,
            "freq": freq,
            "config_json": json.dumps(config, ensure_ascii=False),
            "version": FORMAT_VERSION,
        }
        self.flush_every = flush_every
        self._queue = queue.Queue(maxsize=600)  # 写盘跟不上时反压主循环，绝不悄悄丢帧
        self._file = None
        self._datasets = {}      # dataset 路径 -> h5py.Dataset
        self._buffers = {}       # dataset 路径 -> list[np.ndarray]
        self._written = 0
        self._error = None
        self._thread = threading.Thread(target=self._writer_loop, daemon=True)
        self._thread.start()

    @property
    def num_frames(self):
        return self._written + len(next(iter(self._buffers.values()), []))

    def append(self, frame: dict):
        """frame: {"frame_ts": float, "arms": {...}, "cams": {...}}（见 robot.get_obs + collect）"""
        if self._error:
            raise RuntimeError("recorder writer thread failed") from self._error
        self._queue.put(frame)

    def close(self, discard=False):
        """结束录制。discard=True 时删除文件，否则落盘并重命名为最终文件。返回最终路径或 None。"""
        self._queue.put(None)
        self._thread.join()

        if self._file is not None:
            if not discard:
                self._flush()
                for key, value in self.attrs.items():
                    self._file.attrs[key] = value
            self._file.close()
            self._file = None

        if discard:
            if os.path.exists(self.tmp_path):
                os.remove(self.tmp_path)
            return None
        if self._error:
            raise RuntimeError("recorder writer thread failed") from self._error
        if not os.path.exists(self.tmp_path):
            return None
        os.rename(self.tmp_path, self.path)
        return self.path

    # ---------- 写线程 ----------

    def _writer_loop(self):
        try:
            while True:
                frame = self._queue.get()
                if frame is None:
                    return
                if self._file is None:
                    self._create_datasets(frame)
                self._buffer_frame(frame)
                if len(next(iter(self._buffers.values()))) >= self.flush_every:
                    self._flush()
        except Exception as e:  # 主循环在下一次 append/close 时感知
            self._error = e
            log("recorder", f"writer thread error: {e}", "ERROR")

    def _frame_items(self, frame):
        """把一帧展平为 (dataset 结构, 值) 列表。"""
        items = [("timestamps/frame", np.float64(frame["frame_ts"]))]
        for arm, state in frame["arms"].items():
            items.append((f"observations/{arm}/joint", state["joint"].astype(np.float32)))
            items.append((f"observations/{arm}/gripper", np.array([state["gripper"]], dtype=np.float32)))
            if state["eef_pose"] is not None:
                eef_pose = np.asarray(state["eef_pose"], dtype=np.float64)
                items.append((f"observations/{arm}/eef_pose", eef_pose.astype(np.float32)))
                # 欧拉 roll 在 ±180° 附近会整圈翻转（cover 数据里就有），给吃状态的模型留一份
                # 连续的姿态表示。旋转向量与 eef_pose[3:] 描述同一姿态，按 EULER_SEQ 换算。
                items.append((f"observations/{arm}/eef_rotvec",
                              Rotation.from_euler(EULER_SEQ, eef_pose[3:]).as_rotvec().astype(np.float32)))
            items.append((f"timestamps/arms/{arm}", np.float64(state["ts"])))
        for cam, data in frame["cams"].items():
            items.append((f"observations/images/{cam}", data["image"]))
            items.append((f"timestamps/cams/{cam}/capture", np.float64(data["capture_ts"])))
            items.append((f"timestamps/cams/{cam}/receive", np.float64(data["receive_ts"])))
        return items

    def _create_datasets(self, first_frame):
        self._file = h5py.File(self.tmp_path, "w")
        for key, value in self._frame_items(first_frame):
            value = np.asarray(value)
            shape = value.shape
            self._file.create_dataset(
                key, shape=(0, *shape), maxshape=(None, *shape), dtype=value.dtype,
                chunks=(1, *shape) if value.ndim >= 3 else None,
            )
            self._datasets[key] = self._file[key]
            self._buffers[key] = []

    def _buffer_frame(self, frame):
        for key, value in self._frame_items(frame):
            self._buffers[key].append(np.asarray(value))

    def _flush(self):
        n = len(next(iter(self._buffers.values()), []))
        if n == 0:
            return
        for key, buf in self._buffers.items():
            ds = self._datasets[key]
            ds.resize(self._written + n, axis=0)
            ds[self._written:self._written + n] = np.stack(buf, axis=0)
            buf.clear()
        self._written += n
        self._file.flush()


def next_episode_index(task_dir):
    """扫描目录中已有的 N.hdf5，返回下一个可用编号。

    已发布的 episode 被 publish_batch.sh 搬进了批次目录，任务目录里就看不见了。
    只看现存文件的话编号会退回 0，跟远端已有的段撞名（同名不同内容）。所以还要看
    <任务目录>/.next_index —— 发布脚本在搬走文件前写进去的编号水位。
    """
    import re
    existing = []
    if os.path.isdir(task_dir):
        for fname in os.listdir(task_dir):
            m = re.fullmatch(r"(\d+)\.hdf5", fname)
            if m:
                existing.append(int(m.group(1)))
    nxt = max(existing) + 1 if existing else 0

    marker = os.path.join(task_dir, ".next_index")
    if os.path.exists(marker):
        try:
            with open(marker, "r", encoding="utf-8") as f:
                nxt = max(nxt, int(f.read().strip()))
        except (ValueError, OSError):
            pass                                  # 水位文件坏了就退回按现存文件编号
    return nxt
