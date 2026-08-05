"""robokit HDF5 → RLDS 的 TFDS builder。

数据来源（二选一）:
    export ROBOKIT_DATA_DIR=/path/to/datasets/任务名   # 推荐
    或在本目录放一个 data/ 符号链接指向任务目录

动作空间（环境变量 ROBOKIT_ACTION_MODE）:
    eef_delta（默认）: state = 每臂 [eef_pose(6), gripper]，action = 每臂 [局部坐标系 delta_pose(6), 下一帧 gripper]
    joint            : state = 每臂 [joint(dof), gripper]，action = 下一帧 state
多臂时按臂名排序后拼接。episode 有 T 帧 → 生成 T-1 个 step。
位姿单位一律是米 / 真弧度 / xyz 外旋欧拉序，与部署端 robokit/pose.py 严格一致。

相机命名映射: cam_high → image, cam_wrist → wrist_image, 其余用原名。

测试时可 export NO_USE_EMBED=1 跳过 Universal Sentence Encoder 下载（嵌入用零向量）。

转换前会强制读 <数据目录>/clean_report.json：目录里仍存在被清洗判为 bad 的 episode、
报告过期（有文件没被检查过）、或存在跨 episode 形状不一致时直接拒绝转换。
设 ROBOKIT_ALLOW_UNCLEAN=1 可跳过该闸。

构建:
    cd robokit/rlds/robokit_dataset && tfds build --overwrite
输出: ~/tensorflow_datasets/robokit_dataset/
"""
import glob
import json
import os
from typing import Any, Iterator, Tuple

import h5py
import numpy as np
import tensorflow_datasets as tfds


def _data_dir():
    env = os.environ.get("ROBOKIT_DATA_DIR")
    if env:
        return env
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def _require_clean(data_dir, files):
    """转换前强制检查 clean.py 的报告：清洗判为 bad 的数据不许进训练集。

    没有这道闸，清洗只是"生成了一份报告"，坏数据照样能转成 RLDS 并训练。
    ROBOKIT_ALLOW_UNCLEAN=1 可显式跳过（用于明知有问题仍要做对照实验的场合）。
    """
    if os.environ.get("ROBOKIT_ALLOW_UNCLEAN"):
        print("[rlds] ROBOKIT_ALLOW_UNCLEAN=1 → 跳过清洗报告检查")
        return

    report_path = os.path.join(data_dir, "clean_report.json")
    hint = ("先跑 `python scripts/clean.py --data <目录> --quarantine`；"
            "确实要带着已知问题转换就设 ROBOKIT_ALLOW_UNCLEAN=1")
    if not os.path.exists(report_path):
        raise RuntimeError(f"{data_dir} 没有 clean_report.json，数据未经清洗。{hint}")
    with open(report_path, "r", encoding="utf-8") as f:
        report = json.load(f)

    episodes = report.get("episodes", report)     # 兼容早期扁平格式
    names = [os.path.basename(p) for p in files]
    bad = [n for n in names if episodes.get(n, {}).get("status") == "bad"]
    unchecked = [n for n in names if n not in episodes]
    ds_errors = report.get("dataset", {}).get("errors", [])

    problems = []
    if bad:
        problems.append(f"{len(bad)} 段被清洗判为 bad 且还在目录里（未隔离）: {bad[:5]}")
    if unchecked:
        problems.append(f"{len(unchecked)} 段不在清洗报告里（报告已过期）: {unchecked[:5]}")
    if ds_errors:
        problems.append(f"跨 episode 一致性错误: {ds_errors[:3]}")
    if problems:
        raise RuntimeError("拒绝转换：\n  - " + "\n  - ".join(problems) + f"\n{hint}")
    print(f"[rlds] 清洗报告检查通过：{len(names)} 段全部为 ok/warn")


# 欧拉序必须与 robokit.pose.EULER_SEQ 一致（小写 = 固定轴外旋），否则训练动作与部署端
# 执行的动作定义不同 —— 这类不一致在数字孪生里完全自洽，只能靠 URDF 正运动学发现。
_EULER_SEQ = "xyz"


def _local_delta_pose(base_pose, target_pose):
    """base_pose 局部坐标系下的位姿增量（与 robokit/pose.py 一致，此处内联避免跨包依赖）。"""
    from scipy.spatial.transform import Rotation

    base = np.asarray(base_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)
    base_rot = Rotation.from_euler(_EULER_SEQ, base[3:])
    target_rot = Rotation.from_euler(_EULER_SEQ, target[3:])
    delta_rpy = (base_rot.inv() * target_rot).as_euler(_EULER_SEQ)
    delta_xyz = base_rot.inv().apply(target[:3] - base[:3])
    return np.concatenate([delta_xyz, delta_rpy])


def _cam_feature_name(cam):
    return {"cam_high": "image", "cam_wrist": "wrist_image"}.get(cam, cam)


class RobokitDataset(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("1.1.0")
    RELEASE_NOTES = {
        "1.0.0": "robokit HDF5 (1 file = 1 episode) initial release.",
        "1.1.0": "Euler convention fixed to xyz extrinsic (was XYZ intrinsic). "
                 "eef_delta rotations differ numerically from 1.0.0 — do not mix.",
    }

    def __init__(self, *args, **kwargs):
        self._action_mode = os.environ.get("ROBOKIT_ACTION_MODE", "eef_delta")
        self._probe()  # 先探测 shape，_info() 需要静态形状
        if os.environ.get("NO_USE_EMBED"):
            self._embed = lambda texts: [np.zeros(512, dtype=np.float32) for _ in texts]
        else:
            import tensorflow_hub as hub
            use = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
            self._embed = lambda texts: [t.numpy() for t in use(texts)]
        super().__init__(*args, **kwargs)

    def _probe(self):
        """从第一个 episode 探测相机分辨率与状态/动作维度。"""
        files = sorted(glob.glob(os.path.join(_data_dir(), "*.hdf5")))
        if not files:
            raise FileNotFoundError(
                f"no .hdf5 in {_data_dir()}; set ROBOKIT_DATA_DIR to your task directory")
        _require_clean(_data_dir(), files)
        with h5py.File(files[0], "r") as f:
            obs = f["observations"]
            self._cams = sorted(obs["images"].keys()) if "images" in obs else []
            self._arms = sorted(k for k in obs.keys() if k != "images")
            self._img_shapes = {cam: obs["images"][cam].shape[1:] for cam in self._cams}
            dim = 0
            for arm in self._arms:
                if self._action_mode == "eef_delta":
                    if f"observations/{arm}/eef_pose" not in f:
                        raise ValueError(f"arm '{arm}' has no eef_pose; use ROBOKIT_ACTION_MODE=joint")
                    dim += 7
                else:
                    dim += obs[arm]["joint"].shape[1] + 1
            self._state_dim = dim

    def _info(self) -> tfds.core.DatasetInfo:
        image_features = {
            _cam_feature_name(cam): tfds.features.Image(
                shape=self._img_shapes[cam], dtype=np.uint8, encoding_format="png")
            for cam in self._cams
        }
        return self.dataset_info_from_configs(
            features=tfds.features.FeaturesDict({
                "steps": tfds.features.Dataset({
                    "observation": tfds.features.FeaturesDict({
                        **image_features,
                        "state": tfds.features.Tensor(shape=(self._state_dim,), dtype=np.float32),
                    }),
                    "action": tfds.features.Tensor(shape=(self._state_dim,), dtype=np.float32),
                    "discount": tfds.features.Scalar(dtype=np.float32),
                    "reward": tfds.features.Scalar(dtype=np.float32),
                    "is_first": tfds.features.Scalar(dtype=np.bool_),
                    "is_last": tfds.features.Scalar(dtype=np.bool_),
                    "is_terminal": tfds.features.Scalar(dtype=np.bool_),
                    "language_instruction": tfds.features.Text(),
                    "language_embedding": tfds.features.Tensor(shape=(512,), dtype=np.float32),
                }),
                "episode_metadata": tfds.features.FeaturesDict({
                    "file_path": tfds.features.Text(),
                }),
            }))

    def _split_generators(self, dl_manager):
        return {"train": self._generate_examples(_data_dir())}

    def _episode_states_actions(self, f):
        """返回 (states (T-1, D), actions (T-1, D))。"""
        states, actions = [], []
        for arm in self._arms:
            gripper = f[f"observations/{arm}/gripper"][:, 0]
            if self._action_mode == "eef_delta":
                eef = f[f"observations/{arm}/eef_pose"][:]
                deltas = np.stack([_local_delta_pose(eef[i], eef[i + 1])
                                   for i in range(len(eef) - 1)])
                states.append(np.concatenate([eef[:-1], gripper[:-1, None]], axis=1))
                actions.append(np.concatenate([deltas, gripper[1:, None]], axis=1))
            else:
                joint = f[f"observations/{arm}/joint"][:]
                sa = np.concatenate([joint, gripper[:, None]], axis=1)
                states.append(sa[:-1])
                actions.append(sa[1:])
        return (np.concatenate(states, axis=1).astype(np.float32),
                np.concatenate(actions, axis=1).astype(np.float32))

    def _generate_examples(self, path) -> Iterator[Tuple[str, Any]]:
        embed_cache = {}
        for hdf5_path in sorted(glob.glob(os.path.join(path, "*.hdf5"))):
            with h5py.File(hdf5_path, "r") as f:
                instruction = f.attrs.get("task_name", "pick up something")
                if instruction not in embed_cache:
                    embed_cache[instruction] = self._embed([instruction])[0]
                embedding = embed_cache[instruction]

                states, actions = self._episode_states_actions(f)
                images = {cam: f[f"observations/images/{cam}"][:-1] for cam in self._cams}
                T = states.shape[0]

                steps = []
                for i in range(T):
                    steps.append({
                        "observation": {
                            **{_cam_feature_name(cam): images[cam][i] for cam in self._cams},
                            "state": states[i],
                        },
                        "action": actions[i],
                        "discount": 1.0,
                        "reward": float(i == T - 1),
                        "is_first": i == 0,
                        "is_last": i == T - 1,
                        "is_terminal": i == T - 1,
                        "language_instruction": instruction,
                        "language_embedding": embedding,
                    })

            yield os.path.basename(hdf5_path), {
                "steps": steps,
                "episode_metadata": {"file_path": hdf5_path},
            }
