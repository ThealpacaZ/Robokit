"""清洗判据自检：合成一段干净 episode，逐类注入已知缺陷，断言 clean.py 判成预期结果。

为什么需要：`clean.py` 是采集→训练之间**唯一**的自动判据，它漏判等于坏数据直接进训练集，
它崩掉等于整批数据没人看。所以要像 review 的 G4 安全闸那样，用「先注入缺陷、再看判据抓不抓到」
来验证判据本身，而不是拿判据去验证数据。

干净底样由真正的 `EpisodeRecorder` 写出（不需要机械臂/相机），因此本脚本同时锁住
「recorder 写出的格式能被 clean.py 正常读」这条契约。

用法:
    python scripts/test_clean.py            # 退出码 0 = 全部符合预期
    python scripts/test_clean.py --keep     # 保留临时目录便于排查
"""
import argparse
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import clean  # noqa: E402  scripts/clean.py

from robokit.recorder import EpisodeRecorder  # noqa: E402
from robokit.utils import log  # noqa: E402

T = 90
FREQ = 30.0
H, W = 96, 128
PIPELINE_DELAY = 0.005   # 相机采集→本机收到，5ms
ARM_READ_DELAY = 0.001   # tick 开始→读到臂状态，1ms


def write_clean_episode(path):
    """用真正的 recorder 写一段各项判据都应通过的 episode（确定性，无硬件、无 sleep）。"""
    rng = np.random.default_rng(0)
    recorder = EpisodeRecorder(path, "selftest task", FREQ, {"robot": {}}, flush_every=32)
    t0 = 1_700_000_000.0
    for i in range(T):
        t = i / FREQ
        frame_ts = t0 + t
        # 噪声底 + 移动方块：纹理足够（不触发 flat/blur），且相邻帧绝不逐字节相同
        img = rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)
        x = int((0.5 + 0.4 * np.sin(2 * t)) * (W - 16))
        y = int((0.5 + 0.4 * np.cos(2 * t)) * (H - 16))
        img[y:y + 16, x:x + 16] = 255
        recorder.append({
            "frame_ts": frame_ts,
            "arms": {"right_arm": {
                "joint": 0.3 * np.sin(0.5 * t + np.arange(6) * 0.7),
                "eef_pose": np.array([0.3 + 0.05 * np.sin(0.5 * t), 0.05 * np.cos(0.5 * t), 0.25,
                                      0.0, 0.1 * np.sin(0.3 * t), 0.0]),
                "gripper": 0.5 + 0.5 * np.sin(0.8 * t),
                "ts": frame_ts + ARM_READ_DELAY,
            }},
            "cams": {"cam_high": {
                "image": img,
                "capture_ts": frame_ts - PIPELINE_DELAY,
                "receive_ts": frame_ts,
            }},
        })
    assert recorder.close() == path, "recorder 没有落盘到预期路径"


# ---------- 缺陷注入 ----------
# 每项 = (注入函数, 期望 status, 期望出现的判据关键字)

def nan_joint(f):
    f["observations/right_arm/joint"][10, 2] = np.nan


def nan_eef(f):
    f["observations/right_arm/eef_pose"][10, 1] = np.nan


def inf_eef_rot(f):
    f["observations/right_arm/eef_pose"][20, 4] = np.inf


def eef_len_mismatch(f):
    eef = f["observations/right_arm/eef_pose"][:-5]
    del f["observations/right_arm/eef_pose"]
    f.create_dataset("observations/right_arm/eef_pose", data=eef)


def missing_gripper(f):
    del f["observations/right_arm/gripper"]


def missing_capture_ts(f):
    del f["timestamps/cams/cam_high/capture"]


def gripper_1d(f):
    """(T,) 而非 (T,1)：别处产生的合法变体，不该崩也不该误判。"""
    g = f["observations/right_arm/gripper"][:, 0]
    del f["observations/right_arm/gripper"]
    f.create_dataset("observations/right_arm/gripper", data=g)


def gripper_oob(f):
    # 必须是**物理上不可能**的开度：判据按毫米设界（GRIPPER_SANE_MM），实测全行程
    # 99.6mm，1.6 (=112mm) 这种"比名义行程大但仍可能"的值现在只判 warning
    # （gripper_clip_loss），不是坏数据。2.5 = 175mm，只可能是归一化常数错了或读数损坏。
    f["observations/right_arm/gripper"][30:35, 0] = 2.5


def joint_jump(f):
    d = f["observations/right_arm/joint"]
    d[40:, :] = d[40:, :] + 1.0


def eef_xyz_jump(f):
    d = f["observations/right_arm/eef_pose"]
    d[50:, :3] = d[50:, :3] + 0.3


def eef_rot_jump(f):
    """真实姿态突变 2.5 rad（143°/tick）。"""
    d = f["observations/right_arm/eef_pose"]
    d[55:, 3:] = d[55:, 3:] + 2.5


def rot_joint_inconsistent(f):
    """位姿旋转与关节角不自洽：每 tick 转 0.4 rad（低于 rot_jump 阈值）而关节几乎不动。

    这是"欧拉约定读错"的典型指纹——只看位姿本身永远发现不了（compute/apply 互为逆运算，
    用错约定也自洽），必须拿关节角当独立信息源。2026-07-26 实测数据就是这样暴露的。
    """
    d = f["observations/right_arm/eef_pose"]
    v = d[:, 3:]
    v[:, 2] = v[:, 2] + np.arange(len(v)) * 0.4
    d[:, 3:] = v


def euler_wrap(f):
    """欧拉分量 +2π：数值跳变但物理姿态完全没变，只该告警。"""
    d = f["observations/right_arm/eef_pose"]
    v = d[:, 5]
    v[45:] += 2 * np.pi
    d[:, 5] = v


def frozen_camera(f):
    """相机缓存冻结 40 帧（机械臂仍在动）。"""
    d = f["observations/images/cam_high"]
    frame = d[20]
    for t in range(21, 61):
        d[t] = frame


def dup_frames_slow_arm(f):
    """隔帧重复 + 每 tick 关节位移 < 1e-3 rad：旧采集重复帧的慢速段形态。"""
    img = f["observations/images/cam_high"]
    joint = f["observations/right_arm/joint"]
    eef = f["observations/right_arm/eef_pose"]
    for t in range(20, 60, 2):
        img[t + 1] = img[t]
    joint[20:60] = joint[20] + np.linspace(0, 5e-4, 40)[:, None]
    eef[20:60, :3] = eef[20, :3]


def dark_frames(f):
    d = f["observations/images/cam_high"]
    for t in range(10, 40):
        d[t] = 0


def stale_receive(f):
    d = f["timestamps/cams/cam_high/receive"]
    d[:] = d[:] - 0.3


def capture_ts_frozen(f):
    d = f["timestamps/cams/cam_high/capture"]
    d[20:80] = d[20]


def frame_ts_backwards(f):
    """tick 时间倒退：后台写线程乱序落盘的指纹。"""
    d = f["timestamps/frame"]
    v = d[:]
    v[40], v[41] = v[41], v[40]
    d[:] = v


def arm_ts_desync(f):
    """图像与状态系统性错位 250ms ≈ 7.5 tick。"""
    f["timestamps/arms/right_arm"][:] += 0.25


def constant_camera_latency(f):
    """相机固有流水线延迟 80ms：整批数据共有的属性，只该告警、不该按 episode 隔离。"""
    f["timestamps/cams/cam_high/capture"][:] -= 0.080
    f["timestamps/cams/cam_high/receive"][:] -= 0.010


def clock_base_mismatch(f):
    """capture 用了另一套时钟基准：同步误差无法判定，只能告警。"""
    f["timestamps/cams/cam_high/capture"][:] -= 3600.0


def too_short(f):
    for name in ["timestamps/frame", "observations/right_arm/joint",
                 "observations/right_arm/gripper", "observations/right_arm/eef_pose",
                 "timestamps/arms/right_arm", "observations/images/cam_high",
                 "timestamps/cams/cam_high/capture", "timestamps/cams/cam_high/receive"]:
        v = f[name][:12]
        del f[name]
        f.create_dataset(name, data=v)


CASES = [
    (None,                    "ok",   None),                  # 干净底样不许被误判
    (gripper_1d,              "ok",   None),
    (constant_camera_latency, "warn", "sync_lag"),
    (clock_base_mismatch,     "warn", "capture_ts_clock_suspect"),
    (euler_wrap,              "warn", "euler_wrap"),
    (nan_joint,               "bad",  "nan_inf"),
    (nan_eef,                 "bad",  "nan_inf"),
    (inf_eef_rot,             "bad",  "nan_inf"),
    (eef_len_mismatch,        "bad",  "length_mismatch"),
    (missing_gripper,         "bad",  "missing_dataset"),
    (missing_capture_ts,      "bad",  "missing_dataset"),
    (gripper_oob,             "bad",  "gripper_range"),
    (joint_jump,              "bad",  "joint_jump"),
    (eef_xyz_jump,            "bad",  "eef_jump"),
    (eef_rot_jump,            "bad",  "rot_jump"),
    (rot_joint_inconsistent,  "bad",  "rot_joint_inconsistent"),
    (frozen_camera,           "bad",  "frozen_camera"),
    (dup_frames_slow_arm,     "bad",  "duplicate_frames"),
    (dark_frames,             "bad",  "bad_frames"),
    (stale_receive,           "bad",  "misaligned"),
    (capture_ts_frozen,       "bad",  "sync_jitter"),
    (frame_ts_backwards,      "bad",  "non_monotonic_frame_ts"),
    (arm_ts_desync,           "bad",  "image_arm_desync"),
    (too_short,               "bad",  "too_short"),
    ("truncate_file",         "bad",  "unreadable"),          # 特殊：落盘后截断字节
]


def run(tmpdir):
    base = os.path.join(tmpdir, "base.hdf5")
    write_clean_episode(base)

    args = clean.build_parser().parse_args(["--data", tmpdir])
    failures = []
    for i, (inject, want_status, want_key) in enumerate(CASES):
        name = "clean_baseline" if inject is None else (
            inject if isinstance(inject, str) else inject.__name__)
        path = os.path.join(tmpdir, f"{i}.hdf5")
        shutil.copy(base, path)
        if inject == "truncate_file":
            with open(path, "r+b") as fh:
                fh.truncate(os.path.getsize(path) // 2)
        elif inject is not None:
            with h5py.File(path, "a") as f:
                inject(f)

        result = clean.check_episode(path, args)
        found = result["errors"] + result["warnings"]
        ok_status = result["status"] == want_status
        ok_key = want_key is None or any(want_key in msg for msg in found)
        if ok_status and ok_key:
            log("test_clean", f"PASS {name}: {want_status}"
                + (f" ({want_key})" if want_key else ""), "INFO")
        else:
            reason = (f"want status={want_status} got={result['status']}" if not ok_status
                      else f"缺少判据关键字 '{want_key}'")
            failures.append(name)
            log("test_clean", f"FAIL {name}: {reason}; 实际={found}", "ERROR")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--keep", action="store_true", help="保留临时目录")
    args = parser.parse_args()

    tmpdir = tempfile.mkdtemp(prefix="robokit-clean-selftest-")
    try:
        failures = run(tmpdir)
    finally:
        if args.keep:
            log("test_clean", f"临时数据保留在 {tmpdir}", "INFO")
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)

    total = len(CASES)
    if failures:
        log("test_clean", f"{len(failures)}/{total} 例不符合预期: {failures}", "ERROR")
        return 1
    log("test_clean", f"全部 {total} 例符合预期", "INFO")
    return 0


if __name__ == "__main__":
    sys.exit(main())
