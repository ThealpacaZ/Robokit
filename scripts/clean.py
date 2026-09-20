"""数据清洗：扫描任务目录下所有 episode，输出质量报告，可选隔离坏数据。

用法:
    python scripts/clean.py --data datasets/任务名                 # 只生成报告
    python scripts/clean.py --data datasets/任务名 --quarantine    # 并把 error 级 episode 移入 _quarantine/

判定规则:
    error   → 自动化可确定的坏数据（结构损坏/NaN/长度错位/相机冻结或重复帧/时间倒退/
              动作或旋转跳变/夹爪越界/图像-状态错位/过短），--quarantine 时移走。
    warning → 需要人工复查的可疑项（模糊/亮度突变/静止 episode/少量陈旧帧/欧拉数值 wrap/
              相机时钟基准可疑），只提示不移动。

检查顺序有依赖：`check_structure` 先保证「该有的 dataset 都有、长度都等于 T、数值都有限」，
后续检查才敢直接读。单个 episode 的任何异常都被隔离成该 episode 的 error，不会中断整批扫描。

报告写入 <data>/clean_report.json，结构为 {summary, dataset, episodes}：
episodes 下每段除 status/errors/warnings 外还带 metrics（重复帧比例、最长冻结长度、跳变极值、
同步误差 P95 等），用于「是否需要重采」这类决策；dataset 下是跨 episode 的一致性结论。
RLDS 转换（rlds/build.py）会读这份报告，有 bad 段就拒绝转换，所以清洗结论是硬闸不是建议。

退出码：有 bad episode 或 dataset 级 error → 1，否则 0。
"""
import argparse
import glob
import json
import os
import shutil
import sys
import traceback

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robokit.arms.piper import _GRIPPER_FULL
from robokit.episode import EpisodeFile
from robokit.utils import log

IMG_BATCH = 64            # 图像分批读取，控制内存
BLUR_SAMPLE_STRIDE = 10   # 模糊检查抽帧间隔
FROZEN_DIFF = 0.3         # 下采样灰度的帧间平均差，低于此认为画面没变
CLOCK_SUSPECT = 1.0       # |capture-receive| 中位数超过该秒数 → 两者疑似不同时钟基准
BAD_FRAC = 0.10           # 超过该比例的帧异常 → 从 warning 升级为 error
ROT_SLACK = 0.05          # 位姿旋转允许超出关节角变化之和的裕量 rad（读数噪声/不同 CAN 帧）

# 夹爪读数的「物理合理」区间，超出才判 error。
# 判据按**毫米**定义，再用 PiperArm._GRIPPER_FULL 换算成归一化值 —— 因为问题的根源
# 正是那个常量不可信（名义 70mm），把界直接写成归一化数会跟着一起错。
#
# 2026-07-27 用 scripts/gripper_probe.py --observe 实测全行程：
#   原始读数 [-1500, 99600] = [-1.50mm, 99.60mm]，当前常量下归一化到 1.4229。
#   全闭附近为负是零点偏移（正常抖动）；全开 99.6mm 远超名义 70mm。
# 早先 [-0.02, 1.02] 的判据把 stack cups 19 段里的 8 段正常采集判成了坏数据；
# 之后改的 1.3 上界也偏窄（开到底就会误判）。现按物理不可能设界。
#
# 越过 [0,1] 不在这里判 error，只给 warning：那是**下发截断**问题
# （_move_gripper 会 clip 到 [0,1]），数据本身没坏。
GRIPPER_SANE_MM = (-5.0, 120.0)   # 实测行程 99.6mm，两端留足余量
_FULL_MM = _GRIPPER_FULL * 0.001  # SDK 原始单位是 0.001mm
GRIPPER_SANE = (GRIPPER_SANE_MM[0] / _FULL_MM, GRIPPER_SANE_MM[1] / _FULL_MM)


def _gripper_full_mm(ep) -> float:
    """这段数据归一化夹爪时用的满行程：优先读采集配置（attrs.config_json 里的
    arms.<name>.gripper_full_mm，2026-09-11 起可配置），没有就是历史的 70mm。"""
    try:
        cfg = json.loads(ep.attrs.get("config_json", "{}"))
        for arm_cfg in cfg.get("robot", {}).get("arms", {}).values():
            if "gripper_full_mm" in arm_cfg:
                return float(arm_cfg["gripper_full_mm"])
    except (ValueError, AttributeError, TypeError):
        pass
    return _FULL_MM

# eef_pose[3:] 的欧拉约定。Piper 固件 `GetArmEndPoseMsgs` 的 RX/RY/RZ 是**固定轴 xyz
# 外旋**（≡ 内旋 ZYX）：按 xyz 解读时，记录位姿与官方 URDF 正运动学在实测 5 段 1129 帧上
# 最大只差 1.03°；按内旋 XYZ 解读则处处差到 ~180°。
# robokit/pose.py（EULER_SEQ）与 rlds builder 现已统一到同一约定。
DEFAULT_EULER = "xyz"


def check_structure(ep: EpisodeFile, min_frames):
    """结构完整性：dataset 是否齐全、长度是否都等于 T、数值是否有限、tick 是否单调。

    这一关同时是后续检查的前置条件——它过了，后面就不必再判存在性和长度。
    """
    errors = []
    T = ep.length
    if T < min_frames:
        errors.append(f"too_short: {T} frames < {min_frames}")
    if not ep.arms:
        errors.append("no_arm: observations 下没有任何机械臂分组")

    frame_ts = ep.frame_ts()
    if not np.isfinite(frame_ts).all():
        errors.append("nan_inf: timestamps/frame")
    elif T >= 2:
        back = int((np.diff(frame_ts) < 0).sum())
        if back:
            errors.append(f"non_monotonic_frame_ts: {back} ticks 时间倒退（帧可能乱序落盘）")

    def check_array(key, name, arr):
        if arr.shape[0] != T:
            errors.append(f"length_mismatch: {name} has {arr.shape[0]} != {T}")
        if not np.isfinite(arr).all():
            errors.append(f"nan_inf: {name}")

    for arm in ep.arms:
        for name, required, read in (
            ("joint", True, ep.joint),
            ("gripper", True, ep.gripper),
            ("eef_pose", False, ep.eef_pose),          # 臂不提供 EEF 位姿时允许缺省
        ):
            key = f"observations/{arm}/{name}"
            if ep.has(key):
                check_array(key, f"{arm}/{name}", read(arm))
            elif required:
                errors.append(f"missing_dataset: {key}")
        key = f"timestamps/arms/{arm}"
        if ep.has(key):
            check_array(key, key, ep.arm_ts(arm))
        else:
            errors.append(f"missing_dataset: {key}")

    for cam in ep.cams:
        if ep.images(cam).shape[0] != T:
            errors.append(f"length_mismatch: images/{cam} has {ep.images(cam).shape[0]} != {T}")
        for kind in ("capture", "receive"):
            key = f"timestamps/cams/{cam}/{kind}"
            if ep.has(key):
                check_array(key, key, ep.cam_ts(cam, kind))
            else:
                errors.append(f"missing_dataset: {key}")
    return errors


def check_actions(ep: EpisodeFile, joint_jump, eef_jump, rot_jump, euler):
    """动作/状态合理性：平移与旋转跳变、夹爪越界、静止。

    旋转分三件事判：

    1. **物理跳变**：相邻位姿的相对旋转角（SO(3) 测地角，天然模 2π）超过阈值。
    2. **与关节不自洽**：串联臂末端姿态的变化不可能超过各关节角变化之和
       （几何上是严格上界）。超了只有两种可能——位姿数据坏了，或者**欧拉约定读错了**。
       后者更常见，且只靠位姿本身永远查不出来（compute/apply 互为逆运算，
       用错约定也自洽），必须引入关节角这个独立信息源才能发现。
    3. **数值 wrap**：欧拉分量差 >π 而物理旋转很小，无害但会骗过任何直接对欧拉数值做差的代码。

    `euler` 是解读 `eef_pose[3:]` 用的约定（scipy 记法，大写=内旋、小写=外旋）。
    """
    from scipy.spatial.transform import Rotation

    errors, warnings, metrics = [], [], {}
    total_motion = 0.0
    for arm in ep.arms:
        m = {}
        joint = ep.joint(arm)
        if joint.shape[0] >= 2:
            diff = np.abs(np.diff(joint, axis=0))
            step = diff.max(axis=1)
            m["joint_step_max_rad"] = float(step.max())
            jumps = int((step > joint_jump).sum())
            if jumps:
                errors.append(f"joint_jump: {arm} has {jumps} steps > {joint_jump} rad/tick "
                              f"(max {step.max():.3f})")
            total_motion += float(diff.sum())

        eef = ep.eef_pose(arm)
        if eef is not None and eef.shape[0] >= 2:
            step = np.linalg.norm(np.diff(eef[:, :3], axis=0), axis=1)
            m["eef_step_max_m"] = float(step.max())
            jumps = int((step > eef_jump).sum())
            if jumps:
                errors.append(f"eef_jump: {arm} has {jumps} steps > {eef_jump} m/tick "
                              f"(max {step.max():.3f})")

            rots = Rotation.from_euler(euler, eef[:, 3:])
            geo = (rots[:-1].inv() * rots[1:]).magnitude()      # 相邻帧转过的角度 [0,π]
            m["rot_step_max_rad"] = float(geo.max())
            rot_jumps = int((geo > rot_jump).sum())
            if rot_jumps:
                errors.append(f"rot_jump: {arm} has {rot_jumps} steps > {rot_jump} rad/tick "
                              f"(max {geo.max():.3f})")

            if joint.shape[0] == eef.shape[0] and joint.shape[0] >= 2:
                bound = np.abs(np.diff(joint, axis=0)).sum(axis=1)   # 末端姿态变化的严格上界
                over = geo - bound
                m["rot_vs_joint_max_rad"] = float(over.max())
                n_incons = int((over > ROT_SLACK).sum())
                if n_incons:
                    t = int(np.argmax(over))
                    errors.append(
                        f"rot_joint_inconsistent: {arm} {n_incons}/{len(geo)} ticks 位姿旋转超过"
                        f"关节角变化之和（最大 tick {t}: 位姿说转了 {np.degrees(geo[t]):.1f}°，"
                        f"关节只动了 {np.degrees(bound[t]):.1f}°）。串联臂不可能，"
                        f"要么位姿数据坏了，要么 eef_pose 的欧拉约定不是当前假设的 '{euler}'")
            raw = np.abs(np.diff(eef[:, 3:], axis=0)).max(axis=1)
            # wrap 判据与约定无关：任何欧拉约定下，单个分量加减 2π 都不改变姿态
            wraps = int(((raw > np.pi) & (geo <= rot_jump)).sum())
            m["euler_wrap"] = wraps
            if wraps:
                warnings.append(
                    f"euler_wrap: {arm} {wraps} ticks 欧拉分量数值跳变 >π 而物理旋转很小；"
                    "robokit RLDS 用相对旋转算 delta 不受影响，任何直接对欧拉数值做差的"
                    "消费方（旧转换脚本/曲线判读）会读到假跳变")

        gripper = ep.gripper(arm)
        m["gripper_min"] = float(gripper.min())
        m["gripper_max"] = float(gripper.max())
        full_mm = _gripper_full_mm(ep)
        lo, hi = GRIPPER_SANE_MM[0] / full_mm, GRIPPER_SANE_MM[1] / full_mm
        out = int(((gripper < lo) | (gripper > hi)).sum())
        if out:
            errors.append(f"gripper_range: {arm} has {out} frames outside [{lo}, {hi}] "
                          f"(min={gripper.min():.4f}, max={gripper.max():.4f}) —— "
                          "超出 SDK 正常抖动量级，多半是归一化常数错了或读数损坏")
        # 超过 1.0 的开度：不是坏数据，但**下发时会被静默截断**成 1.0（见常量注释）。
        # 只判正侧：负侧 -0.019 截断成 0 就是「全闭」，物理上本来就是那个意思，无损失；
        # 正侧 1.131 截断成 1.0 则是真机少开 9.2mm，是会影响抓取的系统性误差。
        over = gripper > 1.0
        m["gripper_over_full"] = int(over.sum())
        if over.any():
            lost_mm = float((gripper.max() - 1.0) * 70.0)
            warnings.append(
                f"gripper_clip_loss: {arm} {int(over.sum())}/{len(gripper)} 帧开度 >1.0"
                f"（max={gripper.max():.4f}）。采集侧不截断、下发侧 clip 到 1.0，"
                f"这些标签真机最多少开 {lost_mm:.1f}mm，且不会报错")
        metrics[arm] = m

    if total_motion < 0.01:
        warnings.append("static_episode: arm barely moved, likely junk collection")
    return errors, warnings, metrics


def check_timestamps(ep: EpisodeFile, stale_threshold):
    """时间戳：相机缓存健康度（cache_age）与图像-状态同步误差（sync_error）分别判。

    两者含义不同：cache_age 大说明相机缓存陈旧（掉帧/回调阻塞/解码慢）；
    sync_error 大说明这一帧图像的**采集时刻**和机械臂状态读取时刻本身错位。
    sync_error 依赖 ROS header 时间与本机 time.time() 同基准，因此先用
    |capture-receive| 的中位数做基准合理性检查，可疑时只告警、不下 error 结论。
    """
    errors, warnings, metrics = [], [], {}
    frame_ts = ep.frame_ts()
    T = ep.length
    arm_ts = {arm: ep.arm_ts(arm) for arm in ep.arms}

    if len(arm_ts) >= 2:
        stacked = np.stack(list(arm_ts.values()))
        span = stacked.max(axis=0) - stacked.min(axis=0)
        metrics["arm_read_span_p95_ms"] = float(np.percentile(span, 95) * 1000)
        n_span = int((span > stale_threshold).sum())
        if n_span:
            warnings.append(f"arm_read_span: {n_span}/{T} ticks 多臂状态读取时间跨度 > "
                            f"{stale_threshold * 1000:.0f}ms（顺序读取造成的臂间错位）")

    for cam in ep.cams:
        capture = ep.cam_ts(cam, "capture")
        receive = ep.cam_ts(cam, "receive")
        cm = {}

        cache_age = frame_ts - receive      # 本轮 tick 时该帧已经在缓存里放了多久
        n_stale = int((cache_age > stale_threshold).sum())
        cm["cache_age_p95_ms"] = float(np.percentile(cache_age, 95) * 1000)
        if n_stale > BAD_FRAC * T:
            errors.append(f"misaligned: {cam} {n_stale}/{T} frames older than "
                          f"{stale_threshold * 1000:.0f}ms at tick time")
        elif n_stale:
            warnings.append(f"stale_frames: {cam} {n_stale}/{T} frames older than "
                            f"{stale_threshold * 1000:.0f}ms")

        if (np.diff(capture) < 0).any():
            warnings.append(f"non_monotonic_capture_ts: {cam}")

        reuse = int((np.diff(capture) == 0).sum())
        cm["capture_ts_reuse"] = reuse
        if reuse > 0.3 * T:
            warnings.append(f"low_camera_fps: {cam} reused previous frame {reuse}/{T} ticks, "
                            "camera fps likely below collection freq")

        offset = float(np.median(np.abs(capture - receive)))
        cm["capture_receive_median_ms"] = offset * 1000
        if offset > CLOCK_SUSPECT:
            warnings.append(f"capture_ts_clock_suspect: {cam} |capture-receive| 中位数 "
                            f"{offset:.3f}s，ROS header 时间疑似与本机时钟不同基准，"
                            "本 episode 的图像-状态同步误差无法判定")
            continue
        for arm, ts in arm_ts.items():
            # 系统性滞后与逐帧抖动分开判：相机固有流水线延迟是**恒定**偏移，整段数据一起偏，
            # 不该按 episode 隔离（一隔就是全隔）；真正毁掉帧-状态配对的是偏移的抖动。
            sync = ts - capture
            lag = float(np.median(sync))
            jitter = np.abs(sync - lag)
            cm[f"sync_lag_median_ms/{arm}"] = lag * 1000
            cm[f"sync_jitter_p95_ms/{arm}"] = float(np.percentile(jitter, 95) * 1000)

            if abs(lag) > 2 * stale_threshold:
                errors.append(f"image_arm_desync: {cam} vs {arm} 图像采集与状态读取系统性相差 "
                              f"{lag * 1000:.0f}ms ≈ {abs(lag) * ep.freq:.1f} 个 tick，"
                              "训练里图像与标签整体错位")
            elif abs(lag) > stale_threshold:
                warnings.append(f"sync_lag: {cam} vs {arm} 系统性相差 {lag * 1000:.0f}ms "
                                f"(≈{abs(lag) * ep.freq:.1f} tick)，属采集链路固有延迟，需整批评估")

            n_bad = int((jitter > stale_threshold).sum())
            if n_bad > BAD_FRAC * T:
                errors.append(f"sync_jitter: {cam} vs {arm} {n_bad}/{T} frames 相对自身中位偏移"
                              f"抖动 > {stale_threshold * 1000:.0f}ms "
                              f"(P95 {cm[f'sync_jitter_p95_ms/{arm}']:.0f}ms)，帧-状态配对不可信")
            elif n_bad:
                warnings.append(f"sync_jitter: {cam} vs {arm} {n_bad}/{T} frames 抖动 > "
                                f"{stale_threshold * 1000:.0f}ms")
        metrics[cam] = cm
    return errors, warnings, metrics


def check_images(ep: EpisodeFile, blur_threshold, dup_ratio):
    """图像异常：黑/白/无纹理帧、逐字节重复帧、冻结帧、亮度突变、模糊。

    重复帧与冻结帧是两个独立信号：
      - 逐字节完全重复 → 上游把同一条消息当新帧反复取用。真实传感器有噪声，
        相邻帧不可能完全相同，所以这个判据与机械臂是否在动无关（慢速段也抓得到）。
      - 下采样画面几乎不变但机械臂在动 → 画面与状态错位。
    """
    import cv2

    errors, warnings, metrics = [], [], {}
    T = ep.length
    joints = np.concatenate([ep.joint(arm) for arm in ep.arms], axis=1)
    joint_step = np.abs(np.diff(joints, axis=0)).max(axis=1)  # (T-1,)

    for cam in ep.cams:
        ds = ep.images(cam)
        n_dark = n_bright = n_flat = n_frozen = n_bjump = n_blur = n_dup = 0
        n_blur_sampled = 0
        dup_run = max_dup_run = 0
        prev_small = prev_full = prev_mean = None
        for start in range(0, T, IMG_BATCH):
            batch = ds[start:start + IMG_BATCH]
            gray = batch.mean(axis=3, dtype=np.float32)    # (B,H,W) 亮度
            means = gray.mean(axis=(1, 2))
            stds = gray.std(axis=(1, 2))
            n_dark += int((means < 5).sum())
            n_bright += int((means > 250).sum())
            n_flat += int(((stds < 2) & (means >= 5) & (means <= 250)).sum())

            small = gray[:, ::8, ::8]
            for i in range(small.shape[0]):
                if prev_small is not None:
                    t = start + i
                    if np.array_equal(batch[i], prev_full):
                        n_dup += 1
                        dup_run += 1
                        max_dup_run = max(max_dup_run, dup_run)
                    else:
                        dup_run = 0
                    diff = float(np.abs(small[i] - prev_small).mean())
                    if diff < FROZEN_DIFF and joint_step[t - 1] > 1e-3:
                        n_frozen += 1
                    if abs(means[i] - prev_mean) > 60:
                        n_bjump += 1
                # 存副本而不是视图：视图会把整批 gray/batch 一直留在内存里
                prev_small = small[i].copy()
                prev_full = batch[i].copy()
                prev_mean = means[i]

            # 模糊检查（抽帧，Laplacian 方差）
            for i in range(0, batch.shape[0], BLUR_SAMPLE_STRIDE):
                n_blur_sampled += 1
                lap = cv2.Laplacian(gray[i].astype(np.uint8), cv2.CV_64F).var()
                if lap < blur_threshold:
                    n_blur += 1

        pairs = max(T - 1, 1)
        metrics[cam] = {
            "dup_ratio": n_dup / pairs,
            "max_dup_run": max_dup_run,
            "frozen": n_frozen,
            "dark": n_dark, "bright": n_bright, "flat": n_flat,
            "blur_sampled": n_blur_sampled, "blur": n_blur,
        }

        bad = n_dark + n_bright + n_flat
        if bad > max(3, BAD_FRAC * T):
            errors.append(f"bad_frames: {cam} dark={n_dark} bright={n_bright} flat={n_flat} of {T}")
        elif bad:
            warnings.append(f"bad_frames: {cam} dark={n_dark} bright={n_bright} flat={n_flat} of {T}")

        if n_dup / pairs > dup_ratio:
            errors.append(f"duplicate_frames: {cam} {n_dup}/{pairs} 相邻帧逐字节完全重复 "
                          f"({n_dup / pairs:.1%} > {dup_ratio:.0%}，最长连续 {max_dup_run})，"
                          "上游在复用同一条图像消息")
        elif n_dup:
            warnings.append(f"duplicate_frames: {cam} {n_dup}/{pairs} 相邻帧完全重复 "
                            f"({n_dup / pairs:.1%}，最长连续 {max_dup_run})")

        if n_frozen > max(3, 0.05 * T):
            errors.append(f"frozen_camera: {cam} {n_frozen}/{T} frames unchanged while arm moving")
        elif n_frozen:
            warnings.append(f"frozen_frames: {cam} {n_frozen}/{T}")

        if n_bjump:
            warnings.append(f"brightness_jump: {cam} {n_bjump} sudden changes (lighting?)")
        if n_blur:
            warnings.append(f"blurry: {cam} {n_blur}/{n_blur_sampled} sampled frames below "
                            f"Laplacian var {blur_threshold} (human review)")
    return errors, warnings, metrics


def episode_schema(ep: EpisodeFile):
    """这段 episode 的形状指纹，供跨 episode 一致性比较。"""
    return {
        "task_name": str(ep.attrs.get("task_name", "")),
        "freq": ep.freq,
        "cams": {cam: list(ep.images(cam).shape[1:]) for cam in ep.cams},
        "arms": {arm: (int(ep.joint(arm).shape[1]) if ep.has(f"observations/{arm}/joint") else None)
                 for arm in ep.arms},
        "eef": sorted(arm for arm in ep.arms if ep.has(f"observations/{arm}/eef_pose")),
    }


def check_dataset_consistency(report):
    """跨 episode 一致性。

    RLDS builder 只按（字典序）第一段 episode 探测相机分辨率与状态维度，其余 episode
    直接按该形状写入，所以混了不同分辨率/相机数/自由度的目录会在转换**末尾**才报
    `Shapes (a) and (b) are incompatible`，而且要等前面的 episode 全部处理完。
    这里一秒钟就能提前判掉。
    """
    errors, warnings = [], []
    schemas = {name: r["metrics"]["schema"] for name, r in report.items()
               if r.get("metrics", {}).get("schema")}
    if len(schemas) < 2:
        return errors, warnings

    ref_name, ref = next(iter(schemas.items()))
    for name, s in schemas.items():
        if s["cams"] != ref["cams"]:
            errors.append(f"schema_mismatch: {name} 相机/分辨率 {s['cams']} != {ref_name} 的 "
                          f"{ref['cams']}；RLDS 转换会在末尾失败")
        if s["arms"] != ref["arms"] or s["eef"] != ref["eef"]:
            errors.append(f"schema_mismatch: {name} 机械臂/自由度 {s['arms']} eef={s['eef']} != "
                          f"{ref_name} 的 {ref['arms']} eef={ref['eef']}")
        if s["freq"] != ref["freq"]:
            warnings.append(f"freq_mismatch: {name} freq={s['freq']} != {ref_name} 的 {ref['freq']}；"
                            "时间尺度不同的数据混在一个数据集里")
        if s["task_name"] != ref["task_name"]:
            warnings.append(f"task_name_mismatch: {name} task_name='{s['task_name']}' != "
                            f"{ref_name} 的 '{ref['task_name']}'；两者会成为不同的 "
                            "language_instruction，混任务或错标签都会这样")
    return errors, warnings


def check_episode(path, args):
    """单个 episode 的完整检查。任何异常都收敛成本 episode 的 error，不向外抛。"""
    result = {"status": "bad", "errors": [], "warnings": [], "frames": 0, "metrics": {}}
    try:
        ep = EpisodeFile(path)
    except Exception as e:
        result["errors"] = [f"unreadable: {e}"]
        return result

    try:
        with ep:
            result["frames"] = ep.length
            stale_threshold = 2.0 / ep.freq
            errors = check_structure(ep, args.min_frames)
            warnings, metrics = [], {"schema": episode_schema(ep)}
            if not errors:  # 结构坏了后续检查无意义（也不再保证能安全读）
                e2, w2, m_arm = check_actions(ep, args.joint_jump, args.eef_jump,
                                              args.rot_jump, args.euler)
                e3, w3, m_ts = check_timestamps(ep, stale_threshold)
                e4, w4, m_img = check_images(ep, args.blur_threshold, args.dup_ratio)
                errors = e2 + e3 + e4
                warnings = w2 + w3 + w4
                metrics.update({"arms": m_arm, "timestamps": m_ts, "images": m_img})
    except Exception as e:
        log("clean", f"{os.path.basename(path)} check crashed:\n{traceback.format_exc()}", "DEBUG")
        result["errors"] = [f"check_failed: {type(e).__name__}: {e}"]
        return result

    result["errors"] = errors
    result["warnings"] = warnings
    result["metrics"] = metrics
    result["status"] = "bad" if errors else ("warn" if warnings else "ok")
    return result


def quarantine(data_dir, episodes):
    """把 error 级 episode 移入 _quarantine/，不覆盖已有同名文件。"""
    qdir = os.path.join(data_dir, "_quarantine")
    moved = 0
    for name, result in episodes.items():
        if result["status"] != "bad":
            continue
        os.makedirs(qdir, exist_ok=True)
        stem, ext = os.path.splitext(name)
        dest = os.path.join(qdir, name)
        n = 1
        while os.path.exists(dest):   # 编号会被后续采集复用，绝不能覆盖旧的隔离文件
            dest = os.path.join(qdir, f"{stem}.{n}{ext}")
            n += 1
        shutil.move(os.path.join(data_dir, name), dest)
        moved += 1
    log("clean", f"quarantined {moved} bad episodes -> {qdir}", "INFO")


def build_parser():
    """判据阈值集中在这里；自检脚本 test_clean.py 用同一个 parser 取默认值。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="任务目录（含 N.hdf5）")
    parser.add_argument("--quarantine", action="store_true", help="把 error 级 episode 移入 _quarantine/")
    parser.add_argument("--min-frames", type=int, default=30)
    parser.add_argument("--joint-jump", type=float, default=0.3, help="关节单步跳变上限 rad/tick")
    parser.add_argument("--eef-jump", type=float, default=0.05, help="EEF 单步位移上限 m/tick")
    parser.add_argument("--rot-jump", type=float, default=0.5,
                        help="EEF 单步旋转上限 rad/tick（相对旋转角，不受欧拉 wrap 影响）")
    parser.add_argument("--euler", default=DEFAULT_EULER,
                        help=f"eef_pose 旋转分量的欧拉约定（scipy 记法，默认 {DEFAULT_EULER}）")
    parser.add_argument("--dup-ratio", type=float, default=0.05,
                        help="相邻帧完全重复的比例上限，超过判 error")
    parser.add_argument("--blur-threshold", type=float, default=100.0, help="Laplacian 方差模糊阈值")
    return parser


def main():
    args = build_parser().parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.hdf5")),
                   key=lambda p: int(os.path.splitext(os.path.basename(p))[0])
                   if os.path.splitext(os.path.basename(p))[0].isdigit() else 1 << 30)
    if not files:
        log("clean", f"no .hdf5 found in {args.data}", "ERROR")
        return 1

    pending = glob.glob(os.path.join(args.data, "*.hdf5.tmp"))
    if pending:
        log("clean", f"{len(pending)} 个 .hdf5.tmp 残留（采集中断留下的未完成文件），已跳过", "WARNING")

    episodes = {}
    counts = {"ok": 0, "warn": 0, "bad": 0}
    for path in files:
        name = os.path.basename(path)
        result = check_episode(path, args)
        episodes[name] = result
        counts[result["status"]] += 1

        level = {"ok": "INFO", "warn": "WARNING", "bad": "ERROR"}[result["status"]]
        log("clean", f"{name}: {result['status']} ({result['frames']} frames)", level)
        for e in result["errors"]:
            log("clean", f"    [error] {e}", "ERROR")
        for w in result["warnings"]:
            log("clean", f"    [warn]  {w}", "WARNING")

    ds_errors, ds_warnings = check_dataset_consistency(episodes)
    for e in ds_errors:
        log("clean", f"[dataset error] {e}", "ERROR")
    for w in ds_warnings:
        log("clean", f"[dataset warn]  {w}", "WARNING")

    report = {
        "summary": {**counts, "total": len(files)},
        "dataset": {"errors": ds_errors, "warnings": ds_warnings},
        "episodes": episodes,
    }
    report_path = os.path.join(args.data, "clean_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    log("clean", f"report -> {report_path}", "INFO")

    if args.quarantine:
        quarantine(args.data, episodes)

    log("clean", f"summary: {counts['ok']} ok, {counts['warn']} need review, {counts['bad']} bad "
        f"of {len(files)} episodes", "INFO")
    return 1 if counts["bad"] or ds_errors else 0


if __name__ == "__main__":
    sys.exit(main())
