#!/usr/bin/env python
"""低速把 Piper 从臂移动到 stack cups 某个示教 episode 的第一帧。

默认只做预览，不向 CAN 发送任何控制帧：

    python scripts/reset_piper_to_demo_start.py

实际执行前，必须先让主臂退出共享 CAN 联动链路。若从臂刚从主从高跟随模式切换到
位置速度模式，按 Piper 官方要求先清除控制器内部状态：

    python scripts/reset_piper_to_demo_start.py --execute --controller-reset

若主臂断开后已经单独重启过从臂，可显式跳过控制器 reset：

    python scripts/reset_piper_to_demo_start.py --execute --skip-controller-reset

不传 --episode 时，脚本扫描数据目录中所有 HDF5，选择与当前关节角最接近的第一帧。
也可用 --episode 41 固定选择 41.hdf5。实际下发始终使用六关节角，不读取图像
dataset，也不控制夹爪。

注意：--controller-reset 会让机械臂瞬间失力。执行确认前必须托住机械臂，并准备急停。
"""
import argparse
import glob
import os
import sys
import time
from collections import deque
from dataclasses import dataclass

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从 piper_limits 取而不是 piper_ik：后者依赖 scipy，import 要多花 0.33s，而复位
# 只需要这张常量表。全仓唯一权威表见该模块。
from robokit.arms.piper_limits import DEFAULT_JOINT_LIMITS_DEG

_DEG2RAD = np.pi / 180.0
_JOINT_LIMITS_DEG = DEFAULT_JOINT_LIMITS_DEG
_DEFAULT_STEP_DEG = 15.0
# reset 的目标就是所选 HDF5 第一帧；总跨度不另设默认拒绝门限，实际运动仍由
# --step-deg 拆成受监控的小段。需要额外限制总跨度时可显式传正数。
_DEFAULT_MAX_START_GAP_DEG = 0.0

# 固定示教起点：stack cups 的 0.hdf5 首帧，2026-08-06 一次性读出来写死在这里。
# 之前每次复位都要在本机留一份 HDF5 才能跑，而这个位姿从来不变，为它保留 146MB
# 数据（以及"数据不在就复位不了"这个失败模式）不值得。要用别的起点仍可 --dataset
# 指向真实数据集，逻辑完全不变。
FIXED_DEMO_START_JOINT = np.array(
    [-1.433893, 0.004294, -0.002601, 0.262340, 0.224240, -0.203994], dtype=np.float64
)
FIXED_DEMO_START_EEF = np.array(
    [0.012811, -0.054544, 0.194241, 2.716553, 1.426894, 1.336940], dtype=np.float64
)
FIXED_DEMO_START_ARM = "right_arm"


@dataclass(frozen=True)
class DemoStart:
    path: str
    arm_name: str
    joint: np.ndarray
    eef_pose: np.ndarray

    @property
    def episode(self):
        return os.path.splitext(os.path.basename(self.path))[0]


def _fixed_demo_start():
    """写死的示教起点，不读任何文件。"""
    deg = np.degrees(FIXED_DEMO_START_JOINT)
    if np.any(deg < _JOINT_LIMITS_DEG[:, 0]) or np.any(deg > _JOINT_LIMITS_DEG[:, 1]):
        raise ValueError(f"固定示教起点超出 Piper 限位：{deg}")
    return [
        DemoStart(
            "<fixed>", FIXED_DEMO_START_ARM,
            FIXED_DEMO_START_JOINT.copy(), FIXED_DEMO_START_EEF.copy(),
        )
    ]


def _joint_feedback(sdk):
    msg = sdk.GetArmJointMsgs().joint_state
    return np.array(
        [getattr(msg, f"joint_{i}") for i in range(1, 7)], dtype=np.float64
    ) * 0.001 * _DEG2RAD


def _drivers_enabled(sdk):
    return all(_driver_enable_flags(sdk))


def _driver_enable_flags(sdk):
    msg = sdk.GetArmLowSpdInfoMsgs()
    return tuple(
        bool(getattr(msg, f"motor_{i}").foc_status.driver_enable_status)
        for i in range(1, 7)
    )


def _wait_feedback(sdk, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if sdk.GetArmJointMsgs().time_stamp and sdk.GetArmStatus().time_stamp:
            return
        time.sleep(0.05)
    raise RuntimeError(f"{timeout:.1f}s 内没有收到关节/状态反馈；检查 can0 和从臂电源")


def _assert_no_external_controller(sdk, observe_s=0.3):
    """连接后、首次发送前监听标准控制 ID；发现主臂命令就拒绝执行。

    判据是"窗口内是否收到过任何一帧"，所以窗口只要够长到覆盖最慢的外部发送周期。
    主臂/其他控制进程都在 20Hz 以上，0.3s 能收 6 帧以上，检出能力与原来的 1.0s
    没有区别，但每次复位少等 0.7s（原来固定开销 2.85s 里它占四分之一）。
    """
    time.sleep(observe_s)
    seen = []
    for name, getter in (
        ("mode 0x151", sdk.GetArmModeCtrl),
        ("joint 0x155-157", sdk.GetArmJointCtrl),
        ("gripper 0x159", sdk.GetArmGripperCtrl),
    ):
        msg = getter()
        if getattr(msg, "time_stamp", 0):
            seen.append(f"{name} ({getattr(msg, 'Hz', 0):.0f}Hz)")
    if seen:
        raise RuntimeError(
            "监听到外部控制帧：" + ", ".join(seen)
            + "。主臂或其他控制进程仍在发命令；先断开它们，不能抢同一从臂。"
        )


def _load_demo_starts(dataset_dir, arm_name, only_episode=None):
    """读各 episode 首帧。

    only_episode 给定时只打开那一个文件。全量扫 74 个 HDF5 要 0.32s，而指定了
    episode 时其余 73 个只用来打印"最近的示教首帧"参考列表，为此每次复位多等
    0.3s 不值。不指定时仍需全量——要靠它选最近的起点。
    """

    def episode_sort_key(path):
        stem = os.path.splitext(os.path.basename(path))[0]
        return (0, int(stem)) if stem.isdigit() else (1, stem)

    paths = sorted(
        glob.glob(os.path.join(dataset_dir, "*.hdf5")),
        key=episode_sort_key,
    )
    if not paths:
        raise FileNotFoundError(f"{dataset_dir!r} 中没有 .hdf5")
    if only_episode is not None:
        wanted = str(only_episode)
        paths = [
            p for p in paths
            if os.path.splitext(os.path.basename(p))[0] == wanted
        ]
        if not paths:
            raise FileNotFoundError(f"{dataset_dir!r} 中没有 episode {wanted}")

    starts = []
    for path in paths:
        with h5py.File(path, "r") as h:
            arms = [k for k in h["observations"] if k != "images"]
            selected = arm_name or ("right_arm" if "right_arm" in arms else arms[0])
            if selected not in arms:
                raise KeyError(f"{path}: 没有机械臂 {selected!r}，可选 {arms}")
            joint = np.asarray(
                h[f"observations/{selected}/joint"][0], dtype=np.float64
            )
            eef_pose = np.asarray(
                h[f"observations/{selected}/eef_pose"][0], dtype=np.float64
            )
        if joint.shape != (6,) or eef_pose.shape != (6,):
            raise ValueError(
                f"{path}: 第一帧形状异常 joint={joint.shape}, eef_pose={eef_pose.shape}"
            )
        if not np.all(np.isfinite(joint)) or not np.all(np.isfinite(eef_pose)):
            raise ValueError(f"{path}: 第一帧包含 NaN/Inf")
        deg = np.degrees(joint)
        if np.any(deg < _JOINT_LIMITS_DEG[:, 0]) or np.any(
            deg > _JOINT_LIMITS_DEG[:, 1]
        ):
            raise ValueError(f"{path}: 第一帧关节角超出 Piper 限位：{deg}")
        starts.append(DemoStart(path, selected, joint, eef_pose))
    return starts


def _rank_starts(starts, current):
    def score(start):
        delta_deg = np.degrees(np.abs(start.joint - current))
        return float(delta_deg.max()), float(np.linalg.norm(delta_deg))

    return sorted(starts, key=score)


def _select_start(starts, current, episode):
    if episode is None:
        return _rank_starts(starts, current)[0]
    wanted = str(episode)
    matches = [s for s in starts if s.episode == wanted]
    if not matches:
        available = ", ".join(s.episode for s in starts)
        raise ValueError(f"找不到 episode {wanted}.hdf5；可选：{available}")
    return matches[0]


def _find_start_route(starts, current, target, max_step_deg):
    """用示教首帧作安全路标，寻找每轴变化均不超过阈值的最短路线。"""
    def distance(a, b):
        delta = np.degrees(np.abs(a - b))
        return float(delta.max()), float(np.linalg.norm(delta))

    by_episode = {start.episode: start for start in starts}
    queue = deque()
    visited = set()
    first_hops = sorted(
        (
            (distance(current, start.joint), start)
            for start in starts
            if distance(current, start.joint)[0] <= max_step_deg
        ),
        key=lambda item: item[0],
    )
    for _, start in first_hops:
        queue.append((start, [start]))
        visited.add(start.episode)

    while queue:
        node, path = queue.popleft()
        if node.episode == target.episode:
            return path
        neighbors = sorted(
            (
                (distance(node.joint, candidate.joint), candidate)
                for candidate in starts
                if candidate.episode not in visited
                and distance(node.joint, candidate.joint)[0]
                <= max_step_deg
            ),
            key=lambda item: item[0],
        )
        for _, candidate in neighbors:
            visited.add(candidate.episode)
            queue.append((candidate, path + [candidate]))

    available = ", ".join(sorted(by_episode))
    raise RuntimeError(
        f"找不到每轴步长 <= {max_step_deg:.2f}°、通往 episode "
        f"{target.episode} 的首帧路线；可用 episode：{available}"
    )


def _print_route(current, route):
    source = current
    source_name = "current"
    print("\n分段路线：")
    for start in route:
        delta = np.degrees(start.joint - source)
        print(
            f"  {source_name:>8} -> episode {start.episode:>3}: "
            f"max={np.abs(delta).max():5.2f}° "
            f"delta={np.round(delta, 2)}"
        )
        source = start.joint
        source_name = start.episode


def _print_selection(starts, current, selected):
    ranked = _rank_starts(starts, current)
    print(f"当前关节角 (deg): {np.round(np.degrees(current), 3)}")
    print("最近的示教首帧：")
    for start in ranked[:5]:
        delta = np.degrees(np.abs(start.joint - current))
        marker = "  <-- 选择" if start.path == selected.path else ""
        print(
            f"  episode {start.episode:>3}: max={delta.max():5.2f}° "
            f"L2={np.linalg.norm(delta):5.2f}°{marker}"
        )
    delta = np.degrees(selected.joint - current)
    print(f"\n目标文件: {selected.path}")
    print(f"目标关节角 (deg): {np.round(np.degrees(selected.joint), 3)}")
    print(f"逐关节变化 (deg): {np.round(delta, 3)}")
    print(f"目标 EEF xyz (mm，仅参考): {np.round(selected.eef_pose[:3] * 1000, 2)}")


def _enable(sdk, timeout=8.0):
    """使能六轴，并等待使能命令之后的新反馈，避免命中 reset 前的 SDK 缓存。"""
    low_before = sdk.GetArmLowSpdInfoMsgs().time_stamp
    status_before = sdk.GetArmStatus().time_stamp
    deadline = time.monotonic() + timeout
    stable_since = None
    last_print = 0.0
    while time.monotonic() < deadline:
        sdk.EnableArm(7)
        # 0.2s 轮询要 4 轮才够到下面 0.5s 的稳定窗口（向上取整浪费 0.3s）。
        # 降到 0.1s 只是把同一个 0.5s 窗口测得更细，判据本身没有放宽。
        time.sleep(0.1)
        low = sdk.GetArmLowSpdInfoMsgs()
        flags = _driver_enable_flags(sdk)
        status_msg = sdk.GetArmStatus()
        status = status_msg.arm_status
        ready = (
            low.time_stamp > low_before
            and status_msg.time_stamp > status_before
            and all(flags)
            and int(status.arm_status) == 0
            and int(status.err_code) == 0
        )
        now = time.monotonic()
        if ready:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= 0.5:
                print("六个驱动器已使能，状态正常且稳定。")
                return
        else:
            stable_since = None
        if now - last_print >= 0.5:
            print(
                "等待使能："
                f"enabled={list(flags)}, "
                f"arm_status={status.arm_status}, err_code={status.err_code}"
            )
            last_print = now
    status = sdk.GetArmStatus().arm_status
    raise RuntimeError(
        f"{timeout:.1f}s 内六个关节未全部使能；"
        f"enabled={list(_driver_enable_flags(sdk))}, "
        f"arm_status={status.arm_status}, err_code={status.err_code}"
    )


def _wait_reset_settled(sdk, status_before, low_before, timeout=8.0):
    """等待 reset 后的新反馈稳定，不能把 reset 前缓存误认为当前状态。"""
    deadline = time.monotonic() + timeout
    stable_since = None
    last_print = 0.0
    while time.monotonic() < deadline:
        status_msg = sdk.GetArmStatus()
        status = status_msg.arm_status
        low = sdk.GetArmLowSpdInfoMsgs()
        flags = _driver_enable_flags(sdk)
        fresh = (
            status_msg.time_stamp > status_before
            and low.time_stamp > low_before
        )
        settled = (
            fresh
            and not any(flags)
            and int(status.arm_status) == 0
            and int(status.err_code) == 0
        )
        now = time.monotonic()
        if settled:
            if stable_since is None:
                stable_since = now
            if now - stable_since >= 0.5:
                print("reset 后状态已稳定：六轴失能、控制器正常。")
                return
        else:
            stable_since = None
        if now - last_print >= 0.5:
            print(
                "等待 reset 稳定："
                f"enabled={list(flags)}, "
                f"arm_status={status.arm_status}, err_code={status.err_code}"
            )
            last_print = now
        time.sleep(0.1)
    status = sdk.GetArmStatus().arm_status
    raise RuntimeError(
        f"{timeout:.1f}s 内 reset 状态未稳定；"
        f"enabled={list(_driver_enable_flags(sdk))}, "
        f"arm_status={status.arm_status}, err_code={status.err_code}"
    )


def _controller_reset(sdk):
    print("\n即将清除高跟随/示教遗留状态；机械臂会瞬间失力。")
    answer = input("托住机械臂并准备好急停后，输入 RESET 回车：").strip()
    if answer != "RESET":
        raise SystemExit("未确认，未发送 reset")
    status_before = sdk.GetArmStatus().time_stamp
    low_before = sdk.GetArmLowSpdInfoMsgs().time_stamp
    if hasattr(sdk, "ResetPiper"):
        sdk.ResetPiper()
    else:
        sdk.MotionCtrl_1(0x02, 0x00, 0x00)
    _wait_reset_settled(sdk, status_before, low_before)


def _move_to_start(sdk, target, speed, tolerance_deg, timeout, step_deg=15.0):
    """分段走到 target。

    固件收到 JointCtrl 后自己规划整段轨迹，所以直接把远端目标丢过去就是一次
    大扫。这里先按 step_deg 把当前位姿到目标切成若干中间点逐段下发，每段都做
    到位/固件状态/驱动使能检查，任意跨度都不会变成一次无监控的大幅摆动。
    step_deg <= 0 时退回一次性下发。

    中间点只用来限制固件一次能规划多远，不是要去的地方，所以进到半个 step 以内
    就换下一个目标，让运动连续流过去。曾经按 2° 逐点停稳，8 段就多出 8 次停顿，
    这部分开销不随 --speed 缩短，是复位慢的另一半原因。只有最后一段用调用方给
    的精确容差。
    """
    sdk.MotionCtrl_2(0x01, 0x01, speed, 0x00)
    start = _joint_feedback(sdk)
    gap_deg = np.degrees(np.abs(target - start)).max()
    legs = 1 if step_deg <= 0 else max(1, int(np.ceil(gap_deg / step_deg)))
    advance_deg = max(tolerance_deg, step_deg * 0.5)
    for leg in range(1, legs + 1):
        sub_target = start + (target - start) * (leg / legs)
        sub_tol = tolerance_deg if leg == legs else advance_deg
        current, error_deg = _drive_to(sdk, sub_target, sub_tol, timeout)
    return current, np.degrees(np.abs(target - current))


def _drive_to(sdk, target, tolerance_deg, timeout):
    raw = [int(round(v * 1000)) for v in np.degrees(target)]
    deadline = time.monotonic() + timeout
    last_print = 0.0
    while time.monotonic() < deadline:
        sdk.JointCtrl(*raw)
        # 20Hz(0.05s) 时到位检测的粒度本身成了瓶颈：speed 50 以上总耗时不再下降，
        # 因为每段都要多等半个轮询周期。50Hz 仍远低于 policy 的 30Hz 下发压力。
        time.sleep(0.02)
        current = _joint_feedback(sdk)
        error_deg = np.degrees(np.abs(target - current))
        now = time.monotonic()
        if now - last_print >= 0.5:
            eef_z = sdk.GetArmEndPoseMsgs().end_pose.Z_axis * 1e-3
            print(
                f"\rmax误差={error_deg.max():5.2f}° "
                f"eef_z={eef_z:6.1f}mm",
                end="",
                flush=True,
            )
            last_print = now
        status = sdk.GetArmStatus().arm_status
        if int(status.arm_status) != 0 or int(status.err_code) != 0:
            print()
            raise RuntimeError(
                f"固件异常：arm_status={status.arm_status}, err_code={status.err_code}"
            )
        if not _drivers_enabled(sdk):
            print()
            raise RuntimeError("移动中有驱动器失能")
        if error_deg.max() <= tolerance_deg:
            print()
            return current, error_deg
    print()
    current = _joint_feedback(sdk)
    error_deg = np.degrees(np.abs(target - current))
    raise RuntimeError(
        f"{timeout:.1f}s 内未到位；最大关节误差 {error_deg.max():.2f}°，"
        f"逐关节误差 {np.round(error_deg, 2)}°"
    )


def _parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dataset", default=None,
                    help="从这个目录的 HDF5 首帧取起点；不传就用写死的固定起点，"
                         "不需要本机存任何数据")
    ap.add_argument("--arm", default=None, help="HDF5 中的机械臂名；默认优先 right_arm")
    ap.add_argument("--episode", type=int, default=None, help="固定 episode；默认自动选最近首帧")
    ap.add_argument(
        "--route",
        action="store_true",
        help="经其他示教首帧分段移动到指定 episode，每段不超过最大起点差",
    )
    ap.add_argument("--port", default="can0")
    # 原默认 10（=臂的十分之一速度），复位一段 100° 要十几秒。移动本身按
    # --step-deg 分段且逐段查固件状态/驱动使能，提速不会变成一次无监控大扫。
    # 对比：policy 路径 configs/piper_single.yaml 用的是 speed=100。
    ap.add_argument("--speed", type=int, default=50, help="MOVE J 速度百分比，默认 50")
    ap.add_argument("--tolerance-deg", type=float, default=0.5)
    ap.add_argument("--timeout", type=float, default=20.0)
    ap.add_argument(
        "--max-start-gap-deg",
        type=float,
        default=_DEFAULT_MAX_START_GAP_DEG,
        help="当前姿态到目标首帧的最大单关节差，超出则拒绝移动；"
             "默认不限制；传正数可启用额外的总跨度拒绝"
             "（移动本身按 --step-deg 分段，任意跨度都不是一次大扫）",
    )
    ap.add_argument(
        "--step-deg",
        type=float,
        default=_DEFAULT_STEP_DEG,
        help="移动时每段的最大单关节跨度，默认 15°；<=0 退回一次性下发整段",
    )
    ap.add_argument(
        "--current-deg",
        type=float,
        nargs=6,
        default=None,
        metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
        help="仅离线预览：手工给当前关节角，不连接 CAN",
    )
    ap.add_argument("--execute", action="store_true", help="实际下发；默认只预览")
    # 非交互执行。交互确认问的是现场安全，由调用方承担；脚本自身的前置检查
    # （固件状态、驱动保护、路线单步上限、到位容差、超时）照常执行。
    ap.add_argument("--assume-safe", action="store_true",
                    help="跳过交互确认，直接执行（现场安全由调用方保证）")
    reset = ap.add_mutually_exclusive_group()
    reset.add_argument(
        "--controller-reset",
        action="store_true",
        help="移动前清除高跟随/示教遗留状态（会瞬间失力）",
    )
    reset.add_argument(
        "--skip-controller-reset",
        action="store_true",
        help="确认从臂已在主臂断开后单独重启，跳过控制器 reset",
    )
    args = ap.parse_args()
    if not 1 <= args.speed <= 100:
        ap.error("--speed 必须在 1..100")
    if args.tolerance_deg <= 0 or args.timeout <= 0:
        ap.error("容差和超时必须为正数")
    if args.route and args.max_start_gap_deg <= 0:
        # --route 的分段搜索就是以"每段不超过最大起点差"为目标函数的，没有上限
        # 就无从选中间点。
        ap.error("--route 需要正的 --max-start-gap-deg")
    if args.execute and args.current_deg is not None:
        ap.error("--execute 时必须读取真机反馈，不能使用 --current-deg")
    if args.execute and not (args.controller_reset or args.skip_controller_reset):
        ap.error(
            "--execute 必须显式选择 --controller-reset 或 --skip-controller-reset"
        )
    if args.route and args.episode is None:
        ap.error("--route 必须与 --episode 一起使用")
    return args


def main():
    args = _parse_args()
    # --route 要用其他 episode 首帧当中间点，必须全量读；否则指定了 episode 就
    # 只读那一个（"最近的示教首帧"参考列表随之只剩选中项，不影响实际动作）。
    if args.dataset is None:
        if args.episode is not None or args.route:
            raise SystemExit(
                "--episode / --route 需要真实数据集；用 --dataset 指定，"
                "或去掉这两个开关走固定起点"
            )
        starts = _fixed_demo_start()
        print(f"起点：固定值（不读数据集）{np.degrees(starts[0].joint).round(3)}°")
    else:
        only = args.episode if (args.episode is not None and not args.route) else None
        starts = _load_demo_starts(args.dataset, args.arm, only_episode=only)

    if args.current_deg is not None:
        current = np.radians(np.asarray(args.current_deg, dtype=np.float64))
        selected = _select_start(starts, current, args.episode)
        _print_selection(starts, current, selected)
        route = (
            _find_start_route(
                starts,
                current,
                selected,
                args.max_start_gap_deg,
            )
            if args.route
            else [selected]
        )
        if args.route:
            _print_route(current, route)
        print("\n离线预览完成；未连接 CAN。")
        return 0

    from piper_sdk import C_PiperInterface_V2

    sdk = C_PiperInterface_V2(args.port)
    sdk.ConnectPort(piper_init=False)
    try:
        _wait_feedback(sdk)
        _assert_no_external_controller(sdk)
        current = _joint_feedback(sdk)
        selected = _select_start(starts, current, args.episode)
        _print_selection(starts, current, selected)
        route = (
            _find_start_route(
                starts,
                current,
                selected,
                args.max_start_gap_deg,
            )
            if args.route
            else [selected]
        )
        if args.route:
            _print_route(current, route)

        initial_gap = np.degrees(
            np.abs(route[0].joint - current)
        ).max()
        if args.max_start_gap_deg > 0 and initial_gap > args.max_start_gap_deg:
            raise RuntimeError(
                f"第一段最大单关节差 {initial_gap:.2f}° > "
                f"--max-start-gap-deg={args.max_start_gap_deg:.2f}°，拒绝移动。 "
                f"复位本身按 --step-deg={args.step_deg:g}° 分段执行，"
                f"确认现场无障碍后可放大该值或传 --max-start-gap-deg 0 解除限制"
            )
        if not args.execute:
            print("\n预览完成；监听期间未发现外部标准控制帧，全程未向 CAN 发送任何帧。")
            print(
                "确认主臂已断开后，按现场状态选择 "
                "--execute --controller-reset 或 --execute --skip-controller-reset。"
            )
            return 0

        print("\n执行条件：主臂已断开；机械臂周围无人员/障碍；手边有急停。")
        route_text = " -> ".join(
            f"episode {waypoint.episode}" for waypoint in route
        )
        if args.assume_safe:
            print(f"--assume-safe：跳过交互确认，将以 speed={args.speed}% "
                  f"按 {route_text} 移动。")
        else:
            answer = input(
                f"将以 speed={args.speed}% 按 {route_text} 移动，"
                "输入 MOVE 回车："
            ).strip()
            if answer != "MOVE":
                raise SystemExit("未确认，未发送运动指令")

        if args.controller_reset:
            _controller_reset(sdk)

        sdk.MotionCtrl_2(0x01, 0x01, args.speed, 0x00)
        _enable(sdk)

        # reset 后机械臂可能因短暂失力改变姿态；必须重新选择/检查目标距离。
        current = _joint_feedback(sdk)
        selected = _select_start(starts, current, args.episode)
        route = (
            _find_start_route(
                starts,
                current,
                selected,
                args.max_start_gap_deg,
            )
            if args.route
            else [selected]
        )
        post_reset_gap = np.degrees(
            np.abs(route[0].joint - current)
        ).max()
        print(
            f"使能后当前关节角: {np.round(np.degrees(current), 3)}°；"
            f"第一段 episode={route[0].episode}，max差={post_reset_gap:.2f}°"
        )
        if args.max_start_gap_deg > 0 and post_reset_gap > args.max_start_gap_deg:
            raise RuntimeError(
                f"使能后目标差增至 {post_reset_gap:.2f}°，超过 "
                f"{args.max_start_gap_deg:.2f}°，拒绝移动"
            )

        final = current
        error_deg = np.zeros(6, dtype=np.float64)
        for leg, waypoint in enumerate(route, 1):
            print(
                f"\n路线 {leg}/{len(route)}：移动到 "
                f"episode {waypoint.episode} 第一帧"
            )
            final, error_deg = _move_to_start(
                sdk,
                waypoint.joint,
                args.speed,
                args.tolerance_deg,
                args.timeout,
                args.step_deg,
            )
            print(
                f"已到 episode {waypoint.episode}，"
                f"max误差={error_deg.max():.3f}°"
            )
        print(f"已到 episode {selected.episode} 第一帧。")
        print(f"最终关节角 (deg): {np.round(np.degrees(final), 3)}")
        print(f"逐关节误差 (deg): {np.round(error_deg, 3)}")
        print("机械臂保持使能；脚本退出不会主动失能。")
        return 0
    finally:
        sdk.DisconnectPort()


if __name__ == "__main__":
    sys.exit(main())
