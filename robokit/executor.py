"""Action chunk 执行器：把 policy 返回的动作块落到机械臂上。

从部署客户端抽出来独立成类，让真机部署与无硬件测试复用同一份执行语义，
避免递推基准、限幅和时序逻辑在两条路径中分叉。

三种 chunk 递推基准（chunk_base），是当前"动作诡异"排查的核心假设分支：

    feedback    每一步都重新读反馈位姿当基准（历史行为）。反馈滞后于命令时，
                future delta 被反复套在滞后位姿上，整块动作被压缩
                （L1 实测路径长度只有示教的 0.34 倍）—— 已确认是 bug。
    recursive   块内沿上一条 commanded target 递推，每个新 chunk 用「模型看到的那帧
                观测位姿」重新对齐。与"反应式模型把 delta 锚定在它看到的位姿上"的
                语义一致。
    continuous  跨 chunk 也不重新对齐，一直沿 commanded target 递推。在前馈回放中
                能完美复现示教轨迹，但该结果部分来自回放 policy 的前馈构造；
                真实反应式模型下 recursive vs continuous 的优劣需真机低速 A/B 判定。
                continuous 永不重新锚定，真机必须配
                resync_tracking_err 或跟踪误差中止兜底。

反馈位姿在 recursive/continuous 下仍然每步读取，只用于跟踪误差监控与安全中止，
不作为 future trajectory 的数学基准。

单位约定：位姿与动作全链路都是米 / 真弧度 / xyz 外旋欧拉序（见 robokit/pose.py）。
模型输出的旋转 delta 直接就是弧度，guard 和 trace 统计不需要任何换算。
"""
import time

import numpy as np
from scipy.spatial.transform import Rotation

from robokit.arms.base import ArmCommandRejected
from robokit.pose import apply_local_delta_pose, wrap_euler
from robokit.safety import GuardViolation

BASE_MODES = ("feedback", "recursive", "continuous")
GRIPPER_MODES = ("raw", "binary", "hysteresis")


def apply_eef_delta(current, delta, wrap_target=False):
    """对当前 EEF 位姿应用局部增量，返回要下发的目标位姿。

    用 apply_local_delta_pose（局部坐标系变换），与 RLDS 转换生成训练动作时用的
    local_delta_pose 严格互逆。旧 memoryvla_client.py 当年用的是逐元素相加
    current+delta（近似，仅当基座姿态≈0 时才等价），此处不复刻该近似。
    """
    target = apply_local_delta_pose(current, delta)
    return wrap_euler(target) if wrap_target else target


class ChunkExecutor:
    """按行执行 action chunk。一个 executor 对应一次部署会话，跨 chunk 保持递推基准。

    参数:
        action_space   "eef_delta" | "joint"
        control_freq   下发频率 Hz
        fixed_control_rate
                       True 时按绝对 deadline 定频，扣除每步软件处理耗时；
                       False 保留历史的“处理完成后再 sleep 一个周期”
        horizon        每个 chunk 实际执行的前 N 步
        chunk_base     见模块 docstring
        gripper_mode   raw 透传 / binary 0.5 二值 / hysteresis 带死区迟滞
                       （注意：hysteresis 对已二值化的 0/1 输入无效）
        gripper_rate   夹爪单步变化上限（满行程比例，None=不限）。示教数据单帧最大
                       变化约 0.11，限速可把服务端残留的 0/1 跳变摊成斜坡而不是
                       一步打满 70mm 行程
        guard          robokit.safety.ActionGuard 或 None（None = 不做任何限幅，历史行为）
        guard_tracking 是否把真实反馈相对上一目标的落后量交给 guard。仅 dry-run 应关闭：
                       dry-run 不发送目标，静止反馈不可能跟随虚拟轨迹
        trace          robokit.trace.TraceWriter 或 None
        clock/sleep    注入时钟，供无硬件测试做确定性回放
        interrupt      返回 True 表示用户请求中断
    """

    def __init__(self, action_space="eef_delta", control_freq=30, horizon=30,
                 chunk_base="recursive", gripper_mode="raw",
                 gripper_deadband=0.25, gripper_rate=None, guard=None, trace=None,
                 resync_tracking_err=None, wrap_target=False, wait_arrival=None,
                 arrival_rotation_tol_deg=1.0, arrival_joint_tol_deg=0.5,
                 arrival_stable_s=0.1, guard_tracking=True,
                 fixed_control_rate=False,
                 clock=time.time, sleep=time.sleep, interrupt=None):
        if chunk_base not in BASE_MODES:
            raise ValueError(f"chunk_base must be one of {BASE_MODES}, got {chunk_base!r}")
        if gripper_mode not in GRIPPER_MODES:
            raise ValueError(f"gripper_mode must be one of {GRIPPER_MODES}, got {gripper_mode!r}")
        self.action_space = action_space
        self.control_freq = float(control_freq)
        self.fixed_control_rate = bool(fixed_control_rate)
        self.horizon = int(horizon)
        self.chunk_base = chunk_base
        # (位置容差 m, 单步超时 s)；None = 沿用历史行为（发完就走，不等到位）
        self.wait_arrival = None if wait_arrival is None else (
            float(wait_arrival[0]), float(wait_arrival[1]))
        self.arrival_rotation_tol_deg = float(arrival_rotation_tol_deg)
        self.arrival_joint_tol_deg = float(arrival_joint_tol_deg)
        self.arrival_stable_s = float(arrival_stable_s)
        if (
            self.arrival_rotation_tol_deg <= 0
            or self.arrival_joint_tol_deg <= 0
            or self.arrival_stable_s < 0
        ):
            raise ValueError("到位姿态/关节容差必须为正，稳定时间不能为负")
        self.gripper_mode = gripper_mode
        self.gripper_deadband = float(gripper_deadband)
        self.gripper_rate = None if gripper_rate is None else float(gripper_rate)
        self.guard = guard
        self.guard_tracking = bool(guard_tracking)
        self.trace = trace
        # continuous 基准的兜底：机械臂真的跟不上时（接触、限位、限速），命令系与实际位姿
        # 的差会持续变大。超过这个米数就把基准拉回反馈位姿重新对齐，避免命令跑飞。
        # None = 不重新对齐，完全交给 guard 的 tracking 中止。
        self.resync_tracking_err = resync_tracking_err
        # 命令欧拉角是否折回 (-π, π]。默认 False = 历史行为（旧 memoryvla_client 也是
        # 无界累加）。沿欧拉角累加增量时命令角会越过 ±180° 继续增大（实测 policy 的
        # yaw 到 -202°）；真弧度下折回严格保持姿态（见 pose.wrap_euler），但数值会在
        # 边界跳 360°。host IK 按旋转矩阵求解，两种欧拉表示对应相同物理姿态。
        self.wrap_target = bool(wrap_target)
        self.clock = clock
        self.sleep = sleep
        self.interrupt = interrupt or (lambda: False)

        self.step_index = 0
        self.chunk_index = 0
        self._base = {}        # arm -> 上一条 commanded EEF target
        self._last_target = {}  # arm -> 上一条 commanded target（跟踪误差用）
        self._gripper_cmd = {}  # arm -> 上一条夹爪命令

    def reset(self):
        """新 episode / 新连接：清空递推基准与迟滞状态（含 guard 的跨步状态）。"""
        self.step_index = 0
        self.chunk_index = 0
        self._base.clear()
        self._last_target.clear()
        self._gripper_cmd.clear()
        if self.guard is not None:
            self.guard.reset()

    # ── 基准选择 ────────────────────────────────────────────────────────────
    def _resolve_base(self, name, obs_pose, feedback_pose):
        """返回本步递推基准。obs_pose 是模型推理所用那帧观测里的位姿。"""
        if self.chunk_base == "feedback":
            return np.asarray(feedback_pose, dtype=np.float64), "feedback"
        prev = self._base.get(name)
        if prev is None:
            src = obs_pose if obs_pose is not None else feedback_pose
            return np.asarray(src, dtype=np.float64), "obs" if obs_pose is not None else "feedback"
        return np.asarray(prev, dtype=np.float64), "commanded"

    def _resolve_gripper(self, name, value):
        value = float(np.clip(value, 0.0, 1.0))
        prev = self._gripper_cmd.get(name)
        if self.gripper_mode == "binary":
            value = float(value > 0.5)
        elif self.gripper_mode == "hysteresis":
            if prev is None:
                value = float(value > 0.5)
            elif abs(value - prev) < self.gripper_deadband:
                value = prev
        # 单步限速斜坡（P0-2）：模式解析后统一生效。训练标签是连续绝对开度，示教单帧
        # 最大变化约 0.11 满行程；限速把任何来源的开度跳变摊成多步斜坡，而不是硬阈值。
        if self.gripper_rate is not None and prev is not None:
            value = float(np.clip(value, prev - self.gripper_rate, prev + self.gripper_rate))
        return value

    # ── 执行 ────────────────────────────────────────────────────────────────
    def execute(self, robot, chunk, obs=None):
        """执行一个 chunk 的前 horizon 步。

        obs 是产生这个 chunk 的那帧观测（robot.get_obs() 的返回值）。recursive 模式用它
        在每个 chunk 开头把基准重新对齐到「模型真正看到的位姿」。

        返回 ("ok"|"interrupted"|"aborted", 详情 dict)。
        """
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.ndim != 2:
            return self._abort("chunk_shape", f"expected 2-D chunk, got shape {chunk.shape}")
        if not np.all(np.isfinite(chunk)):
            bad = int(np.count_nonzero(~np.isfinite(chunk)))
            return self._abort("chunk_nonfinite", f"{bad} non-finite values in chunk")

        arm_names = sorted(robot.arms)
        if self.chunk_base == "recursive":
            self._base.clear()      # 每个 chunk 重新对齐到本轮观测位姿

        period = 1.0 / self.control_freq
        next_deadline = self.clock()
        for row, action in enumerate(chunk[:self.horizon]):
            offset = 0
            for name in arm_names:
                arm = robot.arms[name]
                if self.action_space == "joint":
                    width = arm.dof + 1
                    status = self._step_joint(arm, name, action[offset:offset + width], row)
                else:
                    width = 7
                    obs_pose = None
                    if obs is not None and name in obs.get("arms", {}):
                        obs_pose = obs["arms"][name]["eef_pose"]
                    status = self._step_eef(arm, name, action[offset:offset + 7], obs_pose, row)
                if status is not None:
                    return status
                offset += width
            self.step_index += 1
            if self.fixed_control_rate:
                next_deadline += period
                now = self.clock()
                remaining = next_deadline - now
                if remaining > 0:
                    self.sleep(remaining)
                else:
                    # Never issue a burst of catch-up commands after an
                    # overrun; resume the 30Hz grid from the current time.
                    next_deadline = now
            else:
                self.sleep(period)
            if self.interrupt():
                return "interrupted", {"step": self.step_index}
        self.chunk_index += 1
        return "ok", {"steps": min(len(chunk), self.horizon)}

    def _step_eef(self, arm, name, action, obs_pose, row):
        state = arm.get_state()
        feedback = state["eef_pose"]
        base, base_src = self._resolve_base(name, obs_pose if row == 0 else None, feedback)

        prev = self._last_target.get(name)
        if (self.resync_tracking_err is not None and prev is not None and feedback is not None
                and np.linalg.norm(np.asarray(feedback)[:3] - prev[:3]) > self.resync_tracking_err
                and base_src == "commanded"):
            base, base_src = np.asarray(feedback, dtype=np.float64), "resync"

        delta = np.asarray(action[:6], dtype=np.float64)
        target = apply_eef_delta(base, delta, self.wrap_target)
        gripper_raw = float(action[6])
        gripper = self._resolve_gripper(name, gripper_raw)

        tracking = None
        prev_target = self._last_target.get(name)
        if prev_target is not None and feedback is not None:
            tracking = float(np.linalg.norm(np.asarray(feedback)[:3] - prev_target[:3]))

        record = {
            "t": self.clock(), "chunk": self.chunk_index, "row": row, "step": self.step_index,
            "arm": name, "base_src": base_src, "base": base.tolist(), "delta": delta.tolist(),
            "target": target.tolist(), "gripper_raw": gripper_raw, "gripper_cmd": gripper,
            "gripper_fb": state.get("gripper"),
            "gripper_oob": bool(gripper_raw < -0.005 or gripper_raw > 1.005),
            "feedback": None if feedback is None else np.asarray(feedback).tolist(),
            "tracking_err_m": tracking,
            "tracking_guarded": self.guard_tracking,
            # 记录命令欧拉角是否已跑出 (-π, π]，供事后分析累计量。
            "euler_out_of_range": bool(np.max(np.abs(target[3:])) > np.pi),
        }
        if self.guard is not None:
            try:
                self.guard.check_eef(delta=delta,
                                     target=target, feedback=feedback,
                                     gripper=gripper, prev_gripper=self._gripper_cmd.get(name),
                                     tracking_err=(
                                         tracking if self.guard_tracking else None
                                     ),
                                     arm=name)
            except GuardViolation as e:
                record["violation"] = str(e)
                self._emit(record)
                return self._abort(e.kind, str(e))

        try:
            command = arm.move_eef(target, gripper)
        except ArmCommandRejected as exc:
            record["violation"] = str(exc)
            self._emit(record)
            return self._abort(exc.kind, str(exc))
        if command is not None:
            record["arm_command"] = command
        try:
            reached = self._wait_arrival(arm, target, command)
        except ArmCommandRejected as exc:
            record["violation"] = str(exc)
            self._emit(record)
            return self._abort(exc.kind, str(exc))
        if reached is not None:
            record.update(reached)
            if reached["arrival_timeout"]:
                self._emit(record)
                return self._abort(
                    "arrival_timeout",
                    f"[{name}] 到位超时："
                    f"xyz={reached['arrival_err_m']*1000:.2f}mm, "
                    f"rot={reached['arrival_rotation_err_deg']:.2f}°, "
                    f"joint={reached['arrival_joint_err_deg']}",
                )
        # 等到位模式下，递推基准用**真实到达位姿**而不是命令目标：delta 是局部坐标系量，
        # 基准姿态一旦偏离真实值，后续 delta 的方向就跟着转偏（实测 chunk 内可放大到
        # 104mm / 17°）。等到位后两者几乎重合，这里取实测值把残余误差也吃掉。
        if self.wait_arrival is not None:
            fb_now = arm.get_state()["eef_pose"]
            self._base[name] = np.asarray(fb_now, dtype=np.float64)
        else:
            self._base[name] = target
        self._last_target[name] = target
        self._gripper_cmd[name] = gripper
        self._emit(record)
        return None

    def _wait_arrival(self, arm, target, command=None):
        """阻塞到机械臂稳定到达完整目标；超时由调用方立即中止。

        按固定周期连发时，命令位姿可能跑在真实位姿前面。等到位把这条链路变成闭环，
        既消掉累积落后，也让下一步的局部 delta 落在正确的姿态基准上。

        host IK 路径同时验收实际 JointCtrl 目标、XYZ 和 SO(3) 姿态。控制器错误、
        驱动保护或反馈陈旧由 arm 的可选 ``assert_healthy`` 钩子立即转成拒绝，
        不会在超时后继续追加下一条命令。
        """
        if self.wait_arrival is None:
            return None
        tol, timeout = self.wait_arrival
        command = command or {}
        requested_target = np.asarray(target, dtype=np.float64)
        realized_target = command.get("eef_target_realized")
        if (
            command.get("backend")
            in {"host_ik_move_j", "pinocchio_ik_move_j"}
            and realized_target is not None
        ):
            # EEF IK does not send the ideal EEF pose.  It sends a quantized
            # joint target whose FK pose may legally differ from that request.
            # Arrival must track the pose represented by the exact JointCtrl
            # payload, computed by the selected production backend's own FK.
            arrival_target = np.asarray(realized_target, dtype=np.float64)
            if arrival_target.shape != (6,) or not np.all(np.isfinite(arrival_target)):
                raise ArmCommandRejected(
                    "host_ik_arrival_target",
                    "host IK returned an invalid realized EEF target",
                )
            arrival_target_source = (
                "pinocchio_fk"
                if command.get("backend") == "pinocchio_ik_move_j"
                else "host_ik_fk"
            )
        else:
            arrival_target = requested_target
            arrival_target_source = "requested"
        joint_target_deg = command.get("joint_target_deg")
        joint_target = (
            None
            if joint_target_deg is None
            else np.radians(np.asarray(joint_target_deg, dtype=np.float64))
        )
        t0 = self.clock()
        err = float("inf")
        rotation_err_deg = float("inf")
        requested_err = float("inf")
        requested_rotation_err_deg = float("inf")
        joint_err_deg = None
        stable_since = None
        while True:
            if hasattr(arm, "assert_healthy"):
                arm.assert_healthy("arrival")
            state = arm.get_state()
            fb = np.asarray(state["eef_pose"], dtype=np.float64)
            err = float(np.linalg.norm(fb[:3] - arrival_target[:3]))
            actual_rotation = Rotation.from_euler("xyz", fb[3:])
            target_rotation = Rotation.from_euler("xyz", arrival_target[3:])
            rotation_err_deg = float(
                np.degrees(
                    (actual_rotation.inv() * target_rotation).magnitude()
                )
            )
            requested_err = float(
                np.linalg.norm(fb[:3] - requested_target[:3])
            )
            requested_rotation = Rotation.from_euler(
                "xyz", requested_target[3:]
            )
            requested_rotation_err_deg = float(
                np.degrees(
                    (actual_rotation.inv() * requested_rotation).magnitude()
                )
            )
            if joint_target is not None:
                joint_err_deg = float(
                    np.degrees(
                        np.abs(
                            np.asarray(state["joint"], dtype=np.float64)
                            - joint_target
                        )
                    ).max()
                )
            arrived = (
                err <= tol
                and rotation_err_deg <= self.arrival_rotation_tol_deg
                and (
                    joint_err_deg is None
                    or joint_err_deg <= self.arrival_joint_tol_deg
                )
            )
            now = self.clock()
            if arrived:
                if stable_since is None:
                    stable_since = now
                if now - stable_since >= self.arrival_stable_s:
                    return {
                        "arrival_err_m": err,
                        "arrival_rotation_err_deg": rotation_err_deg,
                        "arrival_joint_err_deg": joint_err_deg,
                        "arrival_requested_err_m": requested_err,
                        "arrival_requested_rotation_err_deg":
                            requested_rotation_err_deg,
                        "arrival_target_source": arrival_target_source,
                        "arrival_timeout": False,
                        "arrival_stable_s": now - stable_since,
                    }
            else:
                stable_since = None
            if now - t0 >= timeout:
                return {
                    "arrival_err_m": err,
                    "arrival_rotation_err_deg": rotation_err_deg,
                    "arrival_joint_err_deg": joint_err_deg,
                    "arrival_requested_err_m": requested_err,
                    "arrival_requested_rotation_err_deg":
                        requested_rotation_err_deg,
                    "arrival_target_source": arrival_target_source,
                    "arrival_timeout": True,
                    "arrival_stable_s": 0.0,
                }
            self.sleep(0.005)

    def _step_joint(self, arm, name, action, row):
        joint = np.asarray(action[:arm.dof], dtype=np.float64)
        gripper = self._resolve_gripper(name, action[arm.dof])
        record = {"t": self.clock(), "chunk": self.chunk_index, "row": row,
                  "step": self.step_index, "arm": name, "joint": joint.tolist(),
                  "gripper_raw": float(action[arm.dof]), "gripper_cmd": gripper}
        if self.guard is not None:
            try:
                self.guard.check_joint(joint=joint, feedback=arm.get_state()["joint"],
                                       gripper=gripper, prev_gripper=self._gripper_cmd.get(name),
                                       arm=name)
            except GuardViolation as e:
                record["violation"] = str(e)
                self._emit(record)
                return self._abort(e.kind, str(e))
        try:
            command = arm.move_joint(joint, gripper)
        except ArmCommandRejected as exc:
            record["violation"] = str(exc)
            self._emit(record)
            return self._abort(exc.kind, str(exc))
        if command is not None:
            record["arm_command"] = command
        self._gripper_cmd[name] = gripper
        self._emit(record)
        return None

    def _abort(self, kind, message):
        if self.trace is not None:
            self.trace.write({"t": self.clock(), "event": "abort", "kind": kind, "message": message})
        return "aborted", {"kind": kind, "message": message}

    def _emit(self, record):
        if self.trace is not None:
            self.trace.write(record)
