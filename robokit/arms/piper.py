"""Agilex Piper 机械臂（piper_sdk，CAN 总线）。

SDK 原始单位: 关节/姿态 0.001 度，位置 0.001 毫米，夹爪行程 0.001 毫米（满行程 70mm）。
本类对外统一为 弧度 / 米 / [0,1]。

注意：旧代码 (Controller/Piper_controller.py) 的 EEF 姿态换算是 raw*1e-6（得到 度/1000），
并非真正的弧度。本实现修正为真弧度，因此旋转维度与旧数据不兼容。
"""
import time

import numpy as np

from robokit.arms.base import Arm, ArmCommandRejected
from robokit.arms.piper_ik import DEFAULT_JOINT_LIMITS_DEG
from robokit.utils import log

_DEG2RAD = np.pi / 180.0
# 夹爪满行程的缺省值（mm）。SDK 原始值单位 0.001mm。实测夹爪能张到 ~84mm，即按 70 归一化时
# 读数会 >1（2026-09-11 cover 数据 74% 帧 >1，最大 1.196）。历史数据集与已训模型都是按 70
# 归一化的，所以缺省保持 70 不变；新采集用配置项 arms.<name>.gripper_full_mm 指定实测满行程，
# 该值随 config_json 一起写进 HDF5 attrs，转换/部署时可据此还原。
_GRIPPER_FULL_MM_DEFAULT = 70.0
# 兼容旧引用（clean.py 等按历史数据集的 70mm 归一化做统计）：SDK 原始值 (0.001mm)
_GRIPPER_FULL = _GRIPPER_FULL_MM_DEFAULT * 1000

# 松灵官方 URDF 的关节限位，直接复用 piper_ik 的权威表，不再抄一份。
_JOINT_LIMITS_DEG = DEFAULT_JOINT_LIMITS_DEG
_LIMIT_MARGIN_DEG = 2.0     # 恢复时挪进限位内留的余量
_AUTO_RECOVER_MAX_DEG = 15.0  # 单关节越界超过此值不自动恢复（避免意外大动作），改为报错

# 下发前的反馈新鲜度判据。四路反馈实测都是 200Hz（周期 5ms）；在"640x480@60 录像
# + H.264 编码 + 真实推理 + 持续 JointCtrl"的满负载下 max age 仍只有 9ms，所以
# 0.1s 保持不变——它拦的是反馈流真的断了，不是正常抖动。
_FEEDBACK_MAX_AGE_S = 0.1
# 判定陈旧后的有界重读窗口。断流永远等不回来，偶发调度停顿下一帧（5ms）就回来，
# 60ms 足够区分；健康路径一次都不会进这个分支，零开销。
_FEEDBACK_RECOVER_WINDOW_S = 0.06
_FEEDBACK_RECOVER_POLL_S = 0.005


class PiperArm(Arm):
    def __init__(self, name, cfg):
        super().__init__(name, cfg)
        if self.dof != 6:
            raise ValueError(f"Piper is a 6-DOF arm, got dof={self.dof}")
        self.port = cfg.get("port", "can0")
        self.gripper_full_mm = float(cfg.get("gripper_full_mm", _GRIPPER_FULL_MM_DEFAULT))
        if not (10.0 <= self.gripper_full_mm <= 200.0):
            raise ValueError(f"[{self.name}] gripper_full_mm 不合理: {self.gripper_full_mm}")
        self._gripper_full_raw = self.gripper_full_mm * 1000.0  # SDK 单位 0.001mm
        self.speed = int(cfg.get("speed", 10))  # 安全缺省：运动速度百分比
        if not 1 <= self.speed <= 100:
            raise ValueError(f"Piper speed must be in 1..100, got {self.speed}")
        self.eef_backend = str(cfg.get("eef_backend", "host_ik"))
        if self.eef_backend not in {"host_ik", "pinocchio_ik"}:
            raise ValueError(
                "Piper EEF control supports eef_backend="
                "'host_ik'|'pinocchio_ik', "
                f"got {self.eef_backend!r}"
            )
        limits = np.asarray(
            cfg.get("joint_limits_deg", _JOINT_LIMITS_DEG),
            dtype=np.float64,
        )
        if (
            limits.shape != (6, 2)
            or not np.all(np.isfinite(limits))
            or np.any(limits[:, 0] >= limits[:, 1])
        ):
            raise ValueError("Piper joint_limits_deg must be finite shape=(6,2)")
        self.joint_limits_deg = limits
        self.joint_limit_mode = str(
            cfg.get("joint_limit_mode", "reject")
        )
        if self.joint_limit_mode not in {"reject", "clip"}:
            raise ValueError(
                "Piper joint_limit_mode must be 'reject' or 'clip', "
                f"got {self.joint_limit_mode!r}"
            )
        self.host_ik_cfg = dict(cfg.get("host_ik", {}))
        self.pinocchio_ik_cfg = dict(cfg.get("pinocchio_ik", {}))
        self._eef_ik = None
        # False 时 connect() 只报告关节越界、不自动回挪（dry-run 用：保证连接过程零运动）
        self.auto_recover = bool(cfg.get("auto_recover", True))
        self.reset_on_disconnect = bool(
            cfg.get("reset_on_disconnect", False)
        )
        # connect(read_only=True) 时置位：本对象此后不得向 CAN 写入任何一帧
        self.read_only = False
        self.sdk = None
        self._motion_mode = None
        self._reset_guard = None

    def connect(self, read_only=False):
        import time as _time

        from piper_sdk import C_PiperInterface_V2

        self.read_only = bool(read_only)
        self.sdk = C_PiperInterface_V2(self.port)
        # piper_init 会发三条查询帧（最大角速度/最大加速度/固件版本），返回值本类一处都没用。
        # 只读模式下连这几帧也不发，保证"接上采集端"在总线上完全无痕。
        self.sdk.ConnectPort(piper_init=not self.read_only)
        _time.sleep(0.2)  # 等 CAN 缓存收到首批状态帧，避免下面读到默认 0 值
        if self.read_only:
            # 遥操作采集：主臂和从臂挂在同一条 CAN 上，从臂的夹爪由主臂的联动指令驱动。
            # 本进程只是旁路记录，绝不能发帧——发一帧 0x159 GripperCtrl(pos, code=0x01)
            # 就会把从臂夹爪从联动切成"按这个目标位置保持"，之后主臂再张开也带不动
            # （表现为夹爪只张开一点点）；0x471 EnableArm 同理会打到总线上的两条臂。
            try:
                self._wait_feedback(timeout=5.0)
            except BaseException:
                # Robot.connect() 也会统一回滚；这里再保证 PiperArm 被单独使用时，
                # 连接失败同样不会遗留 piper_sdk 的 CAN 接收线程。
                self.disconnect()
                raise
            log(self.name, f"piper connected on {self.port} (read-only: 不下发任何 CAN 控制帧)",
                "INFO")
            return
        if self.reset_on_disconnect:
            from robokit.arms.piper_lifecycle import ControllerResetGuard

            self._reset_guard = ControllerResetGuard(self.sdk)
            # EnableArm below is the first write.  From this point every exit,
            # including a partially failed connect, must perform one reset.
            self._reset_guard.arm()
        try:
            self._connect_writable()
        except BaseException:
            self.disconnect()
            raise

    def _connect_writable(self):
        self.sdk.EnableArm(7)
        self._wait_enabled(timeout=5.0)
        # Never inspect or recover from a cached/default joint sample.  Fresh
        # controller, joint, EEF and drive feedback is a hard prerequisite.
        self._assert_controller_normal("connect")
        self._recover_joint_limits()
        self._assert_controller_normal("connect")

        initial_joint = self.get_state()["joint"]
        self._eef_ik = self._create_eef_ik(initial_joint)
        if self.eef_backend == "pinocchio_ik":
            validation = self._eef_ik.model_validation
            log(
                self.name,
                "EEF backend=pinocchio_ik：Pinocchio URDF/SE(3) "
                "连续有界 IK → MOVE_J/JointCtrl；"
                f"model Δ={validation['position_error_mm']:.3f}mm/"
                f"{validation['rotation_error_deg']:.4f}°",
                "INFO",
            )
        else:
            log(
                self.name,
                "EEF backend=host_ik：SDK FK + SciPy 连续有界 IK "
                "→ MOVE_J/JointCtrl",
                "INFO",
            )
        log(
            self.name,
            f"piper connected on {self.port} "
            f"(eef_backend={self.eef_backend}, "
            f"reset_on_disconnect={self.reset_on_disconnect})",
            "INFO",
        )

    def _create_eef_ik(self, initial_joint):
        """Production IK factory shared by hardware and offline validation."""

        if self.eef_backend == "pinocchio_ik":
            from robokit.arms.piper_pinocchio_ik import PiperPinocchioIK

            cfg = self.pinocchio_ik_cfg
            return PiperPinocchioIK(
                initial_joint,
                joint_limits_deg=self.joint_limits_deg,
                urdf_path=cfg.get("urdf_path"),
                eef_frame=str(cfg.get("eef_frame", "link6")),
                limit_margin_deg=float(
                    cfg.get("limit_margin_deg", 0.0)
                ),
                max_step_deg=float(cfg.get("max_step_deg", 5.0)),
                continuity_weight=float(
                    cfg.get("continuity_weight", 3e-2)
                ),
                position_tolerance_mm=float(
                    cfg.get("position_tolerance_mm", 1.0)
                ),
                rotation_tolerance_deg=float(
                    cfg.get("rotation_tolerance_deg", 1.0)
                ),
                seed_limit_tolerance_deg=float(
                    cfg.get("seed_limit_tolerance_deg", 10.0)
                ),
                max_nfev=int(cfg.get("max_nfev", 200)),
                allow_best_effort=bool(
                    cfg.get("allow_best_effort", False)
                ),
                model_validation_position_mm=float(
                    cfg.get("model_validation_position_mm", 0.25)
                ),
                model_validation_rotation_deg=float(
                    cfg.get("model_validation_rotation_deg", 0.02)
                ),
            )

        from robokit.arms.piper_ik import PiperContinuousIK

        cfg = self.host_ik_cfg
        return PiperContinuousIK(
            initial_joint,
            joint_limits_deg=self.joint_limits_deg,
            limit_margin_deg=float(cfg.get("limit_margin_deg", 0.01)),
            max_step_deg=float(cfg.get("max_step_deg", 5.0)),
            continuity_weight=float(
                cfg.get("continuity_weight", 3e-2)
            ),
            position_tolerance_mm=float(
                cfg.get("position_tolerance_mm", 1.0)
            ),
            rotation_tolerance_deg=float(
                cfg.get("rotation_tolerance_deg", 1.0)
            ),
            seed_limit_tolerance_deg=float(
                cfg.get("seed_limit_tolerance_deg", 0.3)
            ),
            max_nfev=int(cfg.get("max_nfev", 200)),
        )

    def _wait_feedback(self, timeout):
        """只读模式的连接确认：不发任何指令，只等反馈帧到齐。

        没有这一步，CAN 没起来 / 机械臂没上电时 SDK 缓存全是默认 0，get_state() 会安静地
        返回一整段零状态，采出来的 episode 直到清洗阶段才发现是废的。"""
        start = time.time()
        while time.time() - start < timeout:
            if self.sdk.GetArmJointMsgs().time_stamp and self.sdk.GetArmGripperMsgs().time_stamp:
                return
            time.sleep(0.1)
        raise RuntimeError(
            f"[{self.name}] {timeout}s 内没收到 {self.port} 的关节/夹爪反馈帧："
            "检查 CAN 是否 up、机械臂是否上电")

    def _assert_writable(self):
        if self.read_only:
            raise RuntimeError(
                f"[{self.name}] 处于只读模式（遥操作采集），不允许下发运动指令")

    def _read_feedback_wrappers(self):
        """一次取齐四路反馈及其年龄。年龄用同一个 now，四路才可比。"""
        status_wrapper = self.sdk.GetArmStatus()
        joint_wrapper = self.sdk.GetArmJointMsgs()
        eef_wrapper = self.sdk.GetArmEndPoseMsgs()
        low_wrapper = self.sdk.GetArmLowSpdInfoMsgs()
        now = time.time()
        timestamps = {
            "status": float(status_wrapper.time_stamp),
            "joint": float(joint_wrapper.time_stamp),
            "eef": float(eef_wrapper.time_stamp),
            "low_speed": float(low_wrapper.time_stamp),
        }
        stale = {
            name: now - stamp
            for name, stamp in timestamps.items()
            if stamp <= 0
            or now - stamp > _FEEDBACK_MAX_AGE_S
            or now - stamp < -_FEEDBACK_MAX_AGE_S
        }
        return status_wrapper, joint_wrapper, eef_wrapper, low_wrapper, stale

    def _assert_controller_normal(self, where):
        """Require fresh healthy feedback before every control command."""
        (
            status_wrapper,
            joint_wrapper,
            eef_wrapper,
            low_wrapper,
            stale,
        ) = self._read_feedback_wrappers()

        # 四路都是 200Hz（实测周期 5ms，各种负载下 max age 9ms），0.1s 已有十几倍
        # 余量。真正会读到陈旧的只有两种情况：反馈流断了（永远不恢复），或者主机
        # 侧偶发调度停顿（下一帧就恢复）。2026-07-28 真机遇到过一次后者：joint
        # 100.5ms / low_speed 114.7ms，而同一时刻独立进程读 can0 完全正常，说明
        # 帧一直在到，是本进程的 SDK 接收线程被挤掉了；同配置连跑三次未复现。
        # 因此这里给一个有界重读窗口把两者分开：能恢复的放行，恢复不了的照旧中止。
        if stale:
            first_stale = stale
            deadline = time.time() + _FEEDBACK_RECOVER_WINDOW_S
            while time.time() < deadline:
                time.sleep(_FEEDBACK_RECOVER_POLL_S)
                (
                    status_wrapper,
                    joint_wrapper,
                    eef_wrapper,
                    low_wrapper,
                    stale,
                ) = self._read_feedback_wrappers()
                if not stale:
                    break
            if stale:
                raise ArmCommandRejected(
                    "piper_feedback_stale",
                    f"[{self.name}] {where} 前反馈陈旧/时钟异常：{stale}；"
                    f"{_FEEDBACK_RECOVER_WINDOW_S:.2f}s 内未恢复；未发送运动帧",
                )
            # 恢复了也要留痕，否则反复出现的停顿会被静默吃掉。
            log(
                self.name,
                f"{where} 前反馈短暂陈旧后已恢复："
                + ", ".join(f"{k}={v:.3f}s" for k, v in sorted(first_stale.items())),
                "WARNING",
            )

        status = status_wrapper.arm_status
        code = int(status.arm_status)
        err = int(status.err_code)
        if code != 0 or err != 0:
            raise ArmCommandRejected(
                "piper_status",
                f"[{self.name}] {where} 前固件状态异常："
                f"arm_status={status.arm_status}, err_code={err}, "
                f"ctrl={status.ctrl_mode}, mode={status.mode_feed}。"
                "未发送运动帧",
            )
        if int(status.teach_status) != 0:
            raise ArmCommandRejected(
                "piper_teach_mode",
                f"[{self.name}] {where} 前 teach_status="
                f"{status.teach_status}；未发送运动帧",
            )
        enabled = []
        protections = []
        protection_names = (
            "voltage_too_low",
            "motor_overheating",
            "driver_overcurrent",
            "driver_overheating",
            "collision_status",
            "driver_error_status",
            "stall_status",
        )
        for index in range(1, 7):
            foc = getattr(low_wrapper, f"motor_{index}").foc_status
            enabled.append(bool(foc.driver_enable_status))
            active = [
                name for name in protection_names if bool(getattr(foc, name))
            ]
            if active:
                protections.append(f"j{index}:{','.join(active)}")
        if not all(enabled):
            raise ArmCommandRejected(
                "piper_driver_disabled",
                f"[{self.name}] {where} 前驱动使能={enabled}；"
                "未发送运动帧",
            )
        if protections:
            raise ArmCommandRejected(
                "piper_driver_protection",
                f"[{self.name}] {where} 前驱动保护="
                f"{'; '.join(protections)}；未发送运动帧",
            )
        return status

    def assert_healthy(self, where="runtime"):
        """Public read-only health hook used while waiting for arrival."""
        self._assert_writable()
        return self._assert_controller_normal(where)

    def _ensure_motion_mode(self, move_mode):
        """Send MotionCtrl_2 only when changing modes, not before every target."""
        if self._motion_mode == move_mode:
            status = self.sdk.GetArmStatus().arm_status
            if (
                int(status.ctrl_mode) != 0x01
                or int(status.mode_feed) != move_mode
            ):
                raise ArmCommandRejected(
                    "piper_mode_changed",
                    f"[{self.name}] 控制模式被外部改变："
                    f"ctrl={status.ctrl_mode}, mode={status.mode_feed}；"
                    "未发送目标",
                )
            return
        self.sdk.MotionCtrl_2(0x01, move_mode, self.speed, 0x00)
        self._motion_mode = move_mode

    def _recover_joint_limits(self):
        """恢复反馈关节本身的轻微越界；越界过大时要求人工处理。"""
        joints = self.get_state()["joint"]
        limits = np.radians(self.joint_limits_deg)
        if np.all((joints >= limits[:, 0]) & (joints <= limits[:, 1])):
            return  # 都在软限位内，无需恢复
        if self.eef_backend == "pinocchio_ik":
            # Pinocchio's explicit 10° rule is about physical feedback relative
            # to the true mechanical limits.  Do not hide a >10° hard failure
            # behind the legacy connection-time auto-recovery move, and do not
            # move for a <=10° encoder/servo overrun: reset_seed projects it.
            projected = np.clip(joints, limits[:, 0], limits[:, 1])
            overrun_deg = np.degrees(np.abs(joints - projected))
            tolerance_deg = float(
                self.pinocchio_ik_cfg.get(
                    "seed_limit_tolerance_deg", 10.0
                )
            )
            if (
                not np.isfinite(tolerance_deg)
                or tolerance_deg < 0
            ):
                raise ValueError(
                    "pinocchio_ik.seed_limit_tolerance_deg "
                    "必须是非负有限值"
                )
            if float(overrun_deg.max()) > tolerance_deg + 1e-12:
                raise RuntimeError(
                    f"[{self.name}] 物理关节反馈超出机械限位 "
                    f"{overrun_deg.max():.3f}° > "
                    f"{tolerance_deg:.3f}°；拒绝自动恢复/求解"
                )
            log(
                self.name,
                "Pinocchio seed 将投影到机械限位；反馈逐轴超限="
                f"{np.round(overrun_deg, 3).tolist()}°，不发送恢复帧",
                "WARNING",
            )
            return
        lo = limits[:, 0] + np.radians(_LIMIT_MARGIN_DEG)
        hi = limits[:, 1] - np.radians(_LIMIT_MARGIN_DEG)
        clamped = np.clip(joints, lo, hi)
        over = np.degrees(np.abs(clamped - joints))
        bad = [f"j{i+1}={np.degrees(joints[i]):.1f}°" for i in range(6)
               if joints[i] < limits[i, 0] or joints[i] > limits[i, 1]]
        if not self.auto_recover:
            # dry-run / 只读模式：报告越界但绝不产生运动，由人工决定是否处理
            log(self.name, f"关节越界 {bad}，auto_recover=False 不自动回挪；"
                           "控制器将拒绝越界运动指令", "WARNING")
            return
        if over.max() > _AUTO_RECOVER_MAX_DEG:
            raise RuntimeError(
                f"[{self.name}] 关节越界过大({over.max():.1f}°>{_AUTO_RECOVER_MAX_DEG}°)，"
                f"不自动恢复。越界: {bad}。请手动小幅移回工作区后重试")
        log(self.name, f"关节越界 {bad}，低速挪回软限位内(最大 {over.max():.1f}°)...", "WARNING")
        saved_speed = self.speed
        self.speed = min(self.speed, 15)
        try:
            for _ in range(8):
                # connect 内部显式恢复需发送合法 JointCtrl，不能调用会先经过
                # 控制器状态闸的公开 move_joint。
                self._send_joint(clamped, None, sync_eef_ik=False)
                time.sleep(0.5)
                now = self.get_state()["joint"]
                if np.all((now >= limits[:, 0]) & (now <= limits[:, 1])):
                    log(self.name, "反馈关节已回到配置限位内", "INFO")
                    return
        finally:
            self.speed = saved_speed
        raise RuntimeError(f"[{self.name}] 恢复后关节仍越界，请手动检查机械臂")

    def _wait_enabled(self, timeout):
        start = time.time()
        while time.time() - start < timeout:
            low_spd = self.sdk.GetArmLowSpdInfoMsgs()
            motors = [getattr(low_spd, f"motor_{i}") for i in range(1, 7)]
            if all(m.foc_status.driver_enable_status for m in motors):
                return
            self.sdk.EnableArm(7)
            # 机械臂链路使能与夹爪控制解耦。连接阶段绝不发送 GripperCtrl；
            # 第一条夹爪帧只能来自显式 policy/调用者动作。
            time.sleep(0.5)
        raise RuntimeError(f"[{self.name}] failed to enable piper motors within {timeout}s")

    def get_state(self):
        joint_msg = self.sdk.GetArmJointMsgs().joint_state
        joint = np.array([getattr(joint_msg, f"joint_{i}") for i in range(1, 7)],
                         dtype=np.float64) * 0.001 * _DEG2RAD

        eef = self.sdk.GetArmEndPoseMsgs().end_pose
        xyz = np.array([eef.X_axis, eef.Y_axis, eef.Z_axis], dtype=np.float64) * 1e-6      # 0.001mm → m
        rpy = np.array([eef.RX_axis, eef.RY_axis, eef.RZ_axis], dtype=np.float64) * 0.001 * _DEG2RAD

        gripper = self.sdk.GetArmGripperMsgs().gripper_state.grippers_angle / self._gripper_full_raw
        return self._stamped(joint, np.concatenate([xyz, rpy]), gripper)

    def _validated_joint(self, joint, *, clip_limits=False):
        joint = np.asarray(joint, dtype=np.float64)
        if joint.shape != (6,) or not np.all(np.isfinite(joint)):
            raise ArmCommandRejected(
                "joint_target",
                f"[{self.name}] 关节目标必须是有限的 shape=(6,)，"
                f"实际为 {joint.shape}",
            )
        limits = np.radians(self.joint_limits_deg)
        if np.any(joint < limits[:, 0]) or np.any(joint > limits[:, 1]):
            if clip_limits:
                return np.clip(joint, limits[:, 0], limits[:, 1])
            raise ArmCommandRejected(
                "joint_limit",
                f"[{self.name}] 关节目标超出配置限位："
                f"{np.round(np.degrees(joint), 3).tolist()}°",
            )
        return joint

    def _send_joint(
        self,
        joint,
        gripper,
        sync_eef_ik,
        *,
        clip_limits=False,
    ):
        raw = self._quantized_joint(
            joint,
            clip_limits=clip_limits,
        )
        quantized = np.radians(raw.astype(np.float64) / 1000.0)
        if sync_eef_ik and self._eef_ik is not None:
            try:
                self._eef_ik.reset_seed(quantized)
            except ValueError as exc:
                raise ArmCommandRejected(
                    f"{self.eef_backend}_seed",
                    f"[{self.name}] 关节目标不能作为 "
                    f"{self.eef_backend} seed："
                    f"{exc}；未发送运动帧",
                ) from exc
        self._ensure_motion_mode(0x01)
        self.sdk.JointCtrl(*raw.tolist())
        if gripper is not None:
            self._move_gripper(gripper)
        return raw

    def _quantized_joint(self, joint, *, clip_limits=False):
        """Pure joint preprocessing shared by live send and zero-CAN preview."""
        joint = self._validated_joint(
            joint,
            clip_limits=clip_limits,
        )
        raw = np.rint(np.degrees(joint) * 1000.0).astype(np.int64)
        quantized = np.radians(raw.astype(np.float64) / 1000.0)
        # Validate the exact quantized command too.  An unusual non-millidegree
        # mechanical limit must fail closed rather than round across it.
        self._validated_joint(quantized)
        return raw

    def _joint_command_metadata(self, joint, raw):
        requested = np.asarray(joint, dtype=np.float64)
        limits = np.radians(self.joint_limits_deg)
        target_deg = raw.astype(np.float64) / 1000.0
        requested_deg = np.degrees(requested)
        clipped_rad = np.clip(
            requested,
            limits[:, 0],
            limits[:, 1],
        )
        clip_delta_deg = np.degrees(clipped_rad - requested)
        clipped_axes = np.where(np.abs(clip_delta_deg) > 1e-12)[0]
        if clipped_axes.size:
            detail = ", ".join(
                f"j{i+1} {requested_deg[i]:.3f}°→"
                f"{np.degrees(clipped_rad[i]):.3f}°"
                for i in clipped_axes
            )
            log(
                self.name,
                f"joint target clipped to configured limits: {detail}",
                "WARNING",
            )
        return {
            "backend": "joint",
            "joint_limit_mode": self.joint_limit_mode,
            "joint_limit_clipped": bool(clipped_axes.size),
            "joint_clipped_axes": [
                f"j{i+1}" for i in clipped_axes.tolist()
            ],
            "joint_requested_deg": requested_deg.tolist(),
            "joint_limit_clip_delta_deg": clip_delta_deg.tolist(),
            "joint_target_deg": target_deg.tolist(),
        }

    def preview_joint(self, joint):
        """Return the exact direct-joint command metadata without any CAN write."""
        requested = np.asarray(joint, dtype=np.float64)
        raw = self._quantized_joint(
            requested,
            clip_limits=self.joint_limit_mode == "clip",
        )
        return self._joint_command_metadata(requested, raw)

    def move_joint(self, joint, gripper=None):
        self._assert_writable()
        self._assert_controller_normal("move_joint")
        requested = np.asarray(joint, dtype=np.float64)
        # Only finite 6-D direct-joint targets may opt into mechanical-limit
        # projection; EEF/IK commands continue to use strict rejection.
        raw = self._send_joint(
            requested,
            gripper,
            sync_eef_ik=True,
            clip_limits=self.joint_limit_mode == "clip",
        )
        return self._joint_command_metadata(requested, raw)

    def move_eef(self, pose, gripper=None):
        self._assert_writable()
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise ArmCommandRejected(
                "eef_target",
                f"[{self.name}] EEF 目标必须是有限的 shape=(6,)，"
                f"实际为 {pose.shape}",
            )
        self._assert_controller_normal("move_eef")
        if self._eef_ik is None:
            raise ArmCommandRejected(
                f"{self.eef_backend}_state",
                f"[{self.name}] {self.eef_backend} 尚未初始化；"
                "未发送运动帧",
            )

        feedback_joint = self.get_state()["joint"]
        seed_joint = self._eef_ik.seed_joint
        seed_tracking_deg = np.degrees(
            np.abs(feedback_joint - seed_joint)
        )
        # Legacy host_ik keeps its old commanded-target lag gate.  Pinocchio IK
        # always re-seeds from fresh feedback, so ordinary servo lag is only
        # recorded and cannot abort a no-guard policy run.
        if self.eef_backend == "host_ik":
            max_tracking_deg = float(
                self.host_ik_cfg.get("max_seed_tracking_deg", 10.0)
            )
            if (
                not np.isfinite(max_tracking_deg)
                or max_tracking_deg <= 0
            ):
                raise ValueError(
                    "host_ik.max_seed_tracking_deg 必须是正有限值"
                )
            if seed_tracking_deg.max() > max_tracking_deg:
                raise ArmCommandRejected(
                    "host_ik_tracking",
                    f"[{self.name}] 反馈关节落后上一条 IK 目标 "
                    f"{seed_tracking_deg.max():.2f}° > "
                    f"{max_tracking_deg:.2f}°；未发送新目标",
                )

        try:
            # Every solve starts from fresh physical feedback.  The previous
            # commanded target remains useful only as a lag detector above;
            # it must never let candidate targets run another 5° ahead of the
            # actual arm near a singularity.
            seed_limit_overrun_deg = self._eef_ik.reset_seed(
                feedback_joint
            )
            result = self._eef_ik.solve(pose, commit=False)
        except (
            TypeError,
            ValueError,
            FloatingPointError,
            RuntimeError,
        ) as exc:
            raise ArmCommandRejected(
                f"{self.eef_backend}_input",
                f"[{self.name}] {self.eef_backend} 输入/数值异常："
                f"{exc}；"
                "未发送运动帧",
            ) from exc
        if not result.success:
            raise ArmCommandRejected(
                self.eef_backend,
                f"[{self.name}] {self.eef_backend} 无可下发候选："
                f"reason={result.reason}, "
                f"position_error={result.position_error_mm:.3f}mm, "
                f"rotation_error={result.rotation_error_deg:.3f}°, "
                f"joint_step={result.joint_step_deg.max():.3f}°；"
                "未发送运动帧",
            )

        # Compute all metadata needed by arrival tracking before the first CAN
        # motion frame.  A metadata/FK failure must never occur after motion was
        # already sent.
        realized_pose = self._eef_ik.forward_pose(result.joint)
        self._validated_joint(result.joint)
        # Result raw values are the exact JointCtrl payload.  Reconstruct and
        # validate once more immediately before the first motion frame.
        quantized = np.radians(
            result.joint_millideg.astype(np.float64) / 1000.0
        )
        self._validated_joint(quantized)
        self._ensure_motion_mode(0x01)
        self.sdk.JointCtrl(*result.joint_millideg.tolist())
        self._eef_ik.reset_seed(result.joint)
        if gripper is not None:
            self._move_gripper(gripper)
        command = {
            "backend": f"{self.eef_backend}_move_j",
            "joint_target_deg": (
                result.joint_millideg.astype(np.float64) / 1000.0
            ).tolist(),
            "eef_target_realized": realized_pose.tolist(),
            "position_error_mm": result.position_error_mm,
            "rotation_error_deg": result.rotation_error_deg,
            "joint_step_deg": result.joint_step_deg.tolist(),
            "limit_margin_deg": result.limit_margin_deg.tolist(),
            "seed_tracking_deg": seed_tracking_deg.tolist(),
            "seed_limit_overrun_deg": seed_limit_overrun_deg.tolist(),
        }
        if self.eef_backend == "pinocchio_ik":
            command.update(
                converged=result.converged,
                saturated=result.saturated,
                iterations=result.iterations,
            )
        else:
            command["nfev"] = result.nfev
        return command

    def _move_gripper(self, gripper):
        self._assert_writable()
        raw = int(np.clip(gripper, 0.0, 1.0) * self._gripper_full_raw)
        self.sdk.GripperCtrl(raw, 1000, 0x01, 0)

    def set_cleanup_trace(self, trace):
        if self._reset_guard is not None:
            self._reset_guard.trace = trace

    def disconnect(self):
        try:
            if self._reset_guard is not None:
                record = self._reset_guard.close()
                if record is not None and record.get("verified"):
                    log(
                        self.name,
                        "controller reset verified: NORMAL, six drives disabled",
                        "INFO",
                    )
        finally:
            if self.sdk is not None:
                self.sdk.DisconnectPort()
                self.sdk = None
            self._eef_ik = None
            self._motion_mode = None
            self._reset_guard = None
