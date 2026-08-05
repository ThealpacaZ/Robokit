"""真机动作安全闸：在指令下发到机械臂之前做形状/数值/幅度/工作区/跟踪误差检查。

模型一次异常输出原本会原样下发。ActionGuard 在 CAN 下发前拦截构造出的异常动作，
形状、非有限值、步幅、工作区、关节范围和跟踪误差规则均可独立验证。

默认阈值按 Piper + 30 Hz 控制给的保守值，可在 YAML 的 deploy.safety 里覆盖：

    deploy:
      safety:
        enabled: true
        max_delta_xyz: 0.03      # 单步局部位移上限 (m)
        max_delta_rpy: 0.35      # 单步局部旋转上限 (rad)
        max_target_step: 0.05    # 相邻 commanded target 的跳变上限 (m)
        max_tracking_err: 0.08   # 反馈相对上一条 target 的滞后上限 (m)，超出判定未跟随
        gripper_max_rate: 1.0    # 单步夹爪变化上限
        workspace: {x: [-0.10, 0.70], y: [-0.50, 0.50], z: [0.00, 0.60]}
        joint_limits_deg: [[-150,150],[0,180],[-170,0],[-100,100],[-70,70],[-170,170]]

阈值语义都是「超过就抛 GuardViolation」，由 ChunkExecutor 转成中止。宁可停下也不要
让一条异常指令进 CAN 总线。
"""
import numpy as np
from scipy.spatial.transform import Rotation

DEFAULTS = {
    "enabled": True,
    "max_delta_xyz": 0.03,
    "max_delta_rpy": 0.35,
    "max_target_step": 0.05,
    "max_tracking_err": 0.08,
    "gripper_max_rate": 1.0,
    "workspace": {"x": [-0.10, 0.70], "y": [-0.50, 0.50], "z": [0.00, 0.60]},
    "joint_limits_deg": None,
}


class GuardViolation(RuntimeError):
    def __init__(self, kind, message):
        super().__init__(message)
        self.kind = kind


class ActionGuard:
    def __init__(self, **cfg):
        unknown = set(cfg) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"unknown safety option(s): {sorted(unknown)}")
        merged = dict(DEFAULTS)
        merged.update({k: v for k, v in cfg.items() if v is not None})
        self.cfg = merged
        ws = merged["workspace"] or {}
        self.ws = {axis: (None if ws.get(axis) is None else tuple(ws[axis]))
                   for axis in ("x", "y", "z")}
        jl = merged["joint_limits_deg"]
        self.joint_limits = None if jl is None else np.radians(np.asarray(jl, dtype=np.float64))
        self._prev_target = {}

    @classmethod
    def from_config(cls, deploy_cfg):
        """按 deploy.safety 构造；enabled=false 时返回 None（= 不设防，仅用于对照实验）。"""
        cfg = dict(deploy_cfg.get("safety") or {})
        if not cfg.pop("enabled", True):
            return None
        return cls(**cfg)

    # ── 规则 ────────────────────────────────────────────────────────────────
    def check_eef(self, delta, target, feedback=None, gripper=None, prev_gripper=None,
                  tracking_err=None, arm="arm"):
        delta = np.asarray(delta, dtype=np.float64)
        target = np.asarray(target, dtype=np.float64)
        if delta.shape != (6,) or target.shape != (6,):
            raise GuardViolation("shape", f"[{arm}] expected 6-D delta/target, "
                                          f"got {delta.shape}/{target.shape}")
        if not (np.all(np.isfinite(delta)) and np.all(np.isfinite(target))):
            raise GuardViolation("nonfinite", f"[{arm}] non-finite delta/target: {delta}")

        dxyz = float(np.linalg.norm(delta[:3]))
        if dxyz > self.cfg["max_delta_xyz"]:
            raise GuardViolation("delta_xyz", f"[{arm}] step |dxyz|={dxyz*1000:.1f}mm > "
                                              f"{self.cfg['max_delta_xyz']*1000:.1f}mm")
        # 三个欧拉分量分别小于阈值，不代表合成后的物理旋转也小于阈值。
        # 安全闸比较 SO(3) 上的最短旋转角，避免例如三轴各 19.5° 实际合成为
        # 31.6° 却穿过 20° 分量闸。
        rotation_step = float(
            Rotation.from_euler("xyz", delta[3:]).magnitude()
        )
        if rotation_step > self.cfg["max_delta_rpy"]:
            raise GuardViolation(
                "delta_rpy",
                f"[{arm}] step rotation={np.degrees(rotation_step):.1f}° > "
                f"{np.degrees(self.cfg['max_delta_rpy']):.1f}°",
            )

        for i, axis in enumerate("xyz"):
            bounds = self.ws.get(axis)
            if bounds is not None and not (bounds[0] <= target[i] <= bounds[1]):
                raise GuardViolation("workspace", f"[{arm}] target {axis}={target[i]:.3f}m "
                                                  f"outside {bounds}")

        prev = self._prev_target.get(arm)
        if prev is not None:
            jump = float(np.linalg.norm(target[:3] - prev[:3]))
            if jump > self.cfg["max_target_step"]:
                raise GuardViolation("target_jump", f"[{arm}] target jump {jump*1000:.1f}mm > "
                                                    f"{self.cfg['max_target_step']*1000:.1f}mm")
        if tracking_err is not None and tracking_err > self.cfg["max_tracking_err"]:
            raise GuardViolation("tracking", f"[{arm}] feedback lags last target by "
                                             f"{tracking_err*1000:.1f}mm > "
                                             f"{self.cfg['max_tracking_err']*1000:.1f}mm "
                                             f"(未跟随: 碰撞/限位/失能?)")
        self._check_gripper(gripper, prev_gripper, arm)
        self._prev_target[arm] = target

    def check_joint(self, joint, feedback=None, gripper=None, prev_gripper=None, arm="arm"):
        joint = np.asarray(joint, dtype=np.float64)
        if not np.all(np.isfinite(joint)):
            raise GuardViolation("nonfinite", f"[{arm}] non-finite joint target: {joint}")
        if self.joint_limits is not None:
            if joint.shape[0] != self.joint_limits.shape[0]:
                raise GuardViolation("shape", f"[{arm}] joint dim {joint.shape[0]} != "
                                              f"limits {self.joint_limits.shape[0]}")
            lo, hi = self.joint_limits[:, 0], self.joint_limits[:, 1]
            bad = np.where((joint < lo) | (joint > hi))[0]
            if bad.size:
                detail = ", ".join(f"j{i+1}={np.degrees(joint[i]):.1f}°" for i in bad)
                raise GuardViolation("joint_limit", f"[{arm}] joint target out of limits: {detail}")
        self._check_gripper(gripper, prev_gripper, arm)

    def _check_gripper(self, gripper, prev_gripper, arm):
        if gripper is None:
            return
        if not np.isfinite(gripper):
            raise GuardViolation("nonfinite", f"[{arm}] non-finite gripper: {gripper}")
        if not (0.0 <= gripper <= 1.0):
            raise GuardViolation("gripper_range", f"[{arm}] gripper {gripper:.3f} outside [0,1]")
        if prev_gripper is not None:
            rate = abs(gripper - prev_gripper)
            if rate > self.cfg["gripper_max_rate"]:
                raise GuardViolation("gripper_rate", f"[{arm}] gripper step {rate:.2f} > "
                                                     f"{self.cfg['gripper_max_rate']:.2f}")

    def reset(self):
        self._prev_target.clear()

    def describe(self):
        ws = {a: list(b) if b else None for a, b in self.ws.items()}
        return {k: v for k, v in self.cfg.items() if k != "workspace"} | {"workspace": ws}
