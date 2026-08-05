"""Piper 的纯计算、连续有界主机 IK。

这个模块只使用 :class:`piper_sdk.C_PiperForwardKinematics`，不会创建
``C_PiperInterface``、不会打开 CAN，也不会发送控制帧。

输入/输出单位与 Robokit 一致：

* 关节角：弧度
* EEF 位置：米
* EEF 姿态：固定轴 xyz 欧拉角，弧度（SciPy 的小写 ``"xyz"``）

求解目标不仅是达到 EEF 位姿，还要保持上一条已接受关节目标所在的 IK 分支。
因此每次求解都以最后接受的关节角为 seed，并施加相邻帧硬步长和轻量连续性正则。
失败的结果不会更新 seed，调用方应保持上一条已接受的关节目标。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from piper_sdk import C_PiperForwardKinematics
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


# 权威表在 piper_limits（那边只依赖 numpy，纯脚本取表时不必付 scipy 的启动开销）。
# 这里 re-export，历史的 from robokit.arms.piper_ik import DEFAULT_JOINT_LIMITS_DEG
# 仍然可用。
from robokit.arms.piper_limits import DEFAULT_JOINT_LIMITS_DEG

_JOINT_CTRL_RESOLUTION_DEG = 0.001
_DEFAULT_SEED_LIMIT_TOLERANCE_DEG = 0.3


def _six_finite(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (6,):
        raise ValueError(f"{name} 必须是 shape=(6,)，实际为 {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 含 NaN/Inf")
    return array


@dataclass(frozen=True)
class PiperIKResult:
    """一次 IK 求解的纯计算结果。

    ``joint`` 已按 ``JointCtrl`` 的 0.001° 分辨率四舍五入。仅当 ``success``
    为真时才应发送它；失败时使用 ``hold_joint`` 保持上一条已接受目标。
    """

    success: bool
    joint: np.ndarray
    joint_millideg: np.ndarray
    hold_joint: np.ndarray
    position_error_mm: float
    rotation_error_deg: float
    joint_step_deg: np.ndarray
    limit_margin_deg: np.ndarray
    nfev: int
    optimizer_success: bool
    optimizer_status: int
    reason: str
    message: str


class PiperContinuousIK:
    """以上一条关节目标为 seed 的连续有界 Piper IK。

    优化变量 ``q`` 使用弧度，最小化以下残差的平方和：

    ``r_pos = FK_pos(q)[mm] - target_pos[mm]``

    ``r_rot = degrees(Log(R_fk(q).T @ R_target))``

    ``r_cont = continuity_weight * degrees(q - q_seed)``

    同时施加硬边界：

    ``max(q_min + margin, q_seed - max_step) <= q``

    ``q <= min(q_max - margin, q_seed + max_step)``

    默认把 1 mm 与 1° 作为相同数值尺度；连续性项只用于在腕部奇异点的零空间中
    选择靠近 seed 的解，硬步长负责阻止分支跳变。
    """

    def __init__(
        self,
        initial_joint: Sequence[float],
        *,
        joint_limits_deg: Sequence[Sequence[float]] = DEFAULT_JOINT_LIMITS_DEG,
        limit_margin_deg: float = 0.0,
        max_step_deg: float = 5.0,
        continuity_weight: float = 3e-2,
        position_tolerance_mm: float = 1.0,
        rotation_tolerance_deg: float = 1.0,
        seed_limit_tolerance_deg: float = _DEFAULT_SEED_LIMIT_TOLERANCE_DEG,
        max_nfev: int = 200,
        dh_is_offset: int = 1,
    ) -> None:
        limits = np.asarray(joint_limits_deg, dtype=np.float64)
        if limits.shape != (6, 2) or not np.all(np.isfinite(limits)):
            raise ValueError("joint_limits_deg 必须是有限的 shape=(6, 2)")
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("每个关节的下限必须小于上限")
        if not np.isfinite(limit_margin_deg) or limit_margin_deg < 0:
            raise ValueError("limit_margin_deg 必须是非负有限值")
        if np.any(
            limits[:, 0] + limit_margin_deg
            >= limits[:, 1] - limit_margin_deg
        ):
            raise ValueError("limit_margin_deg 使关节上下限相交")
        if not np.isfinite(max_step_deg) or max_step_deg <= 0:
            raise ValueError("max_step_deg 必须是正有限值")
        if not np.isfinite(continuity_weight) or continuity_weight < 0:
            raise ValueError("continuity_weight 必须是非负有限值")
        if not np.isfinite(position_tolerance_mm) or position_tolerance_mm <= 0:
            raise ValueError("position_tolerance_mm 必须是正有限值")
        if not np.isfinite(rotation_tolerance_deg) or rotation_tolerance_deg <= 0:
            raise ValueError("rotation_tolerance_deg 必须是正有限值")
        if (
            not np.isfinite(seed_limit_tolerance_deg)
            or seed_limit_tolerance_deg < 0
        ):
            raise ValueError("seed_limit_tolerance_deg 必须是非负有限值")
        if max_nfev <= 0:
            raise ValueError("max_nfev 必须为正整数")
        if dh_is_offset not in (0, 1):
            raise ValueError("dh_is_offset 只能是 0 或 1")

        self._lower = np.radians(limits[:, 0] + limit_margin_deg)
        self._upper = np.radians(limits[:, 1] - limit_margin_deg)
        self.max_step_deg = float(max_step_deg)
        self.continuity_weight = float(continuity_weight)
        self.position_tolerance_mm = float(position_tolerance_mm)
        self.rotation_tolerance_deg = float(rotation_tolerance_deg)
        self.seed_limit_tolerance_deg = float(seed_limit_tolerance_deg)
        self.max_nfev = int(max_nfev)
        self._fk = C_PiperForwardKinematics(dh_is_offset=dh_is_offset)
        self._seed = np.zeros(6, dtype=np.float64)
        self.reset_seed(initial_joint)

    @property
    def seed_joint(self) -> np.ndarray:
        """最后一条已接受的关节目标（副本，弧度）。"""

        return self._seed.copy()

    def reset_seed(self, joint: Sequence[float]) -> np.ndarray:
        """用当前关节反馈重新同步 IK 分支并返回逐轴超限量（度）。

        编码器反馈在伺服贴住机械限位时可能略过界。只要逐轴超限量不大于配置的
        ``seed_limit_tolerance_deg``，就把 seed 投影到最近的合法限位继续做有界 IK；
        实际 IK 输出仍不可能越过 ``_lower/_upper``。超过容差才拒绝，避免把明显失力
        下垂或错误反馈静默当成正常状态。
        """

        seed = _six_finite(joint, "joint")
        projected = np.clip(seed, self._lower, self._upper)
        overrun_deg = np.degrees(np.abs(seed - projected))
        max_overrun_deg = float(overrun_deg.max())
        if max_overrun_deg > self.seed_limit_tolerance_deg + 1e-12:
            raise ValueError(
                "seed 关节角超出配置限位 "
                f"{max_overrun_deg:.3f}° > 允许的 "
                f"{self.seed_limit_tolerance_deg:.3f}°；"
                "主机 IK 不能代替机械臂复位/大幅越界恢复"
            )
        self._seed = projected.copy()
        return overrun_deg

    def _forward_matrix(
        self, joint: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        xyzrpy = np.asarray(
            self._fk.CalFK(joint.tolist())[-1], dtype=np.float64
        )
        xyz_mm = xyzrpy[:3]
        rotation = Rotation.from_euler(
            "xyz", xyzrpy[3:], degrees=True
        ).as_matrix()
        return xyz_mm, rotation

    @staticmethod
    def _target(pose: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
        target = _six_finite(pose, "pose")
        xyz_mm = target[:3] * 1000.0
        rotation = Rotation.from_euler("xyz", target[3:]).as_matrix()
        return xyz_mm, rotation

    def _pose_residual(
        self,
        joint: np.ndarray,
        xyz_target_mm: np.ndarray,
        rotation_target: np.ndarray,
        seed: np.ndarray | None = None,
        include_continuity: bool = False,
    ) -> np.ndarray:
        xyz_mm, rotation = self._forward_matrix(joint)
        rotation_error_deg = np.degrees(
            Rotation.from_matrix(
                rotation.T @ rotation_target
            ).as_rotvec()
        )
        residual = np.concatenate(
            (xyz_mm - xyz_target_mm, rotation_error_deg)
        )
        if include_continuity:
            if seed is None:
                raise ValueError("连续性残差需要 seed")
            residual = np.concatenate(
                (
                    residual,
                    self.continuity_weight
                    * np.degrees(joint - seed),
                )
            )
        return residual

    @staticmethod
    def _quantize_joint(
        joint: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        # 先生成并保留整数 raw，避免量化后的弧度浮点数再次转换时差 1 millideg。
        raw = np.rint(np.degrees(joint) * 1000.0).astype(np.int64)
        quantized = np.radians(raw.astype(np.float64) / 1000.0)
        return quantized, raw

    def evaluate(
        self, joint: Sequence[float], pose: Sequence[float]
    ) -> tuple[float, float]:
        """计算给定关节角相对目标位姿的位置/旋转矩阵误差。"""

        candidate = _six_finite(joint, "joint")
        xyz_target_mm, rotation_target = self._target(pose)
        residual = self._pose_residual(
            candidate, xyz_target_mm, rotation_target
        )
        return (
            float(np.linalg.norm(residual[:3])),
            float(np.linalg.norm(residual[3:])),
        )

    def forward_pose(self, joint: Sequence[float]) -> np.ndarray:
        """Return the EEF pose physically represented by a joint target.

        Position is returned in meters and orientation as fixed-axis xyz Euler
        radians, matching the rest of robokit.  Arrival tracking uses this pose
        because host IK sends a quantized joint target, not the unattainable
        ideal EEF target passed to the optimizer.
        """
        candidate = _six_finite(joint, "joint")
        xyz_mm, rotation = self._forward_matrix(candidate)
        return np.concatenate(
            (
                xyz_mm / 1000.0,
                Rotation.from_matrix(rotation).as_euler("xyz"),
            )
        )

    def solve(
        self, pose: Sequence[float], *, commit: bool = True
    ) -> PiperIKResult:
        """求一个目标位姿；成功时默认提交为下一帧 seed。

        失败不会修改内部 seed。调用方应检查 ``result.success``，失败时保持
        ``result.hold_joint``，而不是发送未验证的 ``result.joint``。
        """

        xyz_target_mm, rotation_target = self._target(pose)
        seed = self._seed.copy()
        step_rad = np.radians(self.max_step_deg)
        lower = np.maximum(self._lower, seed - step_rad)
        upper = np.minimum(self._upper, seed + step_rad)
        # seed 由 reset_seed/上一次成功结果保证在全局限位内，因此局部区间非空。
        x0 = np.clip(seed, lower + 1e-12, upper - 1e-12)

        optimizer = least_squares(
            self._pose_residual,
            x0,
            args=(xyz_target_mm, rotation_target, seed, True),
            bounds=(lower, upper),
            method="trf",
            x_scale="jac",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=self.max_nfev,
        )

        joint, joint_millideg = self._quantize_joint(
            np.asarray(optimizer.x, dtype=np.float64)
        )
        residual = self._pose_residual(
            joint, xyz_target_mm, rotation_target
        )
        position_error_mm = float(np.linalg.norm(residual[:3]))
        rotation_error_deg = float(np.linalg.norm(residual[3:]))
        joint_step_deg = np.degrees(np.abs(joint - seed))
        limit_margin_deg = np.degrees(
            np.minimum(joint - self._lower, self._upper - joint)
        )

        finite = bool(
            np.all(np.isfinite(joint))
            and np.isfinite(position_error_mm)
            and np.isfinite(rotation_error_deg)
        )
        within_limits = bool(
            np.all(joint >= self._lower - 1e-12)
            and np.all(joint <= self._upper + 1e-12)
        )
        # 量化最多为相邻两端各引入不到 0.001°；允许一个分辨率的数值余量。
        within_step = bool(
            np.max(joint_step_deg)
            <= self.max_step_deg + _JOINT_CTRL_RESOLUTION_DEG + 1e-9
        )
        within_pose = bool(
            position_error_mm <= self.position_tolerance_mm
            and rotation_error_deg <= self.rotation_tolerance_deg
        )
        optimizer_ok = bool(optimizer.success)
        success = (
            finite
            and optimizer_ok
            and within_limits
            and within_step
            and within_pose
        )

        reasons = []
        if not finite:
            reasons.append("non_finite")
        if not optimizer_ok:
            reasons.append("optimizer_failed")
        if not within_limits:
            reasons.append("joint_limit")
        if not within_step:
            reasons.append("joint_step")
        if not within_pose:
            reasons.append("pose_tolerance")
        reason = "ok" if success else ",".join(reasons)

        if success and commit:
            self._seed = joint.copy()

        return PiperIKResult(
            success=success,
            joint=joint.copy(),
            joint_millideg=joint_millideg.copy(),
            hold_joint=seed,
            position_error_mm=position_error_mm,
            rotation_error_deg=rotation_error_deg,
            joint_step_deg=joint_step_deg.copy(),
            limit_margin_deg=limit_margin_deg.copy(),
            nfev=int(optimizer.nfev),
            optimizer_success=optimizer_ok,
            optimizer_status=int(optimizer.status),
            reason=reason,
            message=str(optimizer.message),
        )
