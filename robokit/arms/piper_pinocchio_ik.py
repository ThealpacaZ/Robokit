"""Production Piper IK backed by a real Pinocchio URDF model.

The policy/executor still owns EEF-delta composition.  This module receives one
absolute EEF target at a time and solves only the six-axis joint target.  Every
solve starts from fresh physical feedback (supplied through ``reset_seed``),
uses mechanical and per-step bounds inside the optimizer, and returns a
JointCtrl-quantized candidate plus its Pinocchio FK realized pose.

``piper_sdk.C_PiperForwardKinematics`` is used only once at construction to
cross-check the vendored URDF's joint order, signs, zero and end frame.  It is
never used by the optimizer, FK result, Jacobian or arrival target.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pinocchio as pin
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from robokit.arms.piper_limits import DEFAULT_JOINT_LIMITS_DEG


_JOINT_CTRL_RESOLUTION_DEG = 0.001
_EXPECTED_JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 7))
_MAX_ABS_TARGET_POSITION_M = 5.0
_MAX_ABS_TARGET_EULER_RAD = 16.0 * np.pi
_DEFAULT_URDF = (
    Path(__file__).resolve().parent.parent
    / "assets"
    / "piper_description.urdf"
)


def _six_finite(values: Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape != (6,):
        raise ValueError(f"{name} 必须是 shape=(6,)，实际为 {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 含 NaN/Inf")
    return array


@dataclass(frozen=True)
class PiperPinocchioIKResult:
    """One bounded solve result; ``success`` means safe to command.

    Solver convergence and Cartesian residual are deliberately independent
    metadata.  In best-effort/no-guard mode a finite, bounded candidate remains
    commandable even when ``converged`` is false or the ideal pose is
    unreachable.
    """

    success: bool
    joint: np.ndarray
    joint_millideg: np.ndarray
    hold_joint: np.ndarray
    position_error_mm: float
    rotation_error_deg: float
    joint_step_deg: np.ndarray
    limit_margin_deg: np.ndarray
    converged: bool
    saturated: bool
    iterations: int
    optimizer_status: int
    reason: str
    message: str


class PiperPinocchioIK:
    """Continuous bounded six-axis Piper IK using Pinocchio FK and SE(3) log.

    The optimizer variable is bounded by the intersection of:

    * configured physical joint limits (optionally inset by ``limit_margin``);
    * the JointCtrl quantization-safe interval; and
    * ``seed ± max_step`` for every axis.

    Residuals use ``log6(current.inverse() * target)`` in the end-effector LOCAL
    frame.  The matching Pinocchio LOCAL frame Jacobian and ``Jlog6`` derivative
    are supplied to SciPy's bounded trust-region least-squares optimizer.
    """

    def __init__(
        self,
        initial_joint: Sequence[float],
        *,
        joint_limits_deg: Sequence[Sequence[float]] = DEFAULT_JOINT_LIMITS_DEG,
        urdf_path: str | Path | None = None,
        eef_frame: str = "link6",
        limit_margin_deg: float = 0.0,
        max_step_deg: float = 5.0,
        continuity_weight: float = 3e-2,
        position_tolerance_mm: float = 1.0,
        rotation_tolerance_deg: float = 1.0,
        seed_limit_tolerance_deg: float = 10.0,
        max_nfev: int = 200,
        allow_best_effort: bool = False,
        model_validation_position_mm: float = 0.25,
        model_validation_rotation_deg: float = 0.02,
    ) -> None:
        limits = np.asarray(joint_limits_deg, dtype=np.float64)
        if limits.shape != (6, 2) or not np.all(np.isfinite(limits)):
            raise ValueError("joint_limits_deg 必须是有限的 shape=(6, 2)")
        if np.any(limits[:, 0] >= limits[:, 1]):
            raise ValueError("每个关节的下限必须小于上限")
        for value, name, positive in (
            (limit_margin_deg, "limit_margin_deg", False),
            (max_step_deg, "max_step_deg", True),
            (continuity_weight, "continuity_weight", False),
            (position_tolerance_mm, "position_tolerance_mm", True),
            (rotation_tolerance_deg, "rotation_tolerance_deg", True),
            (
                seed_limit_tolerance_deg,
                "seed_limit_tolerance_deg",
                False,
            ),
            (
                model_validation_position_mm,
                "model_validation_position_mm",
                True,
            ),
            (
                model_validation_rotation_deg,
                "model_validation_rotation_deg",
                True,
            ),
        ):
            if not np.isfinite(value) or (value <= 0 if positive else value < 0):
                relation = "正" if positive else "非负"
                raise ValueError(f"{name} 必须是{relation}有限值")
        if int(max_nfev) <= 0:
            raise ValueError("max_nfev 必须为正整数")

        self._mechanical_lower = np.radians(limits[:, 0])
        self._mechanical_upper = np.radians(limits[:, 1])
        # Optimizer bounds are made safe for the final 0.001° JointCtrl
        # rounding.  This avoids solving outside the command lattice and then
        # clipping an arbitrary solution after the fact.
        command_lower_deg = np.ceil(
            limits[:, 0] / _JOINT_CTRL_RESOLUTION_DEG
        ) * _JOINT_CTRL_RESOLUTION_DEG
        command_upper_deg = np.floor(
            limits[:, 1] / _JOINT_CTRL_RESOLUTION_DEG
        ) * _JOINT_CTRL_RESOLUTION_DEG
        self._lower = np.radians(
            np.maximum(command_lower_deg, limits[:, 0] + limit_margin_deg)
        )
        self._upper = np.radians(
            np.minimum(command_upper_deg, limits[:, 1] - limit_margin_deg)
        )
        if np.any(self._lower >= self._upper):
            raise ValueError("limit_margin_deg 使关节上下限相交")

        self.max_step_deg = float(max_step_deg)
        self.continuity_weight = float(continuity_weight)
        self.position_tolerance_mm = float(position_tolerance_mm)
        self.rotation_tolerance_deg = float(rotation_tolerance_deg)
        self.seed_limit_tolerance_deg = float(seed_limit_tolerance_deg)
        self.max_nfev = int(max_nfev)
        self.allow_best_effort = bool(allow_best_effort)
        self.urdf_path = Path(urdf_path or _DEFAULT_URDF).resolve()
        if not self.urdf_path.is_file():
            raise FileNotFoundError(f"Pinocchio URDF 不存在：{self.urdf_path}")

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        self.joint_names = tuple(self.model.names[1:])
        if (
            self.model.nq != 6
            or self.model.nv != 6
            or self.joint_names != _EXPECTED_JOINT_NAMES
        ):
            raise ValueError(
                "Pinocchio URDF 关节契约不匹配："
                f"nq={self.model.nq}, nv={self.model.nv}, "
                f"joints={self.joint_names}，期望 {_EXPECTED_JOINT_NAMES}"
            )
        if not self.model.existFrame(eef_frame):
            raise ValueError(f"Pinocchio URDF 缺少末端 frame {eef_frame!r}")
        self.eef_frame = str(eef_frame)
        self.eef_frame_id = self.model.getFrameId(self.eef_frame)
        if self.model.frames[self.eef_frame_id].parentJoint != 6:
            raise ValueError(
                f"末端 frame {eef_frame!r} 不属于 joint6"
            )
        if not np.allclose(
            self.model.lowerPositionLimit,
            self._mechanical_lower,
            atol=1e-10,
            rtol=0.0,
        ) or not np.allclose(
            self.model.upperPositionLimit,
            self._mechanical_upper,
            atol=1e-10,
            rtol=0.0,
        ):
            raise ValueError(
                "URDF joint limits 与配置 joint_limits_deg 不一致；"
                "拒绝在未核对的模型上求解"
            )

        self._validate_against_sdk_fk(
            float(model_validation_position_mm),
            float(model_validation_rotation_deg),
        )
        self._seed = np.zeros(6, dtype=np.float64)
        self.reset_seed(initial_joint)

    @property
    def seed_joint(self) -> np.ndarray:
        return self._seed.copy()

    def reset_seed(self, joint: Sequence[float]) -> np.ndarray:
        """Project physical feedback by the 10° rule and return per-axis overrun.

        Overrun is measured only against the physical mechanical limits.  A
        legal feedback sample may be moved a further tiny amount by an
        optimizer-only ``limit_margin_deg``; that is not reported as physical
        limit overrun.
        """

        feedback = _six_finite(joint, "joint")
        projected = np.clip(
            feedback, self._mechanical_lower, self._mechanical_upper
        )
        overrun_deg = np.degrees(np.abs(feedback - projected))
        maximum = float(overrun_deg.max())
        if maximum > self.seed_limit_tolerance_deg + 1e-12:
            raise ValueError(
                "物理关节反馈超出机械限位 "
                f"{maximum:.3f}° > 允许的 "
                f"{self.seed_limit_tolerance_deg:.3f}°"
            )
        self._seed = np.clip(projected, self._lower, self._upper)
        return overrun_deg

    def _placement(self, joint: np.ndarray) -> pin.SE3:
        pin.framesForwardKinematics(self.model, self.data, joint)
        return self.data.oMf[self.eef_frame_id].copy()

    @staticmethod
    def _target(pose: Sequence[float]) -> pin.SE3:
        target = _six_finite(pose, "pose")
        # Best-effort mode accepts kinematically unreachable poses, but a
        # millimetre-as-metre or degree-as-radian payload is an input contract
        # failure rather than an unreachable Cartesian request.  These limits
        # are deliberately far beyond Piper's workspace and normal unwrapped
        # Euler traces, so they do not act as a workspace safety guard.
        if np.max(np.abs(target[:3])) > _MAX_ABS_TARGET_POSITION_M:
            raise ValueError(
                "pose 位置绝对值超过 5m，疑似把 mm 当成 m"
            )
        if np.max(np.abs(target[3:])) > _MAX_ABS_TARGET_EULER_RAD:
            raise ValueError(
                "pose 欧拉角绝对值超过 16π rad，疑似把 degree 当成 rad"
            )
        return pin.SE3(
            Rotation.from_euler("xyz", target[3:]).as_matrix(),
            target[:3],
        )

    def _log_error(
        self, joint: np.ndarray, target: pin.SE3
    ) -> tuple[np.ndarray, pin.SE3]:
        current = self._placement(joint)
        current_to_target = current.actInv(target)
        return np.asarray(pin.log6(current_to_target).vector), current_to_target

    def _residual(
        self, joint: np.ndarray, target: pin.SE3, seed: np.ndarray
    ) -> np.ndarray:
        error, _ = self._log_error(joint, target)
        scaled = error.copy()
        scaled[:3] *= 1000.0
        scaled[3:] *= 180.0 / np.pi
        continuity = (
            self.continuity_weight * np.degrees(joint - seed)
        )
        return np.concatenate((scaled, continuity))

    def _jacobian(
        self, joint: np.ndarray, target: pin.SE3, seed: np.ndarray
    ) -> np.ndarray:
        _, current_to_target = self._log_error(joint, target)
        local = pin.computeFrameJacobian(
            self.model,
            self.data,
            joint,
            self.eef_frame_id,
            pin.ReferenceFrame.LOCAL,
        )
        pose_jacobian = (
            -np.asarray(pin.Jlog6(current_to_target.inverse()))
            @ np.asarray(local)
        )
        pose_jacobian[:3] *= 1000.0
        pose_jacobian[3:] *= 180.0 / np.pi
        continuity = (
            self.continuity_weight
            * (180.0 / np.pi)
            * np.eye(6, dtype=np.float64)
        )
        return np.vstack((pose_jacobian, continuity))

    @staticmethod
    def _quantize_joint(
        joint: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        raw = np.rint(np.degrees(joint) * 1000.0).astype(np.int64)
        quantized = np.radians(raw.astype(np.float64) / 1000.0)
        return quantized, raw

    def forward_pose(self, joint: Sequence[float]) -> np.ndarray:
        """Pinocchio FK pose represented by a quantized JointCtrl target."""

        candidate = _six_finite(joint, "joint")
        placement = self._placement(candidate)
        return np.concatenate(
            (
                np.asarray(placement.translation),
                Rotation.from_matrix(
                    np.asarray(placement.rotation)
                ).as_euler("xyz"),
            )
        )

    def evaluate(
        self, joint: Sequence[float], pose: Sequence[float]
    ) -> tuple[float, float]:
        candidate = _six_finite(joint, "joint")
        target = self._target(pose)
        actual = self._placement(candidate)
        position_error_mm = float(
            np.linalg.norm(actual.translation - target.translation) * 1000.0
        )
        rotation_error_deg = float(
            np.degrees(
                Rotation.from_matrix(
                    np.asarray(actual.rotation).T
                    @ np.asarray(target.rotation)
                ).magnitude()
            )
        )
        return position_error_mm, rotation_error_deg

    def solve(
        self, pose: Sequence[float], *, commit: bool = True
    ) -> PiperPinocchioIKResult:
        """Return the best bounded candidate, independently of convergence."""

        target = self._target(pose)
        seed = self._seed.copy()
        step = np.radians(self.max_step_deg)
        lower = np.maximum(self._lower, seed - step)
        upper = np.minimum(self._upper, seed + step)
        x0 = np.clip(seed, lower + 1e-12, upper - 1e-12)

        optimizer = least_squares(
            self._residual,
            x0,
            jac=self._jacobian,
            args=(target, seed),
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
        position_error_mm, rotation_error_deg = self.evaluate(joint, pose)
        joint_step_deg = np.degrees(np.abs(joint - seed))
        limit_margin_deg = np.degrees(
            np.minimum(
                joint - self._mechanical_lower,
                self._mechanical_upper - joint,
            )
        )

        finite = bool(
            np.all(np.isfinite(joint))
            and np.isfinite(position_error_mm)
            and np.isfinite(rotation_error_deg)
        )
        within_limits = bool(
            np.all(joint >= self._mechanical_lower - 1e-12)
            and np.all(joint <= self._mechanical_upper + 1e-12)
        )
        within_step = bool(
            np.max(joint_step_deg)
            <= self.max_step_deg + _JOINT_CTRL_RESOLUTION_DEG + 1e-9
        )
        within_pose = bool(
            position_error_mm <= self.position_tolerance_mm
            and rotation_error_deg <= self.rotation_tolerance_deg
        )
        converged = bool(optimizer.success)
        bound_distance_deg = np.degrees(
            np.minimum(joint - lower, upper - joint)
        )
        saturated = bool(
            np.any(
                bound_distance_deg
                <= _JOINT_CTRL_RESOLUTION_DEG + 1e-9
            )
        )
        hard_valid = finite and within_limits and within_step
        success = bool(
            hard_valid and (self.allow_best_effort or within_pose)
        )

        reasons: list[str] = []
        if not finite:
            reasons.append("non_finite")
        if not within_limits:
            reasons.append("joint_limit")
        if not within_step:
            reasons.append("joint_step")
        if not within_pose:
            reasons.append("pose_residual")
        if not converged:
            reasons.append("max_iterations")
        if success:
            reason = "ok" if not reasons else "best_effort"
        else:
            reason = ",".join(reasons) or "no_valid_candidate"

        if success and commit:
            self._seed = joint.copy()

        return PiperPinocchioIKResult(
            success=success,
            joint=joint.copy(),
            joint_millideg=joint_millideg.copy(),
            hold_joint=seed,
            position_error_mm=position_error_mm,
            rotation_error_deg=rotation_error_deg,
            joint_step_deg=joint_step_deg.copy(),
            limit_margin_deg=limit_margin_deg.copy(),
            converged=converged,
            saturated=saturated,
            iterations=int(optimizer.nfev),
            optimizer_status=int(optimizer.status),
            reason=reason,
            message=str(optimizer.message),
        )

    def _validate_against_sdk_fk(
        self, position_tolerance_mm: float, rotation_tolerance_deg: float
    ) -> None:
        """Fail closed if URDF coordinates differ from the deployed SDK model."""

        from piper_sdk import C_PiperForwardKinematics

        sdk_fk = C_PiperForwardKinematics(dh_is_offset=1)
        samples_deg = (
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            (-82.0, 2.0, -2.0, -1.0, 25.0, 5.0),
            (-80.0, 20.0, -30.0, 5.0, 25.0, 10.0),
            (30.0, 90.0, -90.0, 20.0, -30.0, 40.0),
        )
        worst_position = 0.0
        worst_rotation = 0.0
        for sample_deg in samples_deg:
            joint = np.radians(sample_deg)
            placement = self._placement(joint)
            sdk_pose = np.asarray(
                sdk_fk.CalFK(joint.tolist())[-1], dtype=np.float64
            )
            sdk_rotation = Rotation.from_euler(
                "xyz", sdk_pose[3:], degrees=True
            ).as_matrix()
            worst_position = max(
                worst_position,
                float(
                    np.linalg.norm(
                        placement.translation * 1000.0 - sdk_pose[:3]
                    )
                ),
            )
            worst_rotation = max(
                worst_rotation,
                float(
                    np.degrees(
                        Rotation.from_matrix(
                            np.asarray(placement.rotation).T @ sdk_rotation
                        ).magnitude()
                    )
                ),
            )
        if (
            worst_position > position_tolerance_mm
            or worst_rotation > rotation_tolerance_deg
        ):
            raise ValueError(
                "Pinocchio URDF 与 Piper SDK FK 不一致："
                f"position={worst_position:.4f}mm "
                f"(限 {position_tolerance_mm:.4f}), "
                f"rotation={worst_rotation:.4f}° "
                f"(限 {rotation_tolerance_deg:.4f})"
            )
        self.model_validation = {
            "position_error_mm": worst_position,
            "rotation_error_deg": worst_rotation,
            "joint_names": list(self.joint_names),
            "eef_frame": self.eef_frame,
        }
