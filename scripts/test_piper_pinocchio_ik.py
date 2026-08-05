#!/usr/bin/env python3
"""Boundary regressions for the production Pinocchio Piper IK backend."""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from robokit.arms.piper import PiperArm  # noqa: E402
from robokit.arms.base import ArmCommandRejected  # noqa: E402
from robokit.arms.piper_pinocchio_ik import (  # noqa: E402
    PiperPinocchioIK,
)


LIMITS = np.array(
    [
        [-150.0, 150.0],
        [0.0, 180.0],
        [-170.0, 0.0],
        [-100.0, 100.0],
        [-70.0, 70.0],
        [-180.0, 180.0],
    ],
    dtype=np.float64,
)
INITIAL_DEG = np.array(
    [-82.0, 2.0, -2.0, -1.0, 25.0, 5.0],
    dtype=np.float64,
)


def make_ik(**kwargs) -> PiperPinocchioIK:
    options = {
        "joint_limits_deg": LIMITS,
        "seed_limit_tolerance_deg": 10.0,
        "allow_best_effort": True,
        "max_step_deg": 5.0,
        "continuity_weight": 0.03,
    }
    options.update(kwargs)
    return PiperPinocchioIK(np.radians(INITIAL_DEG), **options)


class _MemorySDK:
    def __init__(self, arm):
        self.arm = arm
        self.raw_commands = []

    def JointCtrl(self, *values):
        raw = np.asarray(values, dtype=np.int64)
        self.raw_commands.append(raw.copy())
        self.arm._joint = np.radians(raw.astype(np.float64) / 1000.0)
        self.arm._pose = self.arm._eef_ik.forward_pose(self.arm._joint)


class _OfflinePinocchioArm(PiperArm):
    def __init__(self, joint_limit_mode="reject"):
        cfg = {
            "type": "piper",
            "dof": 6,
            "eef_backend": "pinocchio_ik",
            "joint_limits_deg": LIMITS.tolist(),
            "joint_limit_mode": joint_limit_mode,
            "pinocchio_ik": {
                "allow_best_effort": True,
                "seed_limit_tolerance_deg": 10.0,
                "max_step_deg": 5.0,
                "continuity_weight": 0.03,
            },
        }
        super().__init__("right_arm", cfg)
        self.read_only = False
        self._joint = np.radians(INITIAL_DEG)
        self._eef_ik = self._create_eef_ik(self._joint)
        self._pose = self._eef_ik.forward_pose(self._joint)
        self._gripper = 0.0
        self.sdk = _MemorySDK(self)

    def get_state(self):
        return {
            "joint": self._joint.copy(),
            "eef_pose": self._pose.copy(),
            "gripper": self._gripper,
        }

    def _assert_controller_normal(self, where):
        return None

    def _ensure_motion_mode(self, move_mode):
        self._motion_mode = move_mode

    def _move_gripper(self, gripper):
        self._gripper = float(gripper)


class PiperPinocchioIKTest(unittest.TestCase):
    def test_invalid_joint_limit_mode_is_rejected(self):
        with self.assertRaisesRegex(
            ValueError, r"joint_limit_mode must be 'reject' or 'clip'"
        ):
            _OfflinePinocchioArm(joint_limit_mode="unknown")

    def test_direct_joint_default_still_rejects_out_of_limits(self):
        arm = _OfflinePinocchioArm()
        target_deg = np.array(
            [-160.0, -1.2, 0.9, 110.0, -80.0, 190.0],
            dtype=np.float64,
        )

        with self.assertRaises(ArmCommandRejected) as caught:
            arm.move_joint(np.radians(target_deg), gripper=0.4)

        self.assertEqual(caught.exception.kind, "joint_limit")
        self.assertEqual(len(arm.sdk.raw_commands), 0)

    def test_direct_joint_clip_mode_projects_each_axis_to_limits(self):
        arm = _OfflinePinocchioArm(joint_limit_mode="clip")
        requested_deg = np.array(
            [-160.0, -1.2, 0.9, 110.0, -80.0, 190.0],
            dtype=np.float64,
        )

        preview = arm.preview_joint(np.radians(requested_deg))
        self.assertEqual(len(arm.sdk.raw_commands), 0)

        command = arm.move_joint(
            np.radians(requested_deg),
            gripper=0.4,
        )

        expected_deg = np.array(
            [-150.0, 0.0, 0.0, 100.0, -70.0, 180.0],
            dtype=np.float64,
        )
        self.assertEqual(len(arm.sdk.raw_commands), 1)
        self.assertEqual(preview, command)
        np.testing.assert_array_equal(
            arm.sdk.raw_commands[0],
            np.rint(expected_deg * 1000.0).astype(np.int64),
        )
        self.assertTrue(command["joint_limit_clipped"])
        self.assertEqual(
            command["joint_clipped_axes"],
            ["j1", "j2", "j3", "j4", "j5", "j6"],
        )
        np.testing.assert_allclose(
            command["joint_requested_deg"],
            requested_deg,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            command["joint_target_deg"],
            expected_deg,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            command["joint_limit_clip_delta_deg"],
            expected_deg - requested_deg,
            atol=1e-12,
        )

    def test_direct_joint_clip_mode_still_rejects_nonfinite(self):
        arm = _OfflinePinocchioArm(joint_limit_mode="clip")
        target = np.radians(INITIAL_DEG)
        target[2] = np.nan

        with self.assertRaises(ArmCommandRejected) as caught:
            arm.move_joint(target)

        self.assertEqual(caught.exception.kind, "joint_target")
        self.assertEqual(len(arm.sdk.raw_commands), 0)

    def test_backend_is_real_pinocchio_not_host_ik_alias(self):
        cfg = {
            "type": "piper",
            "dof": 6,
            "eef_backend": "pinocchio_ik",
            "joint_limits_deg": LIMITS.tolist(),
            "pinocchio_ik": {
                "allow_best_effort": True,
                "seed_limit_tolerance_deg": 10.0,
            },
        }
        arm = PiperArm("right_arm", cfg)
        solver = arm._create_eef_ik(np.radians(INITIAL_DEG))
        self.assertIsInstance(solver, PiperPinocchioIK)
        self.assertEqual(solver.joint_names, tuple(f"joint{i}" for i in range(1, 7)))

    def test_urdf_fk_matches_sdk_joint_order_zero_sign_and_eef(self):
        ik = make_ik()
        self.assertEqual(ik.model.nq, 6)
        self.assertEqual(ik.model.nv, 6)
        self.assertEqual(ik.eef_frame, "link6")
        self.assertLessEqual(
            ik.model_validation["position_error_mm"], 0.25
        )
        self.assertLessEqual(
            ik.model_validation["rotation_error_deg"], 0.02
        )

    def test_feedback_exactly_ten_degrees_over_limit_projects(self):
        ik = make_ik()
        feedback = INITIAL_DEG.copy()
        feedback[4] = 80.0

        overrun = ik.reset_seed(np.radians(feedback))

        self.assertAlmostEqual(overrun[4], 10.0, places=9)
        self.assertAlmostEqual(
            np.degrees(ik.seed_joint[4]), 70.0, places=9
        )

    def test_feedback_ten_point_zero_zero_one_over_limit_rejects(self):
        ik = make_ik()
        feedback = INITIAL_DEG.copy()
        feedback[4] = 80.001

        with self.assertRaisesRegex(
            ValueError, r"10\.001° > 允许的 10\.000°"
        ):
            ik.reset_seed(np.radians(feedback))

    def test_unreachable_pose_returns_nearest_bounded_candidate(self):
        ik = make_ik()
        unreachable = np.array(
            [1.5, -1.5, 1.5, 2.0, -1.0, 2.5],
            dtype=np.float64,
        )

        result = ik.solve(unreachable, commit=False)

        self.assertTrue(result.success)
        self.assertGreater(result.position_error_mm, 1000.0)
        self.assertTrue(np.all(np.isfinite(result.joint)))
        joint_deg = np.degrees(result.joint)
        self.assertTrue(np.all(joint_deg >= LIMITS[:, 0] - 1e-9))
        self.assertTrue(np.all(joint_deg <= LIMITS[:, 1] + 1e-9))
        self.assertLessEqual(result.joint_step_deg.max(), 5.001)
        self.assertEqual(result.reason, "best_effort")

    def test_obvious_millimetre_as_metre_target_rejects(self):
        ik = make_ik()
        with self.assertRaisesRegex(ValueError, r"把 mm 当成 m"):
            ik.solve(np.array([300.0, 0, 200.0, 0, 0, 0]))

    def test_quantized_jointctrl_never_crosses_mechanical_limits(self):
        ik = make_ik()
        target = np.array(
            [-5.0, 5.0, -5.0, -2.5, 1.2, -2.0],
            dtype=np.float64,
        )

        result = ik.solve(target, commit=False)

        command_deg = result.joint_millideg.astype(np.float64) / 1000.0
        self.assertTrue(np.all(command_deg >= LIMITS[:, 0]))
        self.assertTrue(np.all(command_deg <= LIMITS[:, 1]))
        np.testing.assert_allclose(
            np.radians(command_deg), result.joint, atol=1e-15, rtol=0
        )

    def test_max_iterations_keeps_legal_best_candidate(self):
        ik = make_ik(max_nfev=1)
        target = np.array(
            [2.0, -3.0, 4.0, 1.0, -1.0, 2.0],
            dtype=np.float64,
        )

        result = ik.solve(target, commit=False)

        self.assertTrue(result.success)
        self.assertFalse(result.converged)
        self.assertEqual(result.iterations, 1)
        self.assertLessEqual(result.joint_step_deg.max(), 5.001)
        self.assertEqual(result.reason, "best_effort")

    def test_production_move_eef_records_fields_and_ignores_servo_lag(self):
        arm = _OfflinePinocchioArm()
        stale_command = INITIAL_DEG.copy()
        stale_command[0] += 5.0
        arm._eef_ik.reset_seed(np.radians(stale_command))

        command = arm.move_eef(arm._pose.copy(), gripper=0.4)

        required = {
            "backend",
            "joint_target_deg",
            "eef_target_realized",
            "position_error_mm",
            "rotation_error_deg",
            "joint_step_deg",
            "limit_margin_deg",
            "seed_tracking_deg",
            "seed_limit_overrun_deg",
            "converged",
            "saturated",
            "iterations",
        }
        self.assertTrue(required.issubset(command))
        self.assertEqual(command["backend"], "pinocchio_ik_move_j")
        self.assertAlmostEqual(max(command["seed_tracking_deg"]), 5.0)
        self.assertEqual(len(arm.sdk.raw_commands), 1)

    def test_connect_path_does_not_hide_over_ten_degree_feedback(self):
        arm = _OfflinePinocchioArm()
        feedback = INITIAL_DEG.copy()
        feedback[4] = 80.001
        arm._joint = np.radians(feedback)

        with self.assertRaisesRegex(
            RuntimeError, r"10\.001° > 10\.000°"
        ):
            arm._recover_joint_limits()


if __name__ == "__main__":
    unittest.main(verbosity=2)
