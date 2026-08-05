#!/usr/bin/env python
"""Pure-software regression tests for the policy-to-Piper safety path."""

import os
import sys
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from robokit.executor import ChunkExecutor  # noqa: E402
from robokit.arms.piper_ik import PiperContinuousIK  # noqa: E402
from robokit.safety import ActionGuard, GuardViolation  # noqa: E402
from robokit.deploy.runtime import relax_safety, resolve_control_freq  # noqa: E402


class FakeClock:
    def __init__(self):
        self.value = 0.0

    def now(self):
        return self.value

    def sleep(self, duration):
        self.value += duration


class FakeHostIKArm:
    dof = 6

    def __init__(
        self,
        stuck=False,
        realized_pose_offset=None,
        backend="host_ik_move_j",
    ):
        self.pose = np.zeros(6, dtype=np.float64)
        self.joint = np.zeros(6, dtype=np.float64)
        self.stuck = stuck
        self.realized_pose_offset = (
            np.zeros(6, dtype=np.float64)
            if realized_pose_offset is None
            else np.asarray(realized_pose_offset, dtype=np.float64)
        )
        self.backend = str(backend)
        self.move_calls = 0
        self.health_checks = 0

    def get_state(self):
        return {
            "eef_pose": self.pose.copy(),
            "joint": self.joint.copy(),
            "gripper": 0.0,
        }

    def move_eef(self, target, gripper=None):
        self.move_calls += 1
        joint_target = np.full(6, 0.001, dtype=np.float64)
        realized = (
            np.asarray(target, dtype=np.float64)
            + self.realized_pose_offset
        )
        if not self.stuck:
            self.pose = realized.copy()
            self.joint = joint_target.copy()
        return {
            "backend": self.backend,
            "joint_target_deg": np.degrees(joint_target).tolist(),
            "eef_target_realized": realized.tolist(),
            "position_error_mm": float(
                np.linalg.norm(self.realized_pose_offset[:3]) * 1000
            ),
            "rotation_error_deg": float(
                np.degrees(
                    Rotation.from_euler(
                        "xyz", self.realized_pose_offset[3:]
                    ).magnitude()
                )
            ),
        }

    def assert_healthy(self, where):
        self.health_checks += 1


class FakeJointArm:
    dof = 6

    def __init__(self):
        self.joint = np.zeros(6, dtype=np.float64)
        self.commands = []

    def get_state(self):
        return {
            "eef_pose": np.zeros(6, dtype=np.float64),
            "joint": self.joint.copy(),
            "gripper": 0.0,
        }

    def move_joint(self, joint, gripper=None):
        self.joint = np.asarray(joint, dtype=np.float64)
        self.commands.append((self.joint.copy(), gripper))
        return {"joint_target_deg": np.degrees(self.joint).tolist()}


class FakeRobot:
    def __init__(self, arm):
        self.arms = {"right_arm": arm}


class PolicyControlTest(unittest.TestCase):
    def test_locked_control_frequency_is_permanently_30hz(self):
        cfg = {
            "control_freq": 30,
            "control_freq_locked": True,
        }
        self.assertEqual(resolve_control_freq(cfg), 30.0)
        self.assertEqual(resolve_control_freq(cfg, 30), 30.0)
        with self.assertRaisesRegex(
            ValueError, r"permanently locked.*30Hz.*29Hz"
        ):
            resolve_control_freq(cfg, 29)

    def test_fixed_control_rate_subtracts_per_step_overhead(self):
        clock = FakeClock()
        arm = FakeJointArm()
        command_times = []
        original_move = arm.move_joint

        def move_with_overhead(joint, gripper=None):
            command_times.append(clock.now())
            clock.value += 0.005
            return original_move(joint, gripper)

        arm.move_joint = move_with_overhead
        executor = ChunkExecutor(
            action_space="joint",
            horizon=15,
            control_freq=30,
            fixed_control_rate=True,
            clock=clock.now,
            sleep=clock.sleep,
        )
        target = np.radians(
            [-80.0, 20.0, -30.0, 5.0, 25.0, 10.0]
        )
        chunk = np.tile(
            np.concatenate([target, [0.4]]),
            (15, 1),
        )

        result = executor.execute(FakeRobot(arm), chunk)

        self.assertEqual(result, ("ok", {"steps": 15}))
        np.testing.assert_allclose(
            np.diff(command_times),
            np.full(14, 1.0 / 30.0),
            atol=1e-12,
            rtol=0.0,
        )
        self.assertAlmostEqual(clock.now(), 15.0 / 30.0)

    @staticmethod
    def _execute(
        stuck, realized_pose_offset=None, backend="host_ik_move_j"
    ):
        clock = FakeClock()
        arm = FakeHostIKArm(
            stuck=stuck,
            realized_pose_offset=realized_pose_offset,
            backend=backend,
        )
        robot = FakeRobot(arm)
        executor = ChunkExecutor(
            horizon=1,
            control_freq=30,
            wait_arrival=(0.0005, 0.05),
            arrival_stable_s=0.01,
            clock=clock.now,
            sleep=clock.sleep,
        )
        obs = {"arms": {"right_arm": arm.get_state()}}
        result = executor.execute(
            robot,
            np.array([[0.002, 0, 0, 0, 0, 0, 0]], dtype=np.float64),
            obs,
        )
        return result, arm

    def test_full_pose_arrival_succeeds(self):
        result, arm = self._execute(stuck=False)
        self.assertEqual(result[0], "ok")
        self.assertEqual(arm.move_calls, 1)
        self.assertGreater(arm.health_checks, 0)

    def test_arrival_timeout_aborts_without_second_command(self):
        result, arm = self._execute(stuck=True)
        self.assertEqual(result[0], "aborted")
        self.assertEqual(result[1]["kind"], "arrival_timeout")
        self.assertEqual(arm.move_calls, 1)

    def test_dry_run_skips_only_static_feedback_tracking_guard(self):
        clock = FakeClock()
        arm = FakeHostIKArm(stuck=True)
        guard = ActionGuard()
        executor = ChunkExecutor(
            horizon=30,
            control_freq=30,
            guard=guard,
            guard_tracking=False,
            clock=clock.now,
            sleep=clock.sleep,
        )
        obs = {"arms": {"right_arm": arm.get_state()}}
        chunk = np.tile(
            np.array([[0.003, 0, 0, 0, 0, 0, 0]], dtype=np.float64),
            (30, 1),
        )

        result = executor.execute(FakeRobot(arm), chunk, obs)

        self.assertEqual(result, ("ok", {"steps": 30}))
        self.assertEqual(arm.move_calls, 30)
        self.assertGreater(
            np.linalg.norm(executor._last_target["right_arm"][:3]),
            guard.cfg["max_tracking_err"],
        )

    def test_live_run_still_aborts_on_static_feedback_tracking(self):
        clock = FakeClock()
        arm = FakeHostIKArm(stuck=True)
        executor = ChunkExecutor(
            horizon=30,
            control_freq=30,
            guard=ActionGuard(),
            clock=clock.now,
            sleep=clock.sleep,
        )
        obs = {"arms": {"right_arm": arm.get_state()}}
        chunk = np.tile(
            np.array([[0.003, 0, 0, 0, 0, 0, 0]], dtype=np.float64),
            (30, 1),
        )

        result = executor.execute(FakeRobot(arm), chunk, obs)

        self.assertEqual(result[0], "aborted")
        self.assertEqual(result[1]["kind"], "tracking")
        self.assertLess(arm.move_calls, 30)

    def test_dry_run_keeps_non_tracking_safety_guards(self):
        clock = FakeClock()
        arm = FakeHostIKArm(stuck=True)
        executor = ChunkExecutor(
            horizon=1,
            control_freq=30,
            guard=ActionGuard(),
            guard_tracking=False,
            clock=clock.now,
            sleep=clock.sleep,
        )
        obs = {"arms": {"right_arm": arm.get_state()}}

        result = executor.execute(
            FakeRobot(arm),
            np.array([[0.031, 0, 0, 0, 0, 0, 0]], dtype=np.float64),
            obs,
        )

        self.assertEqual(result[0], "aborted")
        self.assertEqual(result[1]["kind"], "delta_xyz")
        self.assertEqual(arm.move_calls, 0)

    def test_host_ik_arrival_tracks_realized_fk_not_unreachable_request(self):
        result, arm = self._execute(
            stuck=False,
            realized_pose_offset=np.array(
                [0.0006, 0, 0, 0, 0, np.radians(0.91)],
                dtype=np.float64,
            ),
        )
        self.assertEqual(result[0], "ok")
        self.assertEqual(arm.move_calls, 1)

    def test_pinocchio_arrival_tracks_realized_fk_not_ideal_request(self):
        result, arm = self._execute(
            stuck=False,
            realized_pose_offset=np.array(
                [0.0006, 0, 0, 0, 0, np.radians(0.91)],
                dtype=np.float64,
            ),
            backend="pinocchio_ik_move_j",
        )
        self.assertEqual(result[0], "ok")
        self.assertEqual(arm.move_calls, 1)

    def test_host_ik_forward_pose_matches_evaluator_contract(self):
        joint = np.radians([-80.0, 10.0, -20.0, 0.0, 20.0, 5.0])
        ik = PiperContinuousIK(joint)
        pose = ik.forward_pose(joint)
        position_error_mm, rotation_error_deg = ik.evaluate(joint, pose)
        self.assertLess(position_error_mm, 1e-9)
        self.assertLess(rotation_error_deg, 1e-9)

    def test_host_ik_projects_feedback_overrun_up_to_configured_limit(self):
        joint = np.radians([-80.0, 10.0, -20.0, 0.0, 20.0, 5.0])
        ik = PiperContinuousIK(
            joint,
            seed_limit_tolerance_deg=10.0,
        )
        overrun = ik.reset_seed(
            np.radians([-80.0, 10.0, -20.0, 0.0, 80.0, 5.0])
        )
        self.assertAlmostEqual(overrun[4], 10.0)
        self.assertAlmostEqual(np.degrees(ik.seed_joint[4]), 70.0)

    def test_host_ik_rejects_feedback_overrun_above_configured_limit(self):
        joint = np.radians([-80.0, 10.0, -20.0, 0.0, 20.0, 5.0])
        ik = PiperContinuousIK(
            joint,
            seed_limit_tolerance_deg=10.0,
        )
        with self.assertRaisesRegex(ValueError, r"10\.001° > 允许的 10\.000°"):
            ik.reset_seed(
                np.radians(
                    [-80.0, 10.0, -20.0, 0.0, 80.001, 5.0]
                )
            )

    def test_rotation_guard_uses_so3_angle(self):
        guard = ActionGuard(max_delta_rpy=np.radians(20.0))
        delta = np.concatenate(
            (np.zeros(3), np.radians([19.5, 19.5, 19.5]))
        )
        with self.assertRaises(GuardViolation) as caught:
            guard.check_eef(delta, np.zeros(6))
        self.assertEqual(caught.exception.kind, "delta_rpy")

    def test_joint_action_is_an_absolute_radian_target(self):
        clock = FakeClock()
        arm = FakeJointArm()
        target = np.radians([-80.0, 20.0, -30.0, 5.0, 25.0, 10.0])
        guard = ActionGuard(
            joint_limits_deg=[
                [-150, 150],
                [0, 180],
                [-170, 0],
                [-100, 100],
                [-70, 70],
                [-180, 180],
            ]
        )
        executor = ChunkExecutor(
            action_space="joint",
            horizon=1,
            control_freq=30,
            guard=guard,
            clock=clock.now,
            sleep=clock.sleep,
        )
        result = executor.execute(
            FakeRobot(arm),
            np.array([np.concatenate([target, [0.4]])]),
        )
        self.assertEqual(result[0], "ok")
        self.assertEqual(len(arm.commands), 1)
        np.testing.assert_allclose(arm.commands[0][0], target)
        self.assertAlmostEqual(arm.commands[0][1], 0.4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
