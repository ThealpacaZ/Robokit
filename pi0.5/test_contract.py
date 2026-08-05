#!/usr/bin/env python3
"""Small dependency-light tests for the HDF5→LeRobot action contract."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import h5py
import numpy as np


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / "scripts"))
spec = importlib.util.spec_from_file_location("pi05_convert", HERE / "convert_hdf5_to_lerobot.py")
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

from robokit.deploy.rtc import RTCController, RTCDeadlineError
from robokit.deploy.runtime import validate_inference_response
from serve_policy import RTCService

RTC_KEYS = ("action_chunk", "chunk_id", "prediction_horizon")


def _validated_rtc_response(response, action_space):
    """真机 RTC 客户端对回包的那道校验，见 robokit/deploy/loops.py 的 _rtc_response。"""
    return validate_inference_response(
        response, "rtc", action_space, required_keys=RTC_KEYS
    )


class ContractTest(unittest.TestCase):
    def test_identity_orientation_translation_is_local_delta(self):
        poses = np.array(
            [
                [0.1, 0.2, 0.3, 0.0, 0.0, 0.0],
                [0.11, 0.18, 0.33, 0.01, -0.02, 0.03],
            ]
        )
        got = module.local_delta_pose_batch(poses)[0]
        np.testing.assert_allclose(got, [0.01, -0.02, 0.03, 0.01, -0.02, 0.03], atol=1e-7)

    def test_delta_round_trip_matches_robokit_executor_math(self):
        import sys

        sys.path.insert(0, str(HERE.parent))
        from robokit.pose import apply_local_delta_pose

        poses = np.array(
            [
                [0.25, -0.1, 0.2, 0.4, -0.3, 0.2],
                [0.27, -0.08, 0.19, 0.45, -0.25, 0.15],
            ]
        )
        delta = module.local_delta_pose_batch(poses)[0]
        rebuilt = apply_local_delta_pose(poses[0], delta)
        np.testing.assert_allclose(rebuilt, poses[1], atol=1e-6)

    def test_joint_contract_is_next_frame_absolute_radians(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "0.hdf5"
            joint = np.array(
                [
                    [0.0, 0.1, -0.2, 0.3, -0.4, 0.5],
                    [0.01, 0.12, -0.18, 0.28, -0.35, 0.45],
                    [0.02, 0.15, -0.16, 0.25, -0.30, 0.40],
                ],
                dtype=np.float32,
            )
            gripper = np.array([[0.1], [0.2], [0.3]], dtype=np.float32)
            with h5py.File(path, "w") as episode:
                episode.create_dataset("observations/right_arm/joint", data=joint)
                episode.create_dataset("observations/right_arm/gripper", data=gripper)
            state, action = module.episode_arrays(path, "right_arm", "joint")
            np.testing.assert_array_equal(state[:, :6], joint[:-1])
            np.testing.assert_array_equal(action[:, :6], joint[1:])
            np.testing.assert_array_equal(state[:, 6], gripper[:-1, 0])
            np.testing.assert_array_equal(action[:, 6], gripper[1:, 0])
            self.assertEqual(
                module.feature_names("joint"),
                (module.JOINT_STATE_NAMES, module.JOINT_ACTION_NAMES),
            )


class RTCControllerTest(unittest.TestCase):
    def test_algorithm_one_skips_actions_consumed_during_inference(self):
        controller = RTCController(
            prediction_horizon=8,
            min_execution_horizon=3,
            initial_delay=2,
            delay_buffer_size=3,
        )
        first = np.arange(8 * 7, dtype=np.float32).reshape(8, 7)
        second = first + 1000
        controller.initialize(first, "first")

        for expected_row in range(3):
            action, row, chunk_id = controller.next_action()
            self.assertEqual((row, chunk_id), (expected_row, "first"))
            np.testing.assert_array_equal(action, first[expected_row])
        request = controller.start_inference()
        self.assertEqual(request.executed_at_start, 3)
        self.assertEqual(request.inference_delay, 2)

        controller.next_action()
        controller.next_action()
        observed = controller.accept_inference(second, "second")
        self.assertEqual(observed, 2)
        action, row, chunk_id = controller.next_action()
        self.assertEqual((row, chunk_id), (2, "second"))
        np.testing.assert_array_equal(action, second[2])

    def test_deadline_constraint_is_a_hard_abort(self):
        controller = RTCController(
            prediction_horizon=8,
            min_execution_horizon=3,
            initial_delay=3,
        )
        controller.initialize(np.zeros((8, 7), dtype=np.float32), "first")
        for _ in range(6):
            controller.next_action()
        with self.assertRaises(RTCDeadlineError):
            controller.start_inference()

    def test_chunk_exhaustion_never_repeats_last_action(self):
        controller = RTCController(
            prediction_horizon=4,
            min_execution_horizon=2,
            initial_delay=1,
        )
        controller.initialize(np.zeros((4, 7), dtype=np.float32), "first")
        for _ in range(4):
            controller.next_action()
        with self.assertRaises(RTCDeadlineError):
            controller.next_action()


class RTCServerContractTest(unittest.TestCase):
    def test_server_keeps_normalized_prefix_private(self):
        class FakePolicy:
            action_space = "eef_delta"

            def __init__(self):
                self.calls = []

            def infer_rtc(self, message, **kwargs):
                self.calls.append(kwargs)
                value = len(self.calls)
                physical = np.full((4, 7), value, dtype=np.float32)
                normalized = np.full((1, 4, 7), value * 10, dtype=np.float32)
                return physical, normalized

        policy = FakePolicy()
        session = RTCService(policy, cache_size=2)
        initial = session.handle({"images": {}, "state": {}, "instruction": "test"})
        self.assertNotIn("normalized_chunk", initial)
        self.assertEqual(initial["server_mode"], "rtc")
        self.assertEqual(initial["action_space"], "eef_delta")
        guided = session.handle(
            {
                "images": {},
                "state": {},
                "instruction": "test",
                "rtc": {
                    "previous_chunk_id": initial["chunk_id"],
                    "executed_at_start": 2,
                    "inference_delay": 1,
                },
            }
        )
        self.assertTrue(guided["rtc_guided"])
        self.assertEqual(policy.calls[1]["executed_at_start"], 2)
        self.assertEqual(policy.calls[1]["inference_delay"], 1)
        np.testing.assert_array_equal(
            policy.calls[1]["previous_chunk"],
            np.full((1, 4, 7), 10, dtype=np.float32),
        )


class InferenceModeHandshakeTest(unittest.TestCase):
    def test_sync_mode_accepts_only_explicit_sync_service(self):
        response = {
            "action_chunk": np.zeros((10, 7), dtype=np.float32),
            "server_mode": "sync",
            "action_space": "eef_delta",
        }
        self.assertIs(
            validate_inference_response(
                response,
                required_server_mode="sync",
                required_action_space="eef_delta",
            ),
            response,
        )
        with self.assertRaisesRegex(RuntimeError, "sync/RTC port mismatch"):
            validate_inference_response(
                {**response, "server_mode": "rtc"},
                required_server_mode="sync",
            )
        with self.assertRaisesRegex(RuntimeError, "server_mode=None"):
            validate_inference_response(
                {"action_chunk": response["action_chunk"]},
                required_server_mode="sync",
            )
        with self.assertRaisesRegex(RuntimeError, "EEF/joint service mismatch"):
            validate_inference_response(
                {**response, "action_space": "joint"},
                required_action_space="eef_delta",
            )
        with self.assertRaisesRegex(RuntimeError, "action_space=None"):
            validate_inference_response(
                {"action_chunk": response["action_chunk"]},
                required_action_space="joint",
            )

    def test_rtc_mode_rejects_eef_joint_port_mismatch(self):
        response = {
            "action_chunk": np.zeros((50, 7), dtype=np.float32),
            "chunk_id": "chunk-0",
            "prediction_horizon": 50,
            "server_mode": "rtc",
            "action_space": "joint",
        }
        self.assertIs(_validated_rtc_response(response, "joint"), response)
        with self.assertRaisesRegex(RuntimeError, "service mismatch"):
            _validated_rtc_response(response, "eef_delta")


if __name__ == "__main__":
    unittest.main()
