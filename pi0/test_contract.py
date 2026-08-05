#!/usr/bin/env python3
"""Dependency-light checks for the absolute-joint prediction/execution contract.

不加载权重、不连硬件：只验证「模型输出 (H,7) → 登记表说执行前 N 步 → 服务端回包」
这条契约在重构后的统一入口上仍然成立。
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from robokit.deploy.registry import load_registry
from robokit.policies.lerobot_dit import LeRobotDiTPolicy
from serve_policy import RTCService

HERE = Path(__file__).resolve().parent


def _as_action_array(chunk, expected_rows=None, family="pi0"):
    """不实例化 policy（会加载权重）也能测形状校验：只用到 self.family。"""
    return LeRobotDiTPolicy._as_action_array(
        SimpleNamespace(family=family), chunk, expected_rows
    )


class JointActionArrayTest(unittest.TestCase):
    def test_batch_dimension_is_squeezed(self):
        prediction = np.arange(50 * 7, dtype=np.float32).reshape(1, 50, 7)
        result = _as_action_array(prediction)
        self.assertEqual(result.shape, (50, 7))
        np.testing.assert_array_equal(result, prediction[0])

    def test_wrong_action_width_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, r"shape \(N, 7\)"):
            _as_action_array(np.zeros((50, 6), dtype=np.float32))

    def test_non_finite_actions_are_rejected(self):
        chunk = np.zeros((50, 7), dtype=np.float32)
        chunk[7, 2] = np.nan
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            _as_action_array(chunk)

    def test_rtc_requires_the_full_predicted_horizon(self):
        with self.assertRaisesRegex(RuntimeError, "must return 50 actions"):
            _as_action_array(np.zeros((30, 7), dtype=np.float32), expected_rows=50)


class JointRegistryTest(unittest.TestCase):
    """执行 horizon 与动作空间现在由登记表定义，不再由 YAML 或代码里的常量定义。"""

    def setUp(self):
        self.registry = load_registry()

    def test_joint_model_declares_joint_space_and_its_execution_horizon(self):
        spec = self.registry["pi05-joint"]
        self.assertEqual(spec.action_space, "joint")
        self.assertEqual(spec.chunk_size, 50)
        # sync 默认执行前 15 步；RTC 的 s_min 也是 15。
        self.assertEqual(spec.execution_horizon("sync"), 15)
        self.assertEqual(spec.execution_horizon("rtc"), 15)
        # --horizon 覆盖，但不能超过模型真正预测的步数。
        self.assertEqual(spec.execution_horizon("sync", 30), 30)
        with self.assertRaisesRegex(ValueError, "chunk_size=50"):
            spec.execution_horizon("sync", 60)

    def test_joint_execution_yaml_is_locked_30hz_and_protected_by_default(self):
        config = yaml.safe_load(
            (REPO_ROOT / "configs" / "piper_single_joint.yaml").read_text(encoding="utf-8")
        )
        self.assertEqual(config["deploy"]["control_freq"], 30)
        self.assertTrue(config["deploy"]["control_freq_locked"])
        self.assertTrue(config["deploy"]["safety"]["enabled"])
        self.assertEqual(
            config["robot"]["arms"]["right_arm"]["eef_backend"], "pinocchio_ik"
        )
        # 动作空间/执行步数不再在 YAML 里，避免和登记表出现两个真相。
        self.assertNotIn("action_space", config["deploy"])
        self.assertNotIn("action_horizon", config["deploy"])


class JointRTCServiceTest(unittest.TestCase):
    def test_joint_rtc_server_caches_normalized_chunk_only(self):
        class FakePolicy:
            action_space = "joint"

            def __init__(self):
                self.calls = []

            def reset(self):
                pass

            def infer_rtc(self, message, **kwargs):
                self.calls.append(kwargs)
                value = len(self.calls)
                return (
                    np.full((50, 7), value, dtype=np.float32),
                    np.full((1, 50, 7), value * 10, dtype=np.float32),
                )

        policy = FakePolicy()
        service = RTCService(policy, cache_size=2)
        first = service.handle({"images": {}, "state": {}, "instruction": "test"})
        self.assertEqual(first["server_mode"], "rtc")
        self.assertEqual(first["action_space"], "joint")
        self.assertEqual(first["prediction_horizon"], 50)
        self.assertNotIn("normalized_chunk", first)
        second = service.handle(
            {
                "images": {},
                "state": {},
                "instruction": "test",
                "rtc": {
                    "previous_chunk_id": first["chunk_id"],
                    "executed_at_start": 15,
                    "inference_delay": 5,
                },
            }
        )
        self.assertTrue(second["rtc_guided"])
        self.assertEqual(policy.calls[1]["executed_at_start"], 15)
        np.testing.assert_array_equal(
            policy.calls[1]["previous_chunk"],
            np.full((1, 50, 7), 10, dtype=np.float32),
        )


if __name__ == "__main__":
    unittest.main()
