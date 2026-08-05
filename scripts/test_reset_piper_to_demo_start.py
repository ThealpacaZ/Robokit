#!/usr/bin/env python
"""复位脚本参数的纯软件回归测试；不会导入 SDK 或连接 CAN。"""

import importlib.util
import os
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SCRIPT_PATH = os.path.join(ROOT, "scripts", "reset_piper_to_demo_start.py")
SPEC = importlib.util.spec_from_file_location(
    "robokit_reset_piper_to_demo_start",
    SCRIPT_PATH,
)
reset_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = reset_script
SPEC.loader.exec_module(reset_script)


class ResetPiperToDemoStartTest(unittest.TestCase):
    def _parse(self, *args):
        with patch.object(
            sys,
            "argv",
            ["reset_piper_to_demo_start.py", *args],
        ):
            return reset_script._parse_args()

    def test_default_allows_any_total_gap_but_keeps_segment_limit(self):
        args = self._parse()

        self.assertEqual(args.max_start_gap_deg, 0.0)
        self.assertEqual(args.step_deg, 15.0)

    def test_explicit_stricter_start_gap_is_preserved(self):
        args = self._parse("--max-start-gap-deg", "10")

        self.assertEqual(args.max_start_gap_deg, 10.0)
        self.assertEqual(args.step_deg, 15.0)

    def test_large_reset_is_split_directly_into_monitored_segments(self):
        class FakeSDK:
            def MotionCtrl_2(self, *args):
                pass

        start = np.zeros(6, dtype=np.float64)
        target = np.radians([22.0, 91.0, -65.0, 26.0, 52.0, 34.0])
        commanded = []

        def drive_to(sdk, sub_target, tolerance_deg, timeout):
            commanded.append(sub_target.copy())
            return sub_target.copy(), np.zeros(6, dtype=np.float64)

        with (
            patch.object(reset_script, "_joint_feedback", return_value=start),
            patch.object(reset_script, "_drive_to", side_effect=drive_to),
        ):
            final, error = reset_script._move_to_start(
                FakeSDK(),
                target,
                speed=50,
                tolerance_deg=0.5,
                timeout=20.0,
                step_deg=15.0,
            )

        self.assertEqual(len(commanded), 7)
        source = start
        for sub_target in commanded:
            gap = np.degrees(np.abs(sub_target - source)).max()
            self.assertLessEqual(gap, 15.0)
            source = sub_target
        np.testing.assert_allclose(final, target)
        np.testing.assert_allclose(error, np.zeros(6))


if __name__ == "__main__":
    unittest.main(verbosity=2)
