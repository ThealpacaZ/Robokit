#!/usr/bin/env python
"""Piper controller-reset 生命周期纯软件自检；不会导入 SDK 或连接 CAN。"""

import os
import sys
import unittest
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from robokit.arms.piper_lifecycle import (  # noqa: E402
    ControllerResetError,
    ControllerResetGuard,
    controller_reset_and_verify,
)


class FakeTrace:
    def __init__(self):
        self.records = []

    def write(self, record):
        self.records.append(record)


class FakeSDK:
    """只实现生命周期模块读取的 SDK 表面。"""

    def __init__(
        self,
        *,
        post_arm_status=0,
        post_err_code=0,
        post_enabled=False,
        stale=False,
        reset_error=None,
        feedback_error=False,
        precheck_error=False,
    ):
        self.post_arm_status = post_arm_status
        self.post_err_code = post_err_code
        self.post_enabled = post_enabled
        self.stale = stale
        self.reset_error = reset_error
        self.feedback_error = feedback_error
        self.precheck_error = precheck_error
        self.reset_calls = 0
        self.reset_started = False
        self.status_reads = 0
        self.low_reads = 0

    def ResetPiper(self):
        self.reset_calls += 1
        self.reset_started = True
        if self.reset_error is not None:
            raise self.reset_error

    def GetArmStatus(self):
        if not self.reset_started and self.precheck_error:
            raise RuntimeError("pre-reset status feedback failed")
        if self.reset_started and self.feedback_error:
            raise RuntimeError("status feedback failed")
        if self.reset_started:
            self.status_reads += 1
        timestamp = (
            1.0
            if not self.reset_started or self.stale
            else 1.0 + self.status_reads
        )
        status = SimpleNamespace(
            arm_status=(
                self.post_arm_status if self.reset_started else 0
            ),
            err_code=self.post_err_code if self.reset_started else 0,
            ctrl_mode=1,
            mode_feed=1,
            motion_status=0,
        )
        return SimpleNamespace(time_stamp=timestamp, arm_status=status)

    def GetArmLowSpdInfoMsgs(self):
        if self.reset_started:
            self.low_reads += 1
        timestamp = (
            1.0
            if not self.reset_started or self.stale
            else 1.0 + self.low_reads
        )
        enabled = (
            self.post_enabled if self.reset_started else True
        )
        fields = {"time_stamp": timestamp}
        for index in range(1, 7):
            fields[f"motor_{index}"] = SimpleNamespace(
                foc_status=SimpleNamespace(
                    driver_enable_status=enabled
                )
            )
        return SimpleNamespace(**fields)


class PiperLifecycleTest(unittest.TestCase):
    def test_normal_reset_is_fresh_disabled_and_traced(self):
        sdk = FakeSDK()
        trace = FakeTrace()

        record = controller_reset_and_verify(
            sdk, trace=trace, timeout=0.1, stable_s=0.005
        )

        self.assertEqual(sdk.reset_calls, 1)
        self.assertTrue(record["attempted"])
        self.assertTrue(record["verified"])
        self.assertEqual(record["failure"], None)
        self.assertGreater(
            record["post"]["status_timestamp"],
            record["pre"]["status_timestamp"],
        )
        self.assertGreater(
            record["post"]["low_timestamp"],
            record["pre"]["low_timestamp"],
        )
        self.assertEqual(record["post"]["enabled"], [False] * 6)
        self.assertEqual(trace.records, [record])

    def test_abnormal_feedback_is_structured_and_traced(self):
        sdk = FakeSDK(feedback_error=True)
        trace = FakeTrace()

        with self.assertRaises(ControllerResetError) as caught:
            controller_reset_and_verify(
                sdk, trace=trace, timeout=0.05, stable_s=0.005
            )

        record = caught.exception.record
        self.assertEqual(sdk.reset_calls, 1)
        self.assertEqual(record["failure"], "feedback_error")
        self.assertFalse(record["verified"])
        self.assertEqual(trace.records, [record])

    def test_precheck_failure_still_attempts_and_verifies_reset(self):
        sdk = FakeSDK(precheck_error=True)
        trace = FakeTrace()

        record = controller_reset_and_verify(
            sdk, trace=trace, timeout=0.1, stable_s=0.005
        )

        self.assertEqual(sdk.reset_calls, 1)
        self.assertTrue(record["attempted"])
        self.assertTrue(record["verified"])
        self.assertEqual(record["pre"], None)
        self.assertEqual(
            record["precheck_error"]["error_type"], "RuntimeError"
        )
        self.assertEqual(trace.records, [record])

    def test_timeout_never_mistakes_stale_cache_for_reset(self):
        sdk = FakeSDK(stale=True)
        trace = FakeTrace()

        with self.assertRaises(ControllerResetError) as caught:
            controller_reset_and_verify(
                sdk, trace=trace, timeout=0.025, stable_s=0.005
            )

        record = caught.exception.record
        self.assertEqual(sdk.reset_calls, 1)
        self.assertEqual(record["failure"], "timeout")
        self.assertFalse(record["verified"])
        self.assertEqual(trace.records, [record])

    def test_nonzero_controller_state_times_out(self):
        sdk = FakeSDK(post_arm_status=5, post_err_code=63)

        with self.assertRaises(ControllerResetError) as caught:
            controller_reset_and_verify(
                sdk, timeout=0.025, stable_s=0.005
            )

        self.assertEqual(caught.exception.record["failure"], "timeout")
        self.assertEqual(
            caught.exception.record["post"]["arm_status"], 5
        )
        self.assertEqual(sdk.reset_calls, 1)

    def test_resetpiper_exception_is_not_retried(self):
        sdk = FakeSDK(reset_error=RuntimeError("CAN send failed"))
        trace = FakeTrace()

        with self.assertRaises(ControllerResetError) as caught:
            controller_reset_and_verify(
                sdk, trace=trace, timeout=0.05, stable_s=0.005
            )

        record = caught.exception.record
        self.assertEqual(sdk.reset_calls, 1)
        self.assertEqual(record["failure"], "reset_error")
        self.assertEqual(record["error_type"], "RuntimeError")
        self.assertEqual(trace.records, [record])

    def test_guard_is_exactly_once_and_does_not_disconnect(self):
        sdk = FakeSDK()
        guard = ControllerResetGuard(
            sdk, timeout=0.1, stable_s=0.005
        )
        guard.arm()

        first = guard.close()
        second = guard.close()

        self.assertIs(first, second)
        self.assertEqual(sdk.reset_calls, 1)
        self.assertTrue(guard.attempted)
        self.assertFalse(hasattr(sdk, "DisconnectPort"))

    def test_unarmed_context_sends_nothing(self):
        sdk = FakeSDK()

        with ControllerResetGuard(
            sdk, timeout=0.1, stable_s=0.005
        ):
            pass

        self.assertEqual(sdk.reset_calls, 0)

    def test_context_preserves_primary_exception_if_reset_fails(self):
        sdk = FakeSDK(reset_error=RuntimeError("send failed"))
        guard = ControllerResetGuard(
            sdk, timeout=0.05, stable_s=0.005
        )

        with self.assertRaisesRegex(ValueError, "primary") as caught:
            with guard:
                guard.arm()
                raise ValueError("primary")

        self.assertEqual(sdk.reset_calls, 1)
        notes = getattr(caught.exception, "__notes__", [])
        self.assertTrue(
            any("controller reset cleanup" in note for note in notes)
        )

        # 再关闭只复用首次失败，不会再次调用 ResetPiper。
        with self.assertRaises(ControllerResetError):
            guard.close()
        self.assertEqual(sdk.reset_calls, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
