import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from robokit.arms.piper import PiperArm
from robokit.robot import Robot


class FakeDevice:
    def __init__(self, *, connect_error=None, disconnect_error=None):
        self.connect_error = connect_error
        self.disconnect_error = disconnect_error
        self.connect_calls = []
        self.disconnect_calls = 0

    def connect(self, **kwargs):
        self.connect_calls.append(kwargs)
        if self.connect_error is not None:
            raise self.connect_error

    def disconnect(self):
        self.disconnect_calls += 1
        if self.disconnect_error is not None:
            raise self.disconnect_error


class RobotConnectRollbackTest(unittest.TestCase):
    def test_arm_connect_failure_disconnects_camera_and_arm(self):
        original = RuntimeError("no CAN feedback")
        camera = FakeDevice()
        arm = FakeDevice(connect_error=original)
        robot = object.__new__(Robot)
        robot.cameras = {"cam": camera}
        robot.arms = {"arm": arm}

        with self.assertRaisesRegex(RuntimeError, "no CAN feedback"):
            robot.connect(read_only=True)

        self.assertEqual(camera.disconnect_calls, 1)
        self.assertEqual(arm.disconnect_calls, 1)

    def test_cleanup_error_does_not_hide_connect_error(self):
        camera = FakeDevice(disconnect_error=RuntimeError("camera stop failed"))
        arm = FakeDevice(connect_error=ValueError("arm connect failed"))
        robot = object.__new__(Robot)
        robot.cameras = {"cam": camera}
        robot.arms = {"arm": arm}

        with self.assertRaisesRegex(ValueError, "arm connect failed") as caught:
            robot.connect(read_only=True)

        self.assertTrue(
            any("camera stop failed" in note for note in getattr(caught.exception, "__notes__", []))
        )


class FakePiperSDK:
    def __init__(self):
        self.connect_calls = []
        self.disconnect_calls = 0

    def ConnectPort(self, **kwargs):
        self.connect_calls.append(kwargs)

    def DisconnectPort(self):
        self.disconnect_calls += 1


class PiperReadOnlyRollbackTest(unittest.TestCase):
    def test_feedback_timeout_disconnects_sdk(self):
        sdk = FakePiperSDK()
        module = SimpleNamespace(C_PiperInterface_V2=lambda _port: sdk)
        arm = PiperArm("right_arm", {"dof": 6, "port": "can0"})
        arm._wait_feedback = Mock(side_effect=RuntimeError("feedback timeout"))

        with patch.dict(sys.modules, {"piper_sdk": module}), patch("time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "feedback timeout"):
                arm.connect(read_only=True)

        self.assertEqual(sdk.connect_calls, [{"piper_init": False}])
        self.assertEqual(sdk.disconnect_calls, 1)
        self.assertIsNone(arm.sdk)


if __name__ == "__main__":
    unittest.main()
