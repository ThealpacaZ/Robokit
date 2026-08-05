"""Pure-software regression tests for policy adapter contracts."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)

from robokit.policies.memvla_lora import _configure_gripper_contract


class MemVLALoRAGripperContractTest(unittest.TestCase):
    def _codebase(self, source):
        root = Path(self.tempdir.name)
        path = root / "vla" / "memory_vla.py"
        path.parent.mkdir(parents=True)
        path.write_text(source, encoding="utf-8")
        return root

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.previous = os.environ.get("MEMVLA_BINARIZE_GRIPPER")

    def tearDown(self):
        self.tempdir.cleanup()
        if self.previous is None:
            os.environ.pop("MEMVLA_BINARIZE_GRIPPER", None)
        else:
            os.environ["MEMVLA_BINARIZE_GRIPPER"] = self.previous

    def test_continuous_gripper_sets_zero_and_requires_source_gate(self):
        root = self._codebase(
            'flag = os.environ.get("MEMVLA_BINARIZE_GRIPPER", "1")\n'
        )
        self.assertFalse(_configure_gripper_contract(root, False))
        self.assertEqual(os.environ["MEMVLA_BINARIZE_GRIPPER"], "0")

    def test_continuous_gripper_rejects_unpatched_upstream(self):
        root = self._codebase("normalized_actions[:, 6] = 0\n")
        with self.assertRaisesRegex(RuntimeError, "门控补丁"):
            _configure_gripper_contract(root, False)

    def test_explicit_binary_mode_sets_one_without_patch_requirement(self):
        root = self._codebase("unpatched = True\n")
        self.assertTrue(_configure_gripper_contract(root, True))
        self.assertEqual(os.environ["MEMVLA_BINARIZE_GRIPPER"], "1")


if __name__ == "__main__":
    unittest.main(verbosity=2)
