"""Pure-software regression tests for policy adapter contracts."""
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parent.parent)
)

from robokit.deploy.registry import load_registry
from robokit.policies.memvla_lora import _configure_gripper_contract
from robokit.policies.memoryvla import checkpoint_layout


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


class MemoryVLACheckpointLayoutTest(unittest.TestCase):
    """上游 load_vla 断言 <RUN>/checkpoints/x.pt 且 <RUN>/ 下有两个 json；这里提前把话说清楚。"""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.run = Path(self.tempdir.name) / "run"
        (self.run / "checkpoints").mkdir(parents=True)
        self.ckpt = self.run / "checkpoints" / "step-025000.pt"
        self.ckpt.write_bytes(b"x")

    def tearDown(self):
        self.tempdir.cleanup()

    def _metadata(self):
        (self.run / "config.json").write_text("{}")
        (self.run / "dataset_statistics.json").write_text("{}")

    def test_complete_layout_returns_run_dir(self):
        self._metadata()
        checkpoint, run_dir = checkpoint_layout(self.ckpt)
        self.assertEqual(run_dir, self.run)
        self.assertEqual(checkpoint, self.ckpt)

    def test_missing_metadata_names_the_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "config.json"):
            checkpoint_layout(self.ckpt)

    def test_wrong_directory_name_is_rejected(self):
        # zip 里拼错的 checkpints/ 直接解压出来就是这种布局
        bad = self.run / "checkpints"
        bad.mkdir()
        target = bad / "step-025000.pt"
        target.write_bytes(b"x")
        self._metadata()
        with self.assertRaisesRegex(ValueError, "checkpoints"):
            checkpoint_layout(target)


class MemoryVLARegistryEntryTest(unittest.TestCase):
    def test_lamem_v2_is_registered_as_memoryvla_with_policy_args(self):
        spec = load_registry()["lamem-v2"]
        self.assertEqual(spec.family, "memoryvla")
        self.assertEqual(spec.action_space, "eef_delta")
        self.assertEqual(spec.chunk_size, 16)
        self.assertEqual(spec.policy_args["unnorm_key"], "Stack_one_cup_on_top_of_another_cup_b0")
        self.assertTrue(spec.policy_args["codebase"].endswith("MemoryVLA-openvla-codebase"))
        self.assertLessEqual(spec.execution_horizon("sync"), spec.chunk_size)

    def test_pi05_entries_have_no_policy_args(self):
        # pi0/pi05 的适配器不接 policy_args，写了只会被静默忽略；memoryvla 和
        # openvla_oft 则必须靠它拿到 codebase / 底座 / 反归一化键。
        for name, spec in load_registry().items():
            if spec.family in ("pi0", "pi05"):
                self.assertEqual(spec.policy_args, {}, name)

    def test_openvla_oft_entries_carry_required_policy_args(self):
        for name, spec in load_registry().items():
            if spec.family != "openvla_oft":
                continue
            for key in ("base", "codebase", "unnorm_key"):
                self.assertIn(key, spec.policy_args, f"{name} 缺 policy_args.{key}")
            # 基座系与局部系是两套数值不同的契约，OFT 这一版训的是基座系
            self.assertIn(spec.action_space, ("eef_delta", "eef_delta_base"), name)
            self.assertLessEqual(spec.execution_horizon("sync"), spec.chunk_size, name)


if __name__ == "__main__":
    unittest.main(verbosity=2)
