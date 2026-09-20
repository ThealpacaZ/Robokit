#!/usr/bin/env python3
"""统一部署入口的纯软件回归测试：登记表、horizon/max-steps 语义、护栏解除。

不连硬件、不加载权重、不开 socket。覆盖的是重构后最容易悄悄坏掉的四件事：

1. --model 解析出的动作空间/端口/horizon 与登记表一致，越界的 --horizon 被拒。
2. --horizon 与 --max-steps 是两个独立的量：前者是每块执行几步，后者是整场几步；
   max_steps 不为 0 时最后一块必须只执行剩下的步数，不能为了凑满 horizon 越过上限。
3. --safety off 真的把所有拒绝层关掉，且**不**动 IK 分支连续性与硬件限位。
4. 服务端/客户端的模式与动作空间握手仍然 fail closed。
5. 回车中断后的自动复位命令仍然是非交互的，且复位脚本真的认得这些参数。
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from robokit.deploy.loops import LoopConfig, _remaining_steps, _rtc_headroom_advice
from robokit.deploy.obsview import tensor_to_uint8
from robokit.deploy.registry import load_registry, resolve_model
from robokit.deploy.runtime import (
    RESET_SCRIPT,
    build_payload,
    build_reset_command,
    relax_safety,
    reset_to_demo_start,
    resolve_reset_target,
    validate_inference_response,
)
from robokit.utils import load_config


def _load_script(filename):
    """按路径加载 scripts/ 下的入口脚本；它们不是包的一部分，import 时不碰 SDK/权重。"""
    path = ROOT / "scripts" / filename
    spec = importlib.util.spec_from_file_location(f"robokit_entry_{path.stem}", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_reset_script():
    return _load_script(RESET_SCRIPT.name)


class RegistryTest(unittest.TestCase):
    def setUp(self):
        self.registry = load_registry()

    def test_both_pi05_models_are_registered_with_distinct_ports(self):
        self.assertIn("pi05-eef", self.registry)
        self.assertIn("pi05-joint", self.registry)
        eef, joint = self.registry["pi05-eef"], self.registry["pi05-joint"]
        self.assertEqual(eef.action_space, "eef_delta")
        self.assertEqual(joint.action_space, "joint")
        self.assertNotEqual(eef.port, joint.port)

    def test_registry_robot_configs_exist_on_disk(self):
        for spec in self.registry.values():
            self.assertTrue(
                Path(spec.resolved_robot_config()).is_file(),
                f"{spec.name} 指向的机器人配置不存在: {spec.robot_config}",
            )

    def test_rtc_defaults_are_the_paper_values(self):
        spec = self.registry["pi05-eef"]
        self.assertEqual(spec.rtc["delay_buffer_size"], 10)     # 论文 b
        self.assertEqual(spec.rtc["num_inference_steps"], 5)    # 论文 n
        self.assertEqual(spec.rtc["max_guidance_weight"], 5.0)  # 论文 beta
        self.assertEqual(spec.rtc["prefix_attention_schedule"], "EXP")

    def test_horizon_override_is_bounded_by_the_predicted_chunk(self):
        spec = self.registry["pi05-eef"]
        self.assertEqual(spec.execution_horizon("sync"), 30)
        self.assertEqual(spec.execution_horizon("rtc"), 25)
        self.assertEqual(spec.execution_horizon("sync", 50), 50)
        with self.assertRaisesRegex(ValueError, "chunk_size=50"):
            spec.execution_horizon("sync", 51)
        with self.assertRaisesRegex(ValueError, "positive"):
            spec.execution_horizon("sync", 0)
        # RTC 需要 s_min < H，否则没有任何重叠留给推理。
        with self.assertRaisesRegex(ValueError, "s_min"):
            spec.execution_horizon("rtc", 50)

    def test_unknown_model_names_exit_with_the_available_list(self):
        with self.assertRaises(SystemExit) as caught:
            resolve_model("nope")
        self.assertIn("pi05-joint", str(caught.exception))


class MaxStepsTest(unittest.TestCase):
    """--max-steps 是整场总步数，默认 0 = 不限；和 --horizon 无关。"""

    def _config(self, max_steps):
        return LoopConfig(
            host="localhost", port=8080, instruction="stack cups",
            action_space="joint", horizon=15, chunk_size=50,
            control_freq=30, max_steps=max_steps,
        )

    def test_zero_and_negative_mean_unlimited(self):
        for value in (0, -1):
            self.assertEqual(_remaining_steps(self._config(value), 10_000), np.inf)

    def test_last_chunk_executes_only_the_remaining_steps(self):
        config = self._config(40)
        # 15 + 15 = 30 步之后只剩 10 步，最后一块不能执行满 horizon=15。
        self.assertEqual(_remaining_steps(config, 30), 10)
        self.assertEqual(min(config.horizon, _remaining_steps(config, 30)), 10)
        self.assertEqual(_remaining_steps(config, 40), 0)


class RTCHeadroomTest(unittest.TestCase):
    """RTC 下 --horizon 就是 s_min，方向与 sync 相反：越大留给推理的窗口越小。

    2026-07-30 真机实测：H=50、s_min=40、30Hz、往返 297ms(≈9 步) → 窗口只剩 10 步、
    余量 1.1 步，第二次请求就 deadline abort。建议上限是 H-2d=32。
    """

    def test_advice_suggests_a_smaller_horizon_when_the_window_is_tight(self):
        advice = _rtc_headroom_advice(chunk_size=50, s_min=40, delay=9, control_freq=30)
        self.assertIn("H-s_min=10", advice)
        self.assertIn("--horizon <= 32", advice)
        self.assertIn("越大越危险", advice)

    def test_advice_falls_back_to_control_freq_when_no_s_min_works(self):
        # d=30 步时 H=50 放不下任何 s_min（需要 s_min>=30 且 H-s_min>=30）。
        advice = _rtc_headroom_advice(chunk_size=50, s_min=40, delay=30, control_freq=30)
        self.assertIn("无论 s_min 取多少都不成立", advice)
        self.assertIn("--control-freq", advice)
        self.assertNotIn("按 2 倍余量", advice)

    def test_registry_default_s_min_leaves_two_delays_of_headroom(self):
        spec = load_registry()["pi05-joint"]
        s_min = spec.execution_horizon("rtc")
        measured_delay = 9          # 30Hz 下实测 297ms 往返
        self.assertGreaterEqual(
            spec.chunk_size - s_min, 2 * measured_delay,
            "登记表默认的 s_min 必须给实测推理延迟留 2 倍余量",
        )


class SafetyProfileTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config(ROOT / "configs" / "piper_single.yaml")

    def test_protected_config_is_the_production_baseline(self):
        deploy = self.config["deploy"]
        self.assertTrue(deploy["safety"]["enabled"])
        self.assertEqual(deploy["safety"]["max_delta_xyz"], 0.015)
        self.assertEqual(deploy["camera_min_luminance"], 60.0)

    def test_safety_off_disables_every_rejection_layer(self):
        changed = relax_safety(self.config)
        self.assertTrue(changed, "relax_safety 必须回报改动了什么")
        deploy = self.config["deploy"]
        safety = deploy["safety"]
        self.assertFalse(safety["enabled"])
        for key in ("max_delta_xyz", "max_delta_rpy", "max_target_step",
                    "max_tracking_err", "gripper_max_rate"):
            self.assertGreaterEqual(safety[key], 1e9)
            self.assertTrue(np.isfinite(safety[key]), f"{key} 必须是有限值")
        for axis in ("x", "y", "z"):
            low, high = safety["workspace"][axis]
            self.assertLessEqual(low, -1e9)
            self.assertGreaterEqual(high, 1e9)
        self.assertEqual(deploy["gripper_rate"], 0)
        self.assertEqual(deploy["camera_min_luminance"], 0.0)
        self.assertEqual(deploy["camera_max_luminance"], 255.0)
        self.assertGreaterEqual(deploy["camera_timeout_s"], 1e9)
        arm = self.config["robot"]["arms"]["right_arm"]
        self.assertFalse(arm["reset_on_disconnect"])
        self.assertEqual(arm["joint_limit_mode"], "clip")
        self.assertTrue(arm["pinocchio_ik"]["allow_best_effort"])
        self.assertGreaterEqual(arm["pinocchio_ik"]["position_tolerance_mm"], 1e9)

    def test_safety_off_keeps_ik_branch_continuity_and_hardware_limits(self):
        before = dict(self.config["robot"]["arms"]["right_arm"]["pinocchio_ik"])
        limits = [list(row) for row in
                  self.config["robot"]["arms"]["right_arm"]["joint_limits_deg"]]
        relax_safety(self.config)
        after = self.config["robot"]["arms"]["right_arm"]
        # 这两项决定「同一 EEF 位姿的多组关节解里选哪一支」，关掉会让 IK 逐帧乱跳，
        # 臂走出的轨迹反而不是 policy 输出的轨迹。它们不是护栏。
        self.assertEqual(after["pinocchio_ik"]["max_step_deg"], before["max_step_deg"])
        self.assertEqual(
            after["pinocchio_ik"]["continuity_weight"], before["continuity_weight"]
        )
        # 机械行程是硬件事实，不是软件阈值。
        self.assertEqual([list(row) for row in after["joint_limits_deg"]], limits)


class HandshakeTest(unittest.TestCase):
    def _response(self, **overrides):
        base = {
            "action_chunk": np.zeros((50, 7), dtype=np.float32),
            "server_mode": "rtc",
            "action_space": "joint",
            "chunk_id": "chunk-0",
            "prediction_horizon": 50,
        }
        base.update(overrides)
        return base

    def test_mode_and_action_space_mismatches_fail_closed(self):
        response = self._response()
        self.assertIs(validate_inference_response(response, "rtc", "joint"), response)
        with self.assertRaisesRegex(RuntimeError, "sync/RTC port mismatch"):
            validate_inference_response(response, "sync", "joint")
        with self.assertRaisesRegex(RuntimeError, "EEF/joint service mismatch"):
            validate_inference_response(response, "rtc", "eef_delta")

    def test_missing_rtc_keys_are_rejected_before_execution(self):
        response = self._response()
        response.pop("chunk_id")
        with self.assertRaisesRegex(RuntimeError, r"missing \['chunk_id'\]"):
            validate_inference_response(
                response, "rtc", "joint",
                required_keys=("action_chunk", "chunk_id", "prediction_horizon"),
            )

    def test_server_side_errors_surface_as_exceptions(self):
        with self.assertRaisesRegex(RuntimeError, "boom"):
            validate_inference_response({"error": "boom"})


class InterruptResetTest(unittest.TestCase):
    """回车中断后自动复位：参数从哪来、命令长什么样、失败了怎么办。"""

    def setUp(self):
        self.config = load_config(ROOT / "configs" / "piper_single.yaml")
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        # 只换落盘根目录，task_name 与臂的 CAN 口仍取真配置里的值。
        self.config["collect"]["save_path"] = self.tmp.name

    def _make_episode(self, episode, task=None):
        task = task or self.config["collect"]["task_name"]
        directory = os.path.join(self.tmp.name, task)
        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, f"{episode}.hdf5")
        open(path, "wb").close()
        return path

    def test_default_is_the_fixed_start_and_needs_no_dataset(self):
        # 不传 dataset = 用复位脚本里写死的固定起点，本机不需要存任何 HDF5。
        dataset, episode, port = resolve_reset_target(self.config)
        self.assertIsNone(dataset)
        self.assertIsNone(episode)
        # 复位必须打在 policy 刚才控制的那条臂上，而不是写死的 can0。
        self.assertEqual(port, self.config["robot"]["arms"]["right_arm"]["port"])
        # 固定起点的命令里不能出现 --dataset/--episode，否则复位脚本又会去读文件。
        command = build_reset_command(dataset, episode, port)
        self.assertNotIn("--dataset", command)
        self.assertNotIn("--episode", command)

    def test_missing_demo_episode_is_rejected_before_the_run_starts(self):
        # 显式指了数据集就还是老规矩：复位命令写错要在机械臂还停在示教起点时就
        # 报出来，而不是等 policy 跑完、臂停在半空才发现复位跑不起来。
        dataset = os.path.dirname(self._make_episode(0))
        with self.assertRaisesRegex(FileNotFoundError, r"7\.hdf5"):
            resolve_reset_target(self.config, dataset=dataset, episode=7)

    def test_explicit_overrides_win_over_the_config(self):
        other = os.path.dirname(self._make_episode(3, task="other task"))
        dataset, episode, port = resolve_reset_target(
            self.config, dataset=other, episode=3, port="can1"
        )
        self.assertEqual((dataset, episode, port), (other, 3, "can1"))

    def test_command_is_non_interactive_and_never_resets_the_controller(self):
        command = build_reset_command("datasets/stack cups", 0, "can0")
        self.assertEqual(command[1], str(RESET_SCRIPT))
        for flag in ("--execute", "--skip-controller-reset", "--assume-safe"):
            self.assertIn(flag, command)
        # --controller-reset 会阻塞在 input("输入 RESET")，自动流程等不到人回答。
        self.assertNotIn("--controller-reset", command)
        self.assertEqual(command[command.index("--dataset") + 1], "datasets/stack cups")
        self.assertEqual(command[command.index("--episode") + 1], "0")
        self.assertEqual(command[command.index("--port") + 1], "can0")

    def test_reset_script_accepts_every_generated_flag(self):
        """两个脚本各改各的时，这条会先坏掉，而不是等真机中断那一刻才发现。"""
        module = _load_reset_script()
        command = build_reset_command("datasets/stack cups", 4, "can0")
        with patch.object(sys, "argv", ["reset_piper_to_demo_start.py", *command[2:]]):
            args = module._parse_args()
        self.assertTrue(args.execute)
        self.assertTrue(args.skip_controller_reset)
        self.assertTrue(args.assume_safe)
        self.assertFalse(args.controller_reset)
        self.assertEqual((args.dataset, args.episode, args.port),
                         ("datasets/stack cups", 4, "can0"))

    def test_a_failed_reset_reports_instead_of_crashing_the_entry(self):
        # 复位跑不起来只能报告：主流程已经收尾完毕，不能再被它带崩。
        self.assertFalse(reset_to_demo_start([sys.executable, "-c", "raise SystemExit(3)"]))
        self.assertFalse(reset_to_demo_start(["/nonexistent/reset-binary"]))
        self.assertTrue(reset_to_demo_start([sys.executable, "-c", ""]))


class InstructionFlagTest(unittest.TestCase):
    """-L 是改语言指令的开关：三种写法同一个字段，且它真的进了发出去的报文。"""

    def setUp(self):
        self.parser = _load_script("run_policy.py").build_parser()

    def _parse(self, *argv):
        return self.parser.parse_args(["--model", "pi05-joint", *argv])

    def test_all_three_spellings_write_the_same_field(self):
        for flag in ("-L", "--L", "--instruction"):
            self.assertEqual(self._parse(flag, "wipe the table").instruction,
                             "wipe the table")

    def test_L_is_not_swallowed_by_list(self):
        # --L 与 --list 只差大小写；argparse 的前缀匹配区分大小写，且 --L 是精确
        # 匹配，所以两者互不影响。哪天有人加了 --Loop 之类，这条会先坏。
        self.assertTrue(self._parse("--list").list)
        self.assertIsNone(self._parse("--list").instruction)

    def test_default_is_none_so_registry_and_config_can_fill_it(self):
        self.assertIsNone(self._parse().instruction)

    def test_the_instruction_reaches_the_wire_verbatim(self):
        obs = {"cams": {}, "arms": {}}
        self.assertEqual(build_payload(obs, "wipe the table")["instruction"],
                         "wipe the table")

    def test_the_prober_takes_the_same_flag(self):
        """上真机前先用 probe 试新 prompt；两个入口的开关名必须一致。"""
        probe = _load_script("probe_policy_server.py")
        for flag in ("-L", "--L", "--instruction"):
            with patch.object(sys, "argv",
                              ["probe_policy_server.py", "--model", "pi05-joint",
                               flag, "wipe the table"]):
                self.assertEqual(probe.parse_args().instruction, "wipe the table")


class ObsViewTest(unittest.TestCase):
    """看「模型端拿到的图像」时，反解不能把画面弄成一团噪声。"""

    def test_symmetric_normalized_tensor_round_trips(self):
        # CHW, [-1, 1] —— LeRobot 图像预处理最常见的形态。
        gradient = np.linspace(-1.0, 1.0, 16, dtype=np.float32)
        tensor = np.broadcast_to(gradient, (3, 16, 16)).copy()
        image, (low, high) = tensor_to_uint8(tensor[None])
        self.assertEqual(image.shape, (16, 16, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertEqual((int(image.min()), int(image.max())), (0, 255))
        self.assertAlmostEqual(low, -1.0, places=5)
        self.assertAlmostEqual(high, 1.0, places=5)

    def test_unit_range_tensor_is_scaled_not_shifted(self):
        tensor = np.full((3, 4, 4), 0.5, dtype=np.float32)
        image, _ = tensor_to_uint8(tensor)
        self.assertEqual(int(image[0, 0, 0]), 127)

    def test_uint8_hwc_image_passes_through(self):
        tensor = np.full((4, 4, 3), 200, dtype=np.uint8)
        image, _ = tensor_to_uint8(tensor)
        self.assertEqual(int(image[0, 0, 0]), 200)


if __name__ == "__main__":
    os.chdir(ROOT)
    unittest.main()
