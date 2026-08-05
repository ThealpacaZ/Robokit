#!/usr/bin/env python3
"""用现有 HDF5 的 EEF-delta 动作验证生产 Pinocchio IK 下发路径。

本脚本不会连接 CAN、相机或推理服务器，也不会产生真机运动。它把每个 HDF5 episode
当成一个“完美 policy”：

1. 用训练转换相同的 ``local_delta_pose(eef[t], eef[t+1])`` 生成 7 维动作；
2. 按 deploy 的 horizon 分块；
3. 逐块进入真机部署同一个 ``ChunkExecutor``（EEF delta + recursive）；
4. 每一步调用真正的 ``PiperArm.move_eef``，IK 由生产 ``_create_eef_ik`` 创建；
5. 用已下发关节目标的 FK 作为下一步反馈，唯一区别是 CAN 写入由内存 SDK 接收。

因此它专门回答“现有 HDF5 轨迹按 policy 的控制信号路径发送，会不会被本机 IK
abort”，而不是只做数据范围统计或直接回放绝对关节角。
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import re
import sys

import h5py
import numpy as np

sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from robokit.arms.base import ArmCommandRejected
from robokit.arms.piper import PiperArm
from robokit.deploy.runtime import relax_safety
from robokit.executor import ChunkExecutor
from robokit.pose import local_delta_pose
from robokit.utils import load_config


def _natural_key(path: str) -> list[object]:
    return [
        int(part) if part.isdigit() else part
        for part in re.split(r"(\d+)", str(path))
    ]


class _MemoryJointSDK:
    """只接收 PiperArm.move_eef 最终生成的 JointCtrl 整数帧。"""

    def __init__(self, arm: "_OfflinePiperArm") -> None:
        self.arm = arm
        self.commands = 0
        self.limit_violations = 0

    def JointCtrl(self, *joint_millideg: int) -> None:
        raw = np.asarray(joint_millideg, dtype=np.int64)
        if raw.shape != (6,):
            raise AssertionError(f"JointCtrl expected six axes, got {raw.shape}")
        joint = np.radians(raw.astype(np.float64) / 1000.0)
        limits = np.radians(self.arm.joint_limits_deg)
        if np.any(joint < limits[:, 0] - 1e-12) or np.any(
            joint > limits[:, 1] + 1e-12
        ):
            self.limit_violations += 1
            raise AssertionError(
                "PiperArm attempted an out-of-limit JointCtrl: "
                f"{np.degrees(joint).tolist()}"
            )
        self.commands += 1
        self.arm._joint = joint
        self.arm._pose = self.arm._eef_ik.forward_pose(joint)


class _OfflinePiperArm(PiperArm):
    """保留生产 move_eef，只替换硬件反馈、健康检查与 CAN 接收端。"""

    def __init__(
        self,
        name: str,
        cfg: dict,
        initial_joint: np.ndarray,
        initial_pose: np.ndarray,
        initial_gripper: float,
    ) -> None:
        super().__init__(name, cfg)
        if self.eef_backend != "pinocchio_ik":
            raise ValueError(
                "HDF5 policy-path validation requires "
                "eef_backend=pinocchio_ik"
            )
        self.read_only = False
        self._joint = np.asarray(initial_joint, dtype=np.float64).copy()
        self._pose = np.asarray(initial_pose, dtype=np.float64).copy()
        self._gripper = float(initial_gripper)
        # This is the same factory used by PiperArm._connect_writable.  The
        # validator must not maintain a second IK implementation/config parser.
        self._eef_ik = self._create_eef_ik(self._joint)
        self.sdk = _MemoryJointSDK(self)
        self.command_records: list[dict] = []

    def get_state(self) -> dict:
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
        self._gripper = float(np.clip(gripper, 0.0, 1.0))

    def move_eef(self, pose, gripper=None):
        record = super().move_eef(pose, gripper)
        self.command_records.append(record)
        return record


class _OneArmRobot:
    def __init__(self, name: str, arm: _OfflinePiperArm) -> None:
        self.arms = {name: arm}


def _episode_actions(
    path: Path, arm_name: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as episode:
        eef = np.asarray(
            episode[f"observations/{arm_name}/eef_pose"][:],
            dtype=np.float64,
        )
        joint = np.asarray(
            episode[f"observations/{arm_name}/joint"][:],
            dtype=np.float64,
        )
        gripper = np.ravel(
            np.asarray(
                episode[f"observations/{arm_name}/gripper"][:],
                dtype=np.float64,
            )
        )
    if (
        eef.ndim != 2
        or eef.shape[1] != 6
        or joint.shape != eef.shape
        or gripper.shape != (len(eef),)
        or len(eef) < 2
    ):
        raise ValueError(
            f"{path}: expected aligned eef/joint (T,6), gripper (T,), T>=2; "
            f"got {eef.shape}, {joint.shape}, {gripper.shape}"
        )
    deltas = np.stack(
        [local_delta_pose(eef[index], eef[index + 1])
         for index in range(len(eef) - 1)]
    )
    actions = np.concatenate((deltas, gripper[1:, None]), axis=1)
    return eef, joint, gripper, actions


def validate_episode(
    path: Path,
    arm_name: str,
    arm_cfg: dict,
    horizon: int,
) -> dict:
    eef, joint, gripper, actions = _episode_actions(path, arm_name)
    arm = _OfflinePiperArm(
        arm_name,
        arm_cfg,
        joint[0],
        eef[0],
        gripper[0],
    )
    robot = _OneArmRobot(arm_name, arm)
    executor = ChunkExecutor(
        action_space="eef_delta",
        control_freq=30,
        horizon=horizon,
        chunk_base="recursive",
        gripper_mode="raw",
        gripper_rate=None,
        guard=None,
        wait_arrival=None,
        sleep=lambda _duration: None,
        interrupt=lambda: False,
    )

    abort = None
    for start in range(0, len(actions), horizon):
        obs = {"arms": {arm_name: arm.get_state()}}
        status, detail = executor.execute(
            robot,
            actions[start : start + horizon],
            obs,
        )
        if status != "ok":
            abort = {
                "status": status,
                "action_index": start + int(detail.get("row", 0)),
                **detail,
            }
            break

    records = arm.command_records
    if records:
        command_joint_deg = np.asarray(
            [row["joint_target_deg"] for row in records],
            dtype=np.float64,
        )
        # Action i represents HDF5 transition i -> i+1, so the matching
        # measured joint target is observations/joint[i+1].  Compare actuator
        # coordinates directly (no ±360 wrapping): JointCtrl uses these exact
        # mechanical coordinates and the production path is branch-continuous.
        reference_joint_deg = np.degrees(
            joint[1 : 1 + len(records)]
        )
        joint_reference_error_deg = (
            command_joint_deg - reference_joint_deg
        )
        worst_flat = int(
            np.argmax(np.abs(joint_reference_error_deg))
        )
        worst_action, worst_axis = np.unravel_index(
            worst_flat, joint_reference_error_deg.shape
        )
        joint_reference_worst = {
            "episode": str(path),
            "action_index": int(worst_action),
            "axis": int(worst_axis + 1),
            "command_deg": float(
                command_joint_deg[worst_action, worst_axis]
            ),
            "hdf5_deg": float(
                reference_joint_deg[worst_action, worst_axis]
            ),
            "signed_error_deg": float(
                joint_reference_error_deg[worst_action, worst_axis]
            ),
            "abs_error_deg": float(
                abs(joint_reference_error_deg[worst_action, worst_axis])
            ),
        }
    else:
        joint_reference_error_deg = np.empty(
            (0, 6), dtype=np.float64
        )
        joint_reference_worst = None
    return {
        "episode": str(path),
        "actions": len(actions),
        "commands": len(records),
        "jointctrl_commands": arm.sdk.commands,
        "command_limit_violations": arm.sdk.limit_violations,
        "abort": abort,
        "max_position_error_mm": max(
            (float(row["position_error_mm"]) for row in records),
            default=0.0,
        ),
        "max_rotation_error_deg": max(
            (float(row["rotation_error_deg"]) for row in records),
            default=0.0,
        ),
        "max_joint_step_deg": max(
            (
                float(np.max(row["joint_step_deg"]))
                for row in records
            ),
            default=0.0,
        ),
        "max_seed_limit_overrun_deg": max(
            (
                float(np.max(row["seed_limit_overrun_deg"]))
                for row in records
            ),
            default=0.0,
        ),
        "min_command_limit_margin_deg": min(
            (
                float(np.min(row["limit_margin_deg"]))
                for row in records
            ),
            default=0.0,
        ),
        "unconverged_commands": sum(
            not bool(row["converged"]) for row in records
        ),
        "saturated_commands": sum(
            bool(row["saturated"]) for row in records
        ),
        "max_iterations": max(
            (int(row["iterations"]) for row in records),
            default=0,
        ),
        "backend_values": sorted(
            {str(row["backend"]) for row in records}
        ),
        "joint_reference_mean_abs_deg": (
            float(np.mean(np.abs(joint_reference_error_deg)))
            if joint_reference_error_deg.size
            else 0.0
        ),
        "joint_reference_max_abs_deg": (
            float(np.max(np.abs(joint_reference_error_deg)))
            if joint_reference_error_deg.size
            else 0.0
        ),
        "joint_reference_worst": joint_reference_worst,
        # Private aggregation payload; removed before JSON serialization.
        "_joint_reference_error_deg": joint_reference_error_deg,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/piper_single.yaml",
    )
    parser.add_argument(
        "--safety",
        choices=("off", "on"),
        default="off",
        help="off（默认）在内存里解除软件护栏，只看 IK 下发路径本身会不会拒绝；"
             "on 使用配置里的生产阈值",
    )
    parser.add_argument("--data-dir", default="datasets/stack cups")
    parser.add_argument("--arm", default="right_arm")
    parser.add_argument(
        "--horizon",
        type=int,
        default=30,
        help="动作 chunk 执行长度；默认 30，与 PI0.5 sync 真机包装器一致",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="<=0 扫描目录内全部 episode",
    )
    parser.add_argument(
        "--report",
        default="runs/validation/hdf5-policy-path-pinocchio.json",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if args.safety == "off":
        relax_safety(config)
    arm_cfg = dict(config["robot"]["arms"][args.arm])
    horizon = int(args.horizon)
    if horizon <= 0:
        parser.error("horizon must be positive")
    files = [
        Path(path)
        for path in sorted(
            glob.glob(str(Path(args.data_dir) / "*.hdf5")),
            key=_natural_key,
        )
    ]
    if args.max_episodes > 0:
        files = files[: args.max_episodes]
    if not files:
        parser.error(f"no .hdf5 files found in {args.data_dir}")

    results = []
    for index, path in enumerate(files, 1):
        try:
            result = validate_episode(
                path, args.arm, arm_cfg, horizon
            )
        except (
            KeyError,
            ValueError,
            AssertionError,
            ArmCommandRejected,
        ) as exc:
            result = {
                "episode": str(path),
                "actions": 0,
                "commands": 0,
                "abort": {
                    "status": "exception",
                    "kind": type(exc).__name__,
                    "message": str(exc),
                },
            }
        results.append(result)
        state = "PASS" if result["abort"] is None else "ABORT"
        print(
            f"[{index:03d}/{len(files):03d}] {state} {path.name}: "
            f"{result['commands']}/{result['actions']} commands",
            flush=True,
        )

    error_chunks = [
        row.pop("_joint_reference_error_deg")
        for row in results
        if "_joint_reference_error_deg" in row
    ]
    if error_chunks:
        joint_reference_error_deg = np.concatenate(
            error_chunks, axis=0
        )
    else:
        joint_reference_error_deg = np.empty(
            (0, 6), dtype=np.float64
        )
    abs_joint_reference_error_deg = np.abs(
        joint_reference_error_deg
    )
    worst_candidates = [
        row["joint_reference_worst"]
        for row in results
        if row.get("joint_reference_worst") is not None
    ]
    worst_joint_reference = (
        max(worst_candidates, key=lambda row: row["abs_error_deg"])
        if worst_candidates
        else None
    )
    axis_names = [f"j{index}" for index in range(1, 7)]
    if joint_reference_error_deg.size:
        per_axis_joint_reference = {
            name: {
                "mean_signed_deg": float(
                    np.mean(joint_reference_error_deg[:, index])
                ),
                "mean_abs_deg": float(
                    np.mean(
                        abs_joint_reference_error_deg[:, index]
                    )
                ),
                "rmse_deg": float(
                    np.sqrt(
                        np.mean(
                            np.square(
                                joint_reference_error_deg[:, index]
                            )
                        )
                    )
                ),
                "p95_abs_deg": float(
                    np.percentile(
                        abs_joint_reference_error_deg[:, index], 95
                    )
                ),
                "p99_abs_deg": float(
                    np.percentile(
                        abs_joint_reference_error_deg[:, index], 99
                    )
                ),
                "max_abs_deg": float(
                    np.max(abs_joint_reference_error_deg[:, index])
                ),
            }
            for index, name in enumerate(axis_names)
        }
        joint_reference_comparison = {
            "samples": int(len(joint_reference_error_deg)),
            "definition": (
                "Pinocchio JointCtrl[action i] - "
                "HDF5 observations/joint[i+1], degrees, no angle wrapping"
            ),
            "mean_signed_deg": float(
                np.mean(joint_reference_error_deg)
            ),
            "mean_abs_deg": float(
                np.mean(abs_joint_reference_error_deg)
            ),
            "rmse_deg": float(
                np.sqrt(
                    np.mean(np.square(joint_reference_error_deg))
                )
            ),
            "p50_abs_deg": float(
                np.percentile(abs_joint_reference_error_deg, 50)
            ),
            "p95_abs_deg": float(
                np.percentile(abs_joint_reference_error_deg, 95)
            ),
            "p99_abs_deg": float(
                np.percentile(abs_joint_reference_error_deg, 99)
            ),
            "max_abs_deg": float(
                np.max(abs_joint_reference_error_deg)
            ),
            "per_axis": per_axis_joint_reference,
            "worst": worst_joint_reference,
        }
    else:
        joint_reference_comparison = {
            "samples": 0,
            "definition": (
                "Pinocchio JointCtrl[action i] - "
                "HDF5 observations/joint[i+1], degrees, no angle wrapping"
            ),
            "per_axis": {},
            "worst": None,
        }

    failures = [row for row in results if row["abort"] is not None]
    actions = sum(row["actions"] for row in results)
    commands = sum(row["commands"] for row in results)
    jointctrl_commands = sum(
        row.get("jointctrl_commands", 0) for row in results
    )
    limit_violations = sum(
        row.get("command_limit_violations", 0) for row in results
    )
    max_joint_step_deg = max(
        (row.get("max_joint_step_deg", 0.0) for row in results),
        default=0.0,
    )
    acceptance_failures = []
    if commands != actions or jointctrl_commands != actions:
        acceptance_failures.append(
            f"commands={commands}, JointCtrl={jointctrl_commands}, "
            f"actions={actions}"
        )
    if limit_violations:
        acceptance_failures.append(
            f"JointCtrl limit violations={limit_violations}"
        )
    if max_joint_step_deg > 5.001 + 1e-9:
        acceptance_failures.append(
            f"max joint step={max_joint_step_deg:.6f}° > 5.001°"
        )
    summary = {
        "config": args.config,
        "data_dir": args.data_dir,
        "arm": args.arm,
        "action_space": "eef_delta",
        "chunk_base": "recursive",
        "horizon": horizon,
        "guard": False,
        "wait_arrival": False,
        "episodes": len(results),
        "backend": "pinocchio_ik_move_j",
        "actions": actions,
        "commands": commands,
        "jointctrl_commands": jointctrl_commands,
        "command_limit_violations": limit_violations,
        "aborts": len(failures),
        "max_position_error_mm": max(
            (row.get("max_position_error_mm", 0.0) for row in results),
            default=0.0,
        ),
        "max_rotation_error_deg": max(
            (row.get("max_rotation_error_deg", 0.0) for row in results),
            default=0.0,
        ),
        "max_joint_step_deg": max_joint_step_deg,
        "max_seed_limit_overrun_deg": max(
            (
                row.get("max_seed_limit_overrun_deg", 0.0)
                for row in results
            ),
            default=0.0,
        ),
        "min_command_limit_margin_deg": min(
            (
                row.get("min_command_limit_margin_deg", 0.0)
                for row in results
            ),
            default=0.0,
        ),
        "failures": failures,
        "acceptance_failures": acceptance_failures,
        "unconverged_commands": sum(
            row.get("unconverged_commands", 0) for row in results
        ),
        "saturated_commands": sum(
            row.get("saturated_commands", 0) for row in results
        ),
        "max_iterations": max(
            (row.get("max_iterations", 0) for row in results),
            default=0,
        ),
        "joint_reference_comparison": joint_reference_comparison,
        "result": (
            "PASS"
            if not failures and not acceptance_failures
            else "FAIL"
        ),
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"report: {report_path}")
    return 0 if summary["result"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
