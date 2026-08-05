#!/usr/bin/env python3
"""Validate PI0/PI0.5 RTC scheduling through a no-CAN Piper digital twin.

Each HDF5 episode acts as a deterministic perfect policy. New 50-step chunks
start at the observation used to launch inference; the old chunk keeps running
for injected latency steps, then Algorithm 1 skips those elapsed rows. Actions
go through the same RTC controller and production Piper command paths used by
deployment. No socket, camera, CAN interface, or physical arm is opened.

This validates action alignment, deadline handling, chunk-boundary continuity,
joint clipping/quantization, EEF ActionGuard, and Pinocchio IK. It does not
claim that a learned checkpoint's guided samples are task-successful; that
requires the GPU-in-the-loop dry-run described in RTC_DEPLOYMENT.md.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

# 数字孪生用的是生产 RTC 运行时本身，不是它的副本：控制器和执行器都直接从
# robokit.deploy.rtc 导入，真机跑的就是这两个类。
from robokit.deploy.rtc import RTCActionExecutor, RTCController
from robokit.safety import ActionGuard
from robokit.utils import load_config
from validate_hdf5_policy_path import (
    _OfflinePiperArm,
    _OneArmRobot,
    _episode_actions,
    _natural_key,
)


def _chunk(
    actions: np.ndarray,
    start: int,
    horizon: int,
    action_space: str,
) -> np.ndarray:
    """Return a fixed H chunk without inventing motion past episode end."""
    result = np.empty((horizon, actions.shape[1]), dtype=np.float32)
    available = actions[start : start + horizon]
    result[: len(available)] = available
    if len(available) < horizon:
        if len(available):
            pad_gripper = float(available[-1, -1])
        else:
            pad_gripper = float(actions[-1, -1])
        if action_space == "eef_delta":
            result[len(available) :, :] = 0.0
            result[len(available) :, -1] = pad_gripper
        else:
            last = available[-1] if len(available) else actions[-1]
            result[len(available) :, :] = last
    return result


def _parse_delays(value: str) -> list[int]:
    try:
        delays = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--delays must be comma-separated integer controller steps"
        ) from exc
    if not delays or any(item < 0 for item in delays):
        raise argparse.ArgumentTypeError("--delays must contain non-negative steps")
    return delays


def _parse_episode_indices(value: str) -> list[int]:
    try:
        indices = [
            int(item.strip()) for item in value.split(",") if item.strip()
        ]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "--episodes must be comma-separated numeric HDF5 stems"
        ) from exc
    if not indices or any(item < 0 for item in indices):
        raise argparse.ArgumentTypeError(
            "--episodes must contain non-negative indices"
        )
    return indices


def validate_episode(
    path: Path,
    *,
    action_space: str,
    config: dict[str, Any],
    arm_name: str,
    horizon: int,
    s_min: int,
    delays: list[int],
    max_actions: int,
) -> dict[str, Any]:
    eef, joint, gripper, eef_actions = _episode_actions(path, arm_name)
    actions = (
        np.concatenate((joint[1:], gripper[1:, None]), axis=1)
        if action_space == "joint"
        else eef_actions
    )
    action_count = min(len(actions), max_actions) if max_actions > 0 else len(actions)
    actions = np.asarray(actions[:action_count], dtype=np.float64)
    arm_cfg = dict(config["robot"]["arms"][arm_name])
    arm = _OfflinePiperArm(
        arm_name, arm_cfg, joint[0], eef[0], gripper[0]
    )
    robot = _OneArmRobot(arm_name, arm)
    deploy_cfg = config["deploy"]
    guard = (
        ActionGuard.from_config(deploy_cfg)
        if action_space == "eef_delta"
        else None
    )
    executor = RTCActionExecutor(
        action_space=action_space,
        control_freq=float(deploy_cfg.get("control_freq", 30)),
        fixed_control_rate=False,
        # RTC's guided prefix is the chunk-boundary continuity mechanism.
        # Re-anchoring every swap would integrate tiny IK/FK residuals.
        chunk_base="continuous",
        gripper_mode="raw",
        gripper_rate=(
            deploy_cfg.get("gripper_rate")
            if deploy_cfg.get("gripper_rate", 0) > 0
            else None
        ),
        guard=guard,
        wait_arrival=None,
        sleep=lambda _duration: None,
        interrupt=lambda: False,
    )
    initial_delay = max(delays)
    controller = RTCController(
        prediction_horizon=horizon,
        min_execution_horizon=s_min,
        initial_delay=initial_delay,
        delay_buffer_size=10,
    )
    controller.initialize(
        _chunk(actions, 0, horizon, action_space), "chunk-0000"
    )
    executor.begin_chunk({"arms": {arm_name: arm.get_state()}})

    global_step = 0
    swaps = 0
    predicted_delays: list[int] = []
    observed_delays: list[int] = []
    max_action_alignment_error = 0.0
    abort = None

    def execute_one() -> bool:
        nonlocal global_step, max_action_alignment_error, abort
        action, row, chunk_id = controller.next_action()
        error = float(np.max(np.abs(action.astype(np.float64) - actions[global_step])))
        max_action_alignment_error = max(max_action_alignment_error, error)
        status, detail = executor.execute_action(robot, action)
        if status != "ok":
            abort = {
                "global_step": global_step,
                "chunk_id": chunk_id,
                "row": row,
                "status": status,
                **detail,
            }
            return False
        global_step += 1
        return True

    while global_step < len(actions) and abort is None:
        if not controller.should_start_inference():
            execute_one()
            continue

        request = controller.start_inference()
        predicted_delays.append(request.inference_delay)
        replacement = _chunk(
            actions, global_step, horizon, action_space
        )
        actual_delay = delays[swaps % len(delays)]
        remaining_old = horizon - request.executed_at_start
        if actual_delay > remaining_old:
            abort = {
                "status": "deadline",
                "global_step": global_step,
                "message": (
                    f"injected delay {actual_delay} exceeds old-chunk "
                    f"remainder {remaining_old}"
                ),
            }
            break
        for _ in range(min(actual_delay, len(actions) - global_step)):
            if not execute_one():
                break
        if abort is not None or global_step >= len(actions):
            break
        observed = controller.accept_inference(
            replacement, f"chunk-{swaps + 1:04d}"
        )
        observed_delays.append(observed)
        executor.begin_chunk({"arms": {arm_name: arm.get_state()}})
        swaps += 1

    state = arm.get_state()
    result: dict[str, Any] = {
        "episode": str(path),
        "action_space": action_space,
        "actions_requested": len(actions),
        "actions_executed": global_step,
        "swaps": swaps,
        "predicted_delays": predicted_delays,
        "observed_delays": observed_delays,
        "max_action_alignment_error": max_action_alignment_error,
        "abort": abort,
        "jointctrl_commands": arm.sdk.commands,
        "joint_limit_violations": arm.sdk.limit_violations,
    }
    if global_step:
        result["endpoint_joint_error_deg"] = float(
            np.max(
                np.abs(
                    np.degrees(
                        np.asarray(state["joint"]) - joint[global_step]
                    )
                )
            )
        )
        result["endpoint_position_error_mm"] = float(
            np.linalg.norm(
                np.asarray(state["eef_pose"])[:3] - eef[global_step, :3]
            )
            * 1000.0
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/stack cups")
    parser.add_argument("--arm", default="right_arm")
    parser.add_argument(
        "--episodes",
        type=_parse_episode_indices,
        help="exact numeric HDF5 stems, for example 2,5,7",
    )
    parser.add_argument(
        "--episode-offset",
        type=int,
        default=0,
        help="skip this many naturally sorted episodes before --max-episodes",
    )
    parser.add_argument("--max-episodes", type=int, default=3)
    parser.add_argument("--max-actions", type=int, default=180)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--pi0-s-min", type=int, default=15)
    parser.add_argument("--pi05-s-min", type=int, default=25)
    parser.add_argument("--delays", type=_parse_delays, default=[3, 5, 7])
    parser.add_argument(
        "--report",
        default="runs/validation/rtc-policy-path-digital-twin.json",
    )
    args = parser.parse_args()

    paths = sorted(
        (Path(path) for path in glob.glob(str(Path(args.data_dir) / "*.hdf5"))),
        key=_natural_key,
    )
    if args.episodes:
        by_index = {int(path.stem): path for path in paths}
        missing = [index for index in args.episodes if index not in by_index]
        if missing:
            parser.error(f"episodes not found under {args.data_dir}: {missing}")
        paths = [by_index[index] for index in args.episodes]
    else:
        if args.episode_offset < 0:
            parser.error("--episode-offset must be non-negative")
        paths = paths[args.episode_offset :]
        if args.max_episodes > 0:
            paths = paths[: args.max_episodes]
    if not paths:
        raise FileNotFoundError(f"no HDF5 episodes under {args.data_dir}")
    if args.horizon <= 1:
        parser.error("--horizon must be > 1")

    configs = {
        "joint": load_config("configs/piper_single_joint.yaml"),
        "eef_delta": load_config("configs/piper_single.yaml"),
    }
    s_min = {"joint": args.pi0_s_min, "eef_delta": args.pi05_s_min}
    max_delay = max(args.delays)
    for action_space in configs:
        limit = min(s_min[action_space], args.horizon - s_min[action_space])
        if max_delay > limit:
            parser.error(
                f"max delay {max_delay} violates d <= s_min <= H-d for "
                f"{action_space}: H={args.horizon}, s_min={s_min[action_space]}"
            )

    results = []
    for action_space in ("joint", "eef_delta"):
        for path in paths:
            row = validate_episode(
                path,
                action_space=action_space,
                config=configs[action_space],
                arm_name=args.arm,
                horizon=args.horizon,
                s_min=s_min[action_space],
                delays=args.delays,
                max_actions=args.max_actions,
            )
            results.append(row)
            print(
                f"[rtc-twin] {action_space:9s} {path.name}: "
                f"{row['actions_executed']}/{row['actions_requested']} actions, "
                f"swaps={row['swaps']}, abort={row['abort']}"
            )

    failures = [
        row
        for row in results
        if row["abort"] is not None
        or row["actions_executed"] != row["actions_requested"]
        or row["max_action_alignment_error"] > 1e-6
        or row["joint_limit_violations"] != 0
    ]
    report = {
        "schema": "robokit.rtc-digital-twin.v1",
        "official_reference": (
            "https://github.com/Physical-Intelligence/"
            "real-time-chunking-kinetix"
        ),
        "scope": (
            "deterministic RTC scheduler + production Piper command path; "
            "no learned-model task-success claim"
        ),
        "settings": {
            "prediction_horizon": args.horizon,
            "pi0_min_execution_horizon": args.pi0_s_min,
            "pi05_min_execution_horizon": args.pi05_s_min,
            "injected_delay_steps": args.delays,
            "max_episodes": args.max_episodes,
            "episode_offset": args.episode_offset,
            "episodes": args.episodes,
            "max_actions_per_episode": args.max_actions,
        },
        "accepted": not failures,
        "failures": failures,
        "results": results,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        f"[rtc-twin] accepted={report['accepted']} "
        f"cases={len(results)} report={report_path}"
    )
    return 0 if report["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
