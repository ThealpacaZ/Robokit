#!/usr/bin/env python3
"""Validate a PI0/PI0.5 robokit data contract in a local or Hub LeRobot dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


EEF_STATE = ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad", "gripper"]
EEF_ACTION = [
    "local_dx_m",
    "local_dy_m",
    "local_dz_m",
    "local_droll_rad",
    "local_dpitch_rad",
    "local_dyaw_rad",
    "gripper",
]
JOINT_STATE = [
    "joint_1_rad",
    "joint_2_rad",
    "joint_3_rad",
    "joint_4_rad",
    "joint_5_rad",
    "joint_6_rad",
    "gripper",
]
JOINT_ACTION = list(JOINT_STATE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--root", type=Path, help="Validate this local dataset directory")
    parser.add_argument("--revision")
    parser.add_argument("--sample-count", type=int, default=8)
    parser.add_argument(
        "--action-space",
        choices=["eef_delta", "joint"],
        default="eef_delta",
    )
    parser.add_argument(
        "--video-backend",
        default="pyav",
        help="Video decoder. pyav avoids a system FFmpeg dependency on training servers.",
    )
    args = parser.parse_args()

    from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata

    root = args.root.expanduser().resolve() if args.root else None
    meta = LeRobotDatasetMetadata(args.repo_id, root=root, revision=args.revision)
    features = meta.features
    image_keys = sorted(
        key for key, feature in features.items() if feature["dtype"] in {"image", "video"}
    )
    if len(image_keys) != 1:
        raise RuntimeError(f"Expected exactly one real image feature, got {image_keys}")
    expected_state, expected_action = (
        (EEF_STATE, EEF_ACTION) if args.action_space == "eef_delta" else (JOINT_STATE, JOINT_ACTION)
    )
    for key, names in (
        ("observation.state", expected_state),
        ("action", expected_action),
    ):
        if key not in features:
            raise RuntimeError(f"Missing feature {key}")
        if tuple(features[key]["shape"]) != (7,):
            raise RuntimeError(f"{key} shape is {features[key]['shape']}, expected (7,)")
        if list(features[key].get("names") or []) != names:
            raise RuntimeError(f"{key} names are {features[key].get('names')}, expected {names}")
        stats = meta.stats.get(key, {})
        missing_stats = {"q01", "q50", "q99"} - set(stats)
        if missing_stats:
            raise RuntimeError(f"{key} is missing quantile stats: {sorted(missing_stats)}")

    dataset = LeRobotDataset(
        args.repo_id,
        root=root,
        revision=args.revision,
        download_videos=True,
        video_backend=args.video_backend,
    )
    if len(dataset) == 0:
        raise RuntimeError("Dataset has zero frames")
    indices = np.linspace(0, len(dataset) - 1, min(args.sample_count, len(dataset)), dtype=int)
    for index in indices:
        item = dataset[int(index)]
        for key in ("observation.state", "action"):
            value = np.asarray(item[key])
            if value.shape != (7,) or not np.isfinite(value).all():
                raise RuntimeError(f"sample {index} {key}: shape={value.shape}, finite={np.isfinite(value).all()}")

    result = {
        "repo_id": args.repo_id,
        "root": str(meta.root),
        "episodes": meta.total_episodes,
        "frames": meta.total_frames,
        "fps": meta.fps,
        "image_keys": image_keys,
        "action_space": args.action_space,
        "state": expected_state,
        "action": expected_action,
        "sampled_indices": indices.tolist(),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
