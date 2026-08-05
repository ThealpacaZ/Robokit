#!/usr/bin/env python3
"""Convert robokit HDF5 episodes to a PI0/PI0.5-ready LeRobot dataset.

Two action contracts are supported:

``eef_delta`` (the historical PI0.5 dataset):
    observation.state = [eef x/y/z/r/p/y, gripper]
    action            = [local delta x/y/z/r/p/y, next gripper]

``joint`` (the PI0 joint-angle dataset):
    observation.state = [current j1..j6, current gripper]
    action            = [next j1..j6, next gripper]

Joint angles are absolute targets in radians, never deltas.
Only the selected camera is stored. LeRobot encodes it as MP4, so the server
can download the training-ready representation from Hugging Face instead of
downloading raw HDF5 and converting it again.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation


EULER_SEQUENCE = "xyz"
EEF_STATE_NAMES = ["x_m", "y_m", "z_m", "roll_rad", "pitch_rad", "yaw_rad", "gripper"]
EEF_ACTION_NAMES = [
    "local_dx_m",
    "local_dy_m",
    "local_dz_m",
    "local_droll_rad",
    "local_dpitch_rad",
    "local_dyaw_rad",
    "gripper",
]
JOINT_NAMES = ["joint_1_rad", "joint_2_rad", "joint_3_rad", "joint_4_rad", "joint_5_rad", "joint_6_rad"]
JOINT_STATE_NAMES = [*JOINT_NAMES, "gripper"]
JOINT_ACTION_NAMES = [*JOINT_NAMES, "gripper"]


def feature_names(action_space: str) -> tuple[list[str], list[str]]:
    if action_space == "eef_delta":
        return EEF_STATE_NAMES, EEF_ACTION_NAMES
    if action_space == "joint":
        return JOINT_STATE_NAMES, JOINT_ACTION_NAMES
    raise ValueError(f"Unsupported action_space={action_space!r}")


def _natural_key(path: str | Path) -> tuple[int, int | str]:
    stem = Path(path).stem
    return (0, int(stem)) if stem.isdigit() else (1, stem)


def _decode_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def local_delta_pose_batch(eef_pose: np.ndarray) -> np.ndarray:
    """Return local-frame pose deltas for every consecutive EEF pose pair."""
    eef_pose = np.asarray(eef_pose, dtype=np.float64)
    if eef_pose.ndim != 2 or eef_pose.shape[1] != 6 or len(eef_pose) < 2:
        raise ValueError(f"eef_pose must have shape (T>=2, 6), got {eef_pose.shape}")
    rotations = Rotation.from_euler(EULER_SEQUENCE, eef_pose[:, 3:])
    delta_xyz = rotations[:-1].inv().apply(np.diff(eef_pose[:, :3], axis=0))
    delta_rpy = (rotations[:-1].inv() * rotations[1:]).as_euler(EULER_SEQUENCE)
    return np.concatenate([delta_xyz, delta_rpy], axis=1).astype(np.float32)


def require_clean_report(data_dir: Path, files: list[Path], allow_unclean: bool) -> None:
    """Reject data that has not passed the repository's cleaning hard gate."""
    if allow_unclean:
        print("[convert] --allow-unclean: skipping clean_report.json gate")
        return

    report_path = data_dir / "clean_report.json"
    hint = (
        f"Run `python scripts/clean.py --data {data_dir!s} --quarantine` first, "
        "or use --allow-unclean only for an intentional comparison."
    )
    if not report_path.exists():
        raise RuntimeError(f"{report_path} does not exist. {hint}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    episodes = report.get("episodes", report)
    names = [path.name for path in files]
    bad = [name for name in names if episodes.get(name, {}).get("status") == "bad"]
    unchecked = [name for name in names if name not in episodes]
    dataset_errors = report.get("dataset", {}).get("errors", [])

    problems = []
    if bad:
        problems.append(f"{len(bad)} bad episodes remain in source: {bad[:5]}")
    if unchecked:
        problems.append(f"{len(unchecked)} episodes are absent from report: {unchecked[:5]}")
    if dataset_errors:
        problems.append(f"dataset consistency errors: {dataset_errors[:3]}")
    if problems:
        raise RuntimeError("Refusing conversion:\n  - " + "\n  - ".join(problems) + f"\n{hint}")


def inspect_source(
    data_dir: Path,
    camera: str,
    requested_arm: str | None,
    fps_override: int | None,
    allow_unclean: bool,
    action_space: str = "eef_delta",
) -> dict[str, Any]:
    files = sorted((Path(p) for p in glob.glob(str(data_dir / "*.hdf5"))), key=_natural_key)
    if not files:
        raise FileNotFoundError(f"No .hdf5 episodes in {data_dir}")
    require_clean_report(data_dir, files, allow_unclean)

    selected_arm = requested_arm
    image_shape: tuple[int, int, int] | None = None
    fps: int | None = fps_override
    total_frames = 0
    total_source_bytes = 0
    tasks: set[str] = set()
    max_abs_delta = np.zeros(6, dtype=np.float64)
    max_abs_joint_step = np.zeros(6, dtype=np.float64)

    for path in files:
        total_source_bytes += path.stat().st_size
        with h5py.File(path, "r") as episode:
            if episode.attrs.get("version") != "robokit-1.0":
                raise ValueError(f"{path}: expected version=robokit-1.0")
            observations = episode.get("observations")
            if observations is None or "images" not in observations:
                raise ValueError(f"{path}: missing observations/images")

            arms = sorted(key for key in observations.keys() if key != "images")
            if selected_arm is None:
                if len(arms) != 1:
                    raise ValueError(
                        f"{path}: found arms {arms}; pass --arm explicitly (this converter emits one 7-D arm)"
                    )
                selected_arm = arms[0]
            if selected_arm not in arms:
                raise ValueError(f"{path}: arm {selected_arm!r} is absent; available={arms}")
            if camera not in observations["images"]:
                raise ValueError(
                    f"{path}: camera {camera!r} is absent; available={sorted(observations['images'].keys())}"
                )

            state_key = "eef_pose" if action_space == "eef_delta" else "joint"
            state_path = f"observations/{selected_arm}/{state_key}"
            if state_path not in episode:
                raise ValueError(f"{path}: missing {state_path} for action_space={action_space}")
            required = {
                f"observations/images/{camera}": episode[f"observations/images/{camera}"],
                state_path: episode[state_path],
                f"observations/{selected_arm}/gripper": episode[f"observations/{selected_arm}/gripper"],
            }
            lengths = {name: value.shape[0] for name, value in required.items()}
            if len(set(lengths.values())) != 1:
                raise ValueError(f"{path}: inconsistent lengths {lengths}")
            frame_count = next(iter(lengths.values()))
            if frame_count < 2:
                raise ValueError(f"{path}: need at least 2 frames, got {frame_count}")

            this_shape = tuple(required[f"observations/images/{camera}"].shape[1:])
            if len(this_shape) != 3 or this_shape[-1] != 3:
                raise ValueError(f"{path}: expected HWC RGB images, got {this_shape}")
            if image_shape is None:
                image_shape = this_shape
            elif image_shape != this_shape:
                raise ValueError(f"{path}: image shape {this_shape} differs from {image_shape}")

            episode_fps = int(round(float(episode.attrs.get("freq", 0))))
            if fps is None:
                fps = episode_fps
            if episode_fps and fps != episode_fps:
                raise ValueError(f"{path}: fps={episode_fps} differs from selected fps={fps}")

            task = _decode_text(episode.attrs.get("task_name", "")).strip()
            if not task:
                raise ValueError(f"{path}: task_name is empty")
            tasks.add(task)
            if required[state_path].ndim != 2 or required[state_path].shape[1] != 6:
                raise ValueError(f"{path}: expected {state_path} shape (T, 6), got {required[state_path].shape}")
            if action_space == "eef_delta":
                deltas = local_delta_pose_batch(required[state_path][:])
                max_abs_delta = np.maximum(max_abs_delta, np.max(np.abs(deltas), axis=0))
            else:
                joint = np.asarray(required[state_path][:], dtype=np.float64)
                max_abs_joint_step = np.maximum(
                    max_abs_joint_step,
                    np.max(np.abs(np.diff(joint, axis=0)), axis=0),
                )
            total_frames += frame_count - 1

    if fps is None or fps <= 0:
        raise ValueError("Could not infer a positive fps; pass --fps")
    assert selected_arm is not None and image_shape is not None
    return {
        "files": files,
        "arm": selected_arm,
        "camera": camera,
        "image_shape": image_shape,
        "fps": fps,
        "episode_count": len(files),
        "frame_count": total_frames,
        "source_bytes": total_source_bytes,
        "tasks": sorted(tasks),
        "action_space": action_space,
        "max_abs_local_delta": max_abs_delta.tolist(),
        "max_abs_joint_step_rad": max_abs_joint_step.tolist(),
    }


def episode_arrays(
    path: Path,
    arm: str,
    action_space: str = "eef_delta",
) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as episode:
        gripper = np.asarray(episode[f"observations/{arm}/gripper"][:, 0], dtype=np.float32)
        if action_space == "eef_delta":
            eef = np.asarray(episode[f"observations/{arm}/eef_pose"][:], dtype=np.float32)
            states = np.concatenate([eef[:-1], gripper[:-1, None]], axis=1)
            actions = np.concatenate(
                [local_delta_pose_batch(eef), gripper[1:, None]], axis=1
            )
        elif action_space == "joint":
            joint = np.asarray(episode[f"observations/{arm}/joint"][:], dtype=np.float32)
            states = np.concatenate([joint[:-1], gripper[:-1, None]], axis=1)
            actions = np.concatenate([joint[1:], gripper[1:, None]], axis=1)
        else:
            raise ValueError(f"Unsupported action_space={action_space!r}")
    return states, actions


def convert(args: argparse.Namespace, source: dict[str, Any]) -> Path:
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.utils.constants import HF_LEROBOT_HOME
    except ImportError as exc:
        raise RuntimeError(
            "LeRobot is not installed. Install pi0.5/requirements.txt in the conversion environment."
        ) from exc

    root = Path(args.root).expanduser().resolve() if args.root else HF_LEROBOT_HOME / args.repo_id
    if root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{root} exists; choose another --root or pass --overwrite")
        shutil.rmtree(root)

    height, width, channels = source["image_shape"]
    image_key = f"observation.images.{source['camera']}"
    state_names, action_names = feature_names(args.action_space)
    features = {
        image_key: {
            "dtype": "video",
            "shape": (height, width, channels),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (7,),
            "names": state_names,
        },
        "action": {
            "dtype": "float32",
            "shape": (7,),
            "names": action_names,
        },
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=root,
        robot_type="piper",
        fps=source["fps"],
        features=features,
        use_videos=True,
        image_writer_processes=0,
        image_writer_threads=args.image_writer_threads,
        batch_encoding_size=args.batch_encoding_size,
    )

    try:
        for episode_index, path in enumerate(source["files"]):
            with h5py.File(path, "r") as episode:
                task = args.instruction or _decode_text(episode.attrs["task_name"]).strip()
                if args.instruction_suffix:
                    task = f"{task}{args.instruction_suffix}"
                if not task:
                    raise ValueError(f"{path}: task prompt is empty after instruction processing")
                images = episode[f"observations/images/{source['camera']}"]
                states, actions = episode_arrays(path, source["arm"], args.action_space)
                for frame_index in range(len(states)):
                    dataset.add_frame(
                        {
                            image_key: images[frame_index],
                            "observation.state": states[frame_index],
                            "action": actions[frame_index],
                            "task": task,
                        }
                    )
            dataset.save_episode()
            print(
                f"[convert] {episode_index + 1}/{source['episode_count']} "
                f"{path.name}: {len(states)} frames",
                flush=True,
            )
    finally:
        dataset.finalize()

    manifest = {
        "format": "LeRobot v3",
        "source_format": "robokit-1.0 HDF5",
        "repo_id": args.repo_id,
        "camera": source["camera"],
        "arm": source["arm"],
        "fps": source["fps"],
        "episode_count": source["episode_count"],
        "frame_count": source["frame_count"],
        "action_space": args.action_space,
        "action_contract": action_names,
        "state_contract": state_names,
        "euler_sequence": EULER_SEQUENCE if args.action_space == "eef_delta" else None,
        "joint_unit": "rad" if args.action_space == "joint" else None,
        "joint_action_semantics": "next_frame_absolute" if args.action_space == "joint" else None,
        "instruction_override": args.instruction,
        "instruction_suffix": args.instruction_suffix,
        "source_files": [path.name for path in source["files"]],
    }
    (root / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    source_meta = root / "source_meta"
    source_meta.mkdir(exist_ok=True)
    for name in ("clean_report.json", "config.json"):
        source_path = args.data.expanduser().resolve() / name
        if source_path.exists():
            shutil.copy2(source_path, source_meta / name)

    if args.push_to_hub:
        dataset.push_to_hub(
            tags=["piper", "pi0" if args.action_space == "joint" else "pi05", "lerobot"],
            private=args.private,
            push_videos=True,
            upload_large_folder=args.upload_large_folder,
        )
    return root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, type=Path, help="Directory containing robokit HDF5 episodes")
    parser.add_argument("--repo-id", required=True, help="HF dataset repo id, e.g. user/piper-stack-cups-pi05")
    parser.add_argument("--root", help="Local LeRobot output directory (default: HF_LEROBOT_HOME/repo-id)")
    parser.add_argument("--camera", default="cam_high", help="The only image stream to export")
    parser.add_argument("--arm", help="Arm name; auto-detected only when exactly one arm is present")
    parser.add_argument("--fps", type=int, help="Override/verify dataset fps")
    parser.add_argument("--instruction", help="Override task_name for every episode")
    parser.add_argument(
        "--instruction-suffix",
        default="",
        help="Append this exact text to every existing task_name (applied after --instruction, if set)",
    )
    parser.add_argument(
        "--action-space",
        choices=["eef_delta", "joint"],
        default="eef_delta",
        help="eef_delta=local EEF delta; joint=next-frame absolute joint targets in radians",
    )
    parser.add_argument("--image-writer-threads", type=int, default=8)
    parser.add_argument("--batch-encoding-size", type=int, default=1)
    parser.add_argument("--allow-unclean", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--upload-large-folder", action="store_true")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate all HDF5 files and print the resolved contract without importing LeRobot",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    data_dir = args.data.expanduser().resolve()
    source = inspect_source(
        data_dir,
        args.camera,
        args.arm,
        args.fps,
        args.allow_unclean,
        args.action_space,
    )
    printable = {key: value for key, value in source.items() if key != "files"}
    print(json.dumps(printable, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    output = convert(args, source)
    print(f"[convert] complete: {output}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[convert] ERROR: {exc}", file=sys.stderr)
        raise
