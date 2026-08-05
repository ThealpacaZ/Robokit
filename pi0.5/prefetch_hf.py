#!/usr/bin/env python3
"""Prefetch PI0.5 dataset/base/adapter snapshots into the server's HF cache."""

from __future__ import annotations

import argparse
import json
import os

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="LeRobot dataset repo id")
    parser.add_argument("--base", default="lerobot/pi05_base", help="PI0.5 base model repo id")
    parser.add_argument("--adapter", help="Optional trained LoRA adapter repo id")
    parser.add_argument("--dataset-revision")
    parser.add_argument("--base-revision")
    parser.add_argument("--adapter-revision")
    parser.add_argument(
        "--video-backend",
        default="pyav",
        help="Decoder used while verifying downloaded videos (default: pyav).",
    )
    args = parser.parse_args()

    # Use LeRobot's loader for data. It selects the compatible v3 revision and
    # writes to HF_LEROBOT_HOME/hub, which is the exact cache used by training.
    from lerobot.datasets import LeRobotDataset

    print(f"[prefetch] dataset: {args.dataset}", flush=True)
    dataset = LeRobotDataset(
        repo_id=args.dataset,
        revision=args.dataset_revision,
        download_videos=True,
        video_backend=args.video_backend,
    )
    result = {"dataset": str(dataset.root)}

    print(f"[prefetch] base: {args.base}", flush=True)
    result["base"] = snapshot_download(
        repo_id=args.base,
        repo_type="model",
        revision=args.base_revision,
    )
    if args.adapter:
        print(f"[prefetch] adapter: {args.adapter}", flush=True)
        result["adapter"] = snapshot_download(
            repo_id=args.adapter,
            repo_type="model",
            revision=args.adapter_revision,
        )
    result["HF_HOME"] = os.environ.get("HF_HOME")
    result["HF_LEROBOT_HOME"] = os.environ.get("HF_LEROBOT_HOME")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
