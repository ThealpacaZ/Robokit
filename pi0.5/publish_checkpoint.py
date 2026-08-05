#!/usr/bin/env python3
"""Publish one complete PI0/PI0.5 PEFT checkpoint at the root of an HF model repo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import HfApi


REQUIRED = {
    "adapter_model.safetensors",
    "adapter_config.json",
    "config.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--private", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    pretrained = args.checkpoint.expanduser().resolve() / "pretrained_model"
    missing = sorted(name for name in REQUIRED if not (pretrained / name).is_file())
    if missing:
        raise RuntimeError(f"{pretrained}: incomplete checkpoint, missing {missing}")
    config = json.loads((pretrained / "config.json").read_text(encoding="utf-8"))
    policy_type = str(config.get("type", "policy")).upper()

    api = HfApi()
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    commit = api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=pretrained,
        path_in_repo=".",
        commit_message=f"Publish {policy_type} LoRA from {args.checkpoint.name}",
    )
    remote = set(api.list_repo_files(args.repo_id, repo_type="model"))
    remote_missing = sorted(REQUIRED - remote)
    if remote_missing:
        raise RuntimeError(f"HF verification failed; root files missing: {remote_missing}")
    print(
        json.dumps(
            {
                "repo_id": args.repo_id,
                "checkpoint": str(args.checkpoint),
                "policy_type": policy_type,
                "commit": str(commit),
                "verified": sorted(REQUIRED),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
