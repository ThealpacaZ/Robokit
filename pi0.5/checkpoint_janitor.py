#!/usr/bin/env python3
"""Bound checkpoint disk usage by deleting the oldest numeric checkpoint dirs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import time


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def prune(checkpoints_dir: Path, max_retained: int) -> None:
    if not checkpoints_dir.is_dir():
        return
    checkpoints = sorted(
        (
            path
            for path in checkpoints_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )
    while len(checkpoints) > max_retained:
        oldest = checkpoints.pop(0)
        if oldest.parent.resolve() != checkpoints_dir.resolve():
            raise RuntimeError(f"refusing unsafe checkpoint deletion: {oldest}")
        print(f"[checkpoint-janitor] deleting oldest checkpoint {oldest.name}", flush=True)
        shutil.rmtree(oldest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--trainer-pid", type=int, required=True)
    parser.add_argument("--max-retained", type=int, default=2)
    parser.add_argument("--poll-interval", type=float, default=15.0)
    args = parser.parse_args()
    if args.trainer_pid <= 0:
        parser.error("--trainer-pid must be positive")
    if not 1 <= args.max_retained <= 3:
        parser.error("--max-retained must be between 1 and 3")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")

    while process_exists(args.trainer_pid):
        prune(args.checkpoints_dir, args.max_retained)
        time.sleep(args.poll_interval)
    prune(args.checkpoints_dir, args.max_retained)


if __name__ == "__main__":
    main()
