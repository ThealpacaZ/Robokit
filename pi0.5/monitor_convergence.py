#!/usr/bin/env python3
"""Stop a LeRobot trainer after held-out eval loss plateaus and its checkpoint is durable."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import tempfile
import time


EVAL_PATTERN = re.compile(r"step\s+(\d+):\s+eval_loss=([0-9.eE+-]+)")


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _checkpoint_complete(output_dir: Path, step: int) -> bool:
    checkpoint = output_dir / "checkpoints" / f"{step:06d}"
    required = [
        checkpoint / "pretrained_model" / "adapter_model.safetensors",
        checkpoint / "pretrained_model" / "adapter_config.json",
        checkpoint / "pretrained_model" / "config.json",
        checkpoint / "training_state" / "training_step.json",
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def monitor(args: argparse.Namespace) -> int:
    best_loss = math.inf
    best_step = 0
    stale_evals = 0
    evaluations: list[dict[str, float | int]] = []
    offset = 0
    partial = ""
    candidate_step = None
    deadline = None

    while True:
        if args.log.is_file():
            with args.log.open(encoding="utf-8", errors="replace") as stream:
                stream.seek(offset)
                chunk = stream.read()
                offset = stream.tell()
            if chunk:
                partial += chunk
                lines = partial.splitlines(keepends=True)
                partial = ""
                if lines and not lines[-1].endswith(("\n", "\r")):
                    partial = lines.pop()
                for line in lines:
                    match = EVAL_PATTERN.search(line)
                    if match is None:
                        continue
                    step = int(match.group(1))
                    loss = float(match.group(2))
                    if evaluations and step <= int(evaluations[-1]["step"]):
                        continue
                    improved = loss < best_loss * (1.0 - args.min_relative_improvement)
                    if improved:
                        best_loss = loss
                        best_step = step
                        stale_evals = 0
                    else:
                        stale_evals += 1
                    evaluations.append(
                        {
                            "step": step,
                            "eval_loss": loss,
                            "best_step": best_step,
                            "best_eval_loss": best_loss,
                            "stale_evals": stale_evals,
                        }
                    )
                    if (
                        step >= args.min_steps
                        and len(evaluations) >= args.min_evals
                        and stale_evals >= args.patience
                    ):
                        candidate_step = step
                        deadline = time.monotonic() + args.checkpoint_timeout

        if candidate_step is not None:
            if _checkpoint_complete(args.output_dir, candidate_step):
                payload = {
                    "status": "converged",
                    "stop_step": candidate_step,
                    # The plateau is detected at candidate_step, but the model
                    # artifact to publish must be the best checkpoint under
                    # the same significant-improvement threshold, not the
                    # later stale checkpoint that merely proves the plateau.
                    "publish_step": best_step,
                    "best_step": best_step,
                    "best_eval_loss": best_loss,
                    "patience": args.patience,
                    "min_relative_improvement": args.min_relative_improvement,
                    "evaluations": evaluations,
                }
                _atomic_json(args.marker, payload)
                if _process_exists(args.pid):
                    # The trainer is launched as an asynchronous non-interactive
                    # shell job, where SIGINT may be inherited as ignored.
                    os.kill(args.pid, signal.SIGTERM)
                return 0
            if deadline is not None and time.monotonic() > deadline:
                raise TimeoutError(
                    f"checkpoint for converged eval step {candidate_step} was not completed "
                    f"within {args.checkpoint_timeout}s"
                )

        if not _process_exists(args.pid):
            return 2
        time.sleep(args.poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--min-steps", type=int, default=10_000)
    parser.add_argument("--min-evals", type=int, default=6)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-relative-improvement", type=float, default=0.005)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--checkpoint-timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.pid <= 0:
        parser.error("--pid must be positive")
    if args.min_steps < 0 or args.min_evals <= 0 or args.patience <= 0:
        parser.error("min/eval/patience values are invalid")
    if not 0 < args.min_relative_improvement < 1:
        parser.error("--min-relative-improvement must be in (0, 1)")
    raise SystemExit(monitor(args))


if __name__ == "__main__":
    main()
