#!/usr/bin/env python3
"""Monitor epoch checkpoints, retain the significant best, and publish it on convergence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import signal
import tempfile
import time
from typing import Any


EVAL_PATTERN = re.compile(r"step\s+(\d+):\s+eval_loss=([0-9.eE+-]+)")
WANDB_PATTERN = re.compile(r"https://wandb\.ai/([^/\s]+)/([^/\s]+)/runs/([^/\s]+)")
FULL_REQUIRED = {
    "model.safetensors",
    "config.json",
    "train_config.json",
    "policy_preprocessor.json",
    "policy_postprocessor.json",
}
STATE_REQUIRED = {
    "training_step.json",
    "optimizer_state.safetensors",
    "optimizer_param_groups.json",
    "rng_state.safetensors",
    "scheduler_state.json",
}


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _terminate_process_group(pid: int) -> None:
    if not _process_exists(pid):
        return
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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


def _required_files(checkpoint: Path) -> list[Path]:
    pretrained = checkpoint / "pretrained_model"
    state = checkpoint / "training_state"
    return [pretrained / name for name in sorted(FULL_REQUIRED)] + [
        state / name for name in sorted(STATE_REQUIRED)
    ]


def _checkpoint_complete(checkpoint: Path) -> bool:
    # The optimizer tensor may have been removed from an older checkpoint
    # after it was validated. Model selection only needs the full pretrained
    # model plus the lightweight training metadata; a transition/resume
    # controller separately requires the optimizer tensor at its resume step.
    required = [
        path
        for path in _required_files(checkpoint)
        if path.name != "optimizer_state.safetensors"
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def _link_tree(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.new"
    previous = destination.parent / f".{destination.name}.old"
    shutil.rmtree(temporary, ignore_errors=True)
    shutil.rmtree(previous, ignore_errors=True)
    shutil.copytree(source, temporary, copy_function=os.link)
    missing = sorted(name for name in FULL_REQUIRED if not (temporary / name).is_file())
    if missing:
        raise RuntimeError(f"best checkpoint is incomplete after linking: {missing}")
    if destination.exists():
        destination.rename(previous)
    temporary.rename(destination)
    shutil.rmtree(previous, ignore_errors=True)


def _strip_and_prune(checkpoints_dir: Path, keep_step: int) -> None:
    # Keep the newest checkpoint fully resumable. Delete only older
    # checkpoints; a checkpoint for a later step can already exist when a
    # restarted monitor replays earlier eval records from the log.
    for path in checkpoints_dir.iterdir():
        if path.is_dir() and path.name.isdigit() and int(path.name) < keep_step:
            shutil.rmtree(path)


def _cap_checkpoints(
    checkpoints_dir: Path,
    max_checkpoints: int,
    processed_steps: set[int],
) -> None:
    """Delete the oldest evaluated checkpoints before a fourth can accumulate."""
    if max_checkpoints <= 0 or not checkpoints_dir.is_dir():
        return
    checkpoints = sorted(
        (
            path
            for path in checkpoints_dir.iterdir()
            if path.is_dir() and path.name.isdigit()
        ),
        key=lambda path: int(path.name),
    )
    while len(checkpoints) > max_checkpoints:
        oldest = checkpoints[0]
        if int(oldest.name) not in processed_steps:
            # Never discard weights before their eval has participated in best
            # selection. The normal eval path will process and prune it.
            break
        shutil.rmtree(oldest)
        checkpoints.pop(0)


def _has_downward_trend(
    evaluations: list[dict[str, float | int | bool]],
    window: int,
    min_relative_improvement: float,
) -> bool:
    if len(evaluations) < window:
        return False
    recent = evaluations[-window:]
    first = float(recent[0]["eval_loss"])
    last = float(recent[-1]["eval_loss"])
    return last < first * (1.0 - min_relative_improvement)


def _sha256(path: Path, chunk_size: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_and_verify(final_dir: Path, repo_id: str) -> dict[str, Any]:
    from huggingface_hub import HfApi

    missing = sorted(name for name in FULL_REQUIRED if not (final_dir / name).is_file())
    if missing:
        raise RuntimeError(f"{final_dir}: missing final model files {missing}")

    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=final_dir,
        path_in_repo=".",
        commit_message=f"Publish converged full PI0.5 from {final_dir.name}",
    )
    remote = {
        item.path: item
        for item in api.list_repo_tree(repo_id, repo_type="model", recursive=True, expand=True)
        if getattr(item, "path", None)
    }
    remote_missing = sorted(FULL_REQUIRED - set(remote))
    if remote_missing:
        raise RuntimeError(f"HF verification failed; root files missing: {remote_missing}")

    model_path = final_dir / "model.safetensors"
    model_sha256 = _sha256(model_path)
    model_remote = remote["model.safetensors"]
    if int(getattr(model_remote, "size", -1)) != model_path.stat().st_size:
        raise RuntimeError("HF model.safetensors size differs from local final model")
    lfs = getattr(model_remote, "lfs", None)
    remote_sha256 = (
        getattr(lfs, "sha256", None)
        if lfs is not None
        else None
    )
    if remote_sha256 is None and isinstance(lfs, dict):
        remote_sha256 = lfs.get("sha256")
    if remote_sha256 and remote_sha256 != model_sha256:
        raise RuntimeError(
            f"HF model.safetensors SHA-256 mismatch: local={model_sha256}, remote={remote_sha256}"
        )
    return {
        "repo_id": repo_id,
        "commit": str(commit),
        "model_bytes": model_path.stat().st_size,
        "model_sha256": model_sha256,
        "remote_sha256": remote_sha256,
        "verified_files": sorted(FULL_REQUIRED),
    }


def _wandb_url(log: Path) -> str | None:
    if not log.is_file():
        return None
    match = WANDB_PATTERN.search(log.read_text(encoding="utf-8", errors="replace"))
    return match.group(0) if match else None


def monitor(args: argparse.Namespace) -> int:
    best_loss = math.inf
    best_step = 0
    stale_evals = 0
    evaluations: list[dict[str, float | int | bool]] = []
    offset = 0
    partial = ""
    pending: list[tuple[int, float]] = []
    processed_steps: set[int] = set()
    convergence_step: int | None = None
    stop_reason = "plateau"
    trainer_finished_at: float | None = None

    def save_state() -> None:
        if args.state_file is None:
            return
        _atomic_json(
            args.state_file,
            {
                "best_loss": best_loss,
                "best_step": best_step,
                "stale_evals": stale_evals,
                "evaluations": evaluations,
                "offset": offset,
                "partial": partial,
                "pending": pending,
                "processed_steps": sorted(processed_steps),
            },
        )

    if args.state_file is not None and args.state_file.is_file():
        state = json.loads(args.state_file.read_text(encoding="utf-8"))
        best_loss = float(state["best_loss"])
        best_step = int(state["best_step"])
        stale_evals = int(state["stale_evals"])
        evaluations = list(state["evaluations"])
        offset = int(state.get("offset", 0))
        partial = str(state.get("partial", ""))
        pending = [
            (int(item[0]), float(item[1]))
            for item in state.get("pending", [])
        ]
        processed_steps = {
            int(step) for step in state.get("processed_steps", [])
        }

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
                    if not math.isfinite(loss):
                        raise RuntimeError(f"non-finite eval loss at step {step}: {loss}")
                    if step not in processed_steps and all(step != item[0] for item in pending):
                        pending.append((step, loss))
                save_state()

        _cap_checkpoints(
            args.output_dir / "checkpoints",
            args.max_checkpoints,
            processed_steps,
        )

        while pending:
            step, loss = pending[0]
            checkpoint = args.output_dir / "checkpoints" / f"{step:06d}"
            if not _checkpoint_complete(checkpoint):
                break
            improved = loss < best_loss * (1.0 - args.min_relative_improvement)
            if improved:
                best_loss = loss
                best_step = step
                stale_evals = 0
                _link_tree(checkpoint / "pretrained_model", args.final_dir)
            else:
                stale_evals += 1
            evaluations.append(
                {
                    "step": step,
                    "epoch": step / args.steps_per_epoch,
                    "eval_loss": loss,
                    "significant_improvement": improved,
                    "best_step": best_step,
                    "best_eval_loss": best_loss,
                    "stale_evals": stale_evals,
                }
            )
            processed_steps.add(step)
            pending.pop(0)
            _strip_and_prune(args.output_dir / "checkpoints", step)
            _cap_checkpoints(
                args.output_dir / "checkpoints",
                args.max_checkpoints,
                processed_steps,
            )
            save_state()

            if (
                step >= args.min_steps
                and len(evaluations) >= args.min_evals
                and stale_evals >= args.patience
                and not _has_downward_trend(
                    evaluations,
                    args.trend_window,
                    args.trend_relative_improvement,
                )
            ):
                convergence_step = step
                break

        if convergence_step is not None:
            _terminate_process_group(args.pid)
            publish = _publish_and_verify(args.final_dir, args.repo_id)
            payload = {
                "status": "converged",
                "stop_reason": stop_reason,
                "stop_step": convergence_step,
                "stop_epoch": convergence_step / args.steps_per_epoch,
                "publish_step": best_step,
                "best_step": best_step,
                "best_eval_loss": best_loss,
                "steps_per_epoch": args.steps_per_epoch,
                "per_device_batch_size": args.batch_size,
                "global_batch_size": args.batch_size * args.num_processes,
                "patience": args.patience,
                "min_relative_improvement": args.min_relative_improvement,
                "trend_window": args.trend_window,
                "trend_relative_improvement": args.trend_relative_improvement,
                "hard_max_steps": args.hard_max_steps,
                "evaluations": evaluations,
                "wandb_url": _wandb_url(args.log),
                "publish": publish,
            }
            _atomic_json(args.marker, payload)
            shutil.rmtree(args.output_dir / "checkpoints", ignore_errors=True)
            return 0

        if not _process_exists(args.pid):
            if args.hard_max_steps > 0:
                hard_checkpoint = (
                    args.output_dir / "checkpoints" / f"{args.hard_max_steps:06d}"
                )
                if _checkpoint_complete(hard_checkpoint):
                    convergence_step = args.hard_max_steps
                    stop_reason = "hard_limit"
                    continue
            if trainer_finished_at is None:
                trainer_finished_at = time.monotonic()
            # Give the log/checkpoint writers a short grace period after the
            # accelerate launcher exits.
            if pending and time.monotonic() - trainer_finished_at < args.checkpoint_timeout:
                time.sleep(args.poll_interval)
                continue
            return 2
        time.sleep(args.poll_interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--final-dir", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--steps-per-epoch", type=int, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--min-steps", type=int, required=True)
    parser.add_argument("--min-evals", type=int, default=5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--min-relative-improvement", type=float, default=0.005)
    parser.add_argument("--trend-window", type=int, default=3)
    parser.add_argument("--trend-relative-improvement", type=float, default=0.005)
    parser.add_argument("--hard-max-steps", type=int, default=0)
    parser.add_argument("--state-file", type=Path)
    parser.add_argument("--max-checkpoints", type=int, default=3)
    parser.add_argument("--poll-interval", type=float, default=30.0)
    parser.add_argument("--checkpoint-timeout", type=float, default=1800.0)
    args = parser.parse_args()
    if args.pid <= 0 or args.steps_per_epoch <= 0 or args.batch_size <= 0:
        parser.error("pid, steps-per-epoch, and batch-size must be positive")
    if (
        args.min_steps < 0
        or args.min_evals <= 0
        or args.patience <= 0
        or args.trend_window < 2
        or args.hard_max_steps < 0
        or args.max_checkpoints <= 0
    ):
        parser.error("min/eval/patience values are invalid")
    if not 0 < args.min_relative_improvement < 1:
        parser.error("--min-relative-improvement must be in (0, 1)")
    if not 0 < args.trend_relative_improvement < 1:
        parser.error("--trend-relative-improvement must be in (0, 1)")
    raise SystemExit(monitor(args))


if __name__ == "__main__":
    main()
