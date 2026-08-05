#!/usr/bin/env python3
"""Low-frequency convergence monitor, HF publisher, and success-only shutdown."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import signal
import statistics
import subprocess
import tempfile
import time

import wandb
from huggingface_hub import HfApi


def atomic_json(path: Path, payload: dict) -> None:
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


def checkpoint_complete(run_dir: Path) -> bool:
    required = [
        run_dir / "lora_adapter" / "adapter_model.safetensors",
        run_dir / "lora_adapter" / "adapter_config.json",
        run_dir / "action_head--latest_checkpoint.pt",
        run_dir / "dataset_statistics.json",
        run_dir / "processor_config.json",
    ]
    return all(path.is_file() and path.stat().st_size > 0 for path in required)


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def history_points(run, key: str) -> list[tuple[int, float]]:
    points = []
    for row in run.scan_history(keys=["_step", key], page_size=10_000):
        step, value = row.get("_step"), row.get(key)
        if step is None or value is None:
            continue
        value = float(value)
        if math.isfinite(value):
            points.append((int(step), value))
    return points


def latest_saved_step(train_log: Path) -> int:
    if not train_log.is_file():
        return 0
    matches = re.findall(
        r"Saving Model Checkpoint for Step\s+(\d+)",
        train_log.read_text(encoding="utf-8", errors="replace"),
    )
    return int(matches[-1]) if matches else 0


def window_median(points: list[tuple[int, float]], end_step: int, width: int) -> float | None:
    values = [value for step, value in points if end_step - width < step <= end_step]
    return statistics.median(values) if values else None


def publish(run_dir: Path, repo_id: str, token: str, step: int, evidence: dict) -> str:
    manifest = run_dir / "convergence.json"
    atomic_json(manifest, evidence)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)
    allow = [
        "lora_adapter/*",
        "action_head--latest_checkpoint.pt",
        "dataset_statistics.json",
        "processor_config.json",
        "preprocessor_config.json",
        "processing_prismatic.py",
        "tokenizer*",
        "special_tokens_map.json",
        "added_tokens.json",
        "convergence.json",
        "training_config.json",
    ]
    commit = api.upload_folder(
        repo_id=repo_id,
        repo_type="model",
        folder_path=run_dir,
        path_in_repo=".",
        allow_patterns=allow,
        commit_message=f"Publish converged OpenVLA-OFT LoRA at step {step}",
    )
    remote = set(api.list_repo_files(repo_id, repo_type="model"))
    required = {
        "lora_adapter/adapter_model.safetensors",
        "lora_adapter/adapter_config.json",
        "action_head--latest_checkpoint.pt",
        "dataset_statistics.json",
        "convergence.json",
        "training_config.json",
    }
    missing = sorted(required - remote)
    if missing:
        raise RuntimeError(f"HF verification failed, missing {missing}")
    return str(commit)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--run-path", required=True, help="entity/project/run_id")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--poll-seconds", type=int, default=1800)
    parser.add_argument("--save-freq", type=int, default=5000)
    parser.add_argument("--min-step", type=int, default=80000)
    parser.add_argument("--max-step", type=int, default=150000)
    parser.add_argument("--window-steps", type=int, default=2000)
    parser.add_argument("--loss-threshold", type=float, default=0.01)
    parser.add_argument("--relative-delta", type=float, default=0.005)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    api = wandb.Api(timeout=120)
    checked_boundaries: list[dict] = []
    stale = 0
    best = math.inf

    while True:
        try:
            run = api.run(args.run_path)
            points = history_points(run, "VLA Train/Loss")
            (args.run_dir.parent / "monitor_last_error.json").unlink(missing_ok=True)
            latest_step = max((step for step, _ in points), default=0)
            boundary = latest_saved_step(args.train_log)
            already = {item["step"] for item in checked_boundaries}
            # Training blocks while saving. Seeing a later W&B step proves that
            # every file for this latest-only checkpoint has finished writing.
            checkpoint_stable = latest_step > boundary or not process_exists(args.pid)
            if (
                boundary >= args.min_step
                and boundary not in already
                and checkpoint_stable
                and checkpoint_complete(args.run_dir)
            ):
                loss = window_median(points, boundary, args.window_steps)
                if loss is not None:
                    improved = loss < best * (1.0 - args.relative_delta)
                    if improved:
                        best = loss
                        stale = 0
                    else:
                        stale += 1
                    checked_boundaries.append(
                        {"step": boundary, "median_loss": loss, "best_loss": best, "stale": stale}
                    )
                    atomic_json(
                        args.run_dir.parent / "monitor_state.json",
                        {
                            "status": "monitoring",
                            "latest_wandb_step": latest_step,
                            "latest_saved_step": boundary,
                            "checkpoints": checked_boundaries,
                        },
                    )
                    if loss <= args.loss_threshold and stale >= args.patience:
                        evidence = {
                            "status": "converged",
                            "step": boundary,
                            "criterion": {
                                "loss_threshold": args.loss_threshold,
                                "relative_delta": args.relative_delta,
                                "patience": args.patience,
                                "window_steps": args.window_steps,
                            },
                            "checkpoints": checked_boundaries,
                            "wandb_run": args.run_path,
                        }
                        if process_exists(args.pid):
                            os.killpg(args.pid, signal.SIGTERM)
                            deadline = time.time() + 300
                            while process_exists(args.pid) and time.time() < deadline:
                                time.sleep(5)
                        commit = publish(
                            args.run_dir,
                            args.repo_id,
                            os.environ["HF_TOKEN"],
                            boundary,
                            evidence,
                        )
                        evidence["hf_commit"] = commit
                        atomic_json(args.run_dir / "COMPLETED.json", evidence)
                        os.sync()
                        subprocess.run(["/usr/bin/shutdown", "-h", "now"], check=True)
                        return
            alive = process_exists(args.pid)
            if latest_step >= args.max_step and not alive:
                atomic_json(
                    args.run_dir.parent / "NOT_CONVERGED.json",
                    {
                        "status": "max_steps_without_convergence",
                        "latest_step": latest_step,
                        "checkpoints": checked_boundaries,
                    },
                )
                return
        except Exception as error:
            atomic_json(
                args.run_dir.parent / "monitor_last_error.json",
                {"time": time.time(), "error": repr(error), "checkpoints": checked_boundaries},
            )
        alive = process_exists(args.pid)
        if not alive:
            atomic_json(
                args.run_dir.parent / "TRAINER_EXITED.json",
                {
                    "status": "trainer_exited_without_verified_convergence",
                    "checkpoints": checked_boundaries,
                },
            )
            return
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
