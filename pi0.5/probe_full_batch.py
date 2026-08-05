#!/usr/bin/env python3
"""Find the largest stable per-GPU PI0.5 full-finetune batch on two GPUs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any


def _gpu_memory() -> tuple[list[int], list[int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    used: list[int] = []
    total: list[int] = []
    for line in result.stdout.splitlines():
        fields = [int(value.strip()) for value in line.split(",")]
        if len(fields) != 2:
            raise RuntimeError(f"unexpected nvidia-smi row: {line!r}")
        used.append(fields[0])
        total.append(fields[1])
    if len(used) < 2:
        raise RuntimeError(f"expected two GPUs, found {len(used)}")
    return used[:2], total[:2]


def _wait_gpu_idle(timeout: float = 120.0, threshold_mib: int = 1024) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        used, _ = _gpu_memory()
        if all(value <= threshold_mib for value in used):
            return
        time.sleep(5)
    raise TimeoutError("GPUs did not return to an idle memory state after a probe")


def _tail(path: Path, max_bytes: int = 16_000) -> str:
    if not path.is_file():
        return ""
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - max_bytes))
        return stream.read().decode("utf-8", errors="replace")


def run_candidate(args: argparse.Namespace, batch_size: int) -> dict[str, Any]:
    output = args.work_dir / f"bs{batch_size:03d}"
    shutil.rmtree(output, ignore_errors=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    log = args.work_dir / f"bs{batch_size:03d}.log"
    log.unlink(missing_ok=True)

    env = os.environ.copy()
    env.update(
        {
            "DATASET_REPO": args.dataset_repo,
            "POLICY_REPO": args.policy_repo,
            "OUTPUT_DIR": str(output),
            "JOB_NAME": f"{args.job_prefix}_probe_bs{batch_size}",
            "BASE_MODEL": args.base_model,
            "STEPS": str(args.steps),
            "BATCH_SIZE": str(batch_size),
            "EVAL_STEPS": str(args.steps + 100),
            "SAVE_FREQ": str(args.steps + 100),
            "MAX_EVAL_SAMPLES": "1",
            "NUM_WORKERS": str(args.num_workers),
            "WANDB_ENABLE": "false",
            "WANDB_MODE": "disabled",
            "PUSH_TO_HUB": "false",
            "SAVE_CHECKPOINT": "false",
            "SAVE_CHECKPOINT_TO_HUB": "false",
        }
    )
    peak = [0, 0]
    _, total = _gpu_memory()
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            ["bash", str(args.train_script)],
            stdout=stream,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )
        timed_out = False
        while process.poll() is None:
            used, _ = _gpu_memory()
            peak = [max(before, current) for before, current in zip(peak, used, strict=True)]
            if time.monotonic() - started > args.timeout:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
            time.sleep(args.poll_interval)
        return_code = process.wait()
    used, _ = _gpu_memory()
    peak = [max(before, current) for before, current in zip(peak, used, strict=True)]
    ratios = [value / capacity for value, capacity in zip(peak, total, strict=True)]
    stable = return_code == 0 and not timed_out and max(ratios) <= args.max_memory_fraction
    result = {
        "batch_size_per_gpu": batch_size,
        "global_batch_size": batch_size * 2,
        "return_code": return_code,
        "timed_out": timed_out,
        "peak_memory_mib": peak,
        "total_memory_mib": total,
        "peak_fraction": ratios,
        "max_memory_fraction": args.max_memory_fraction,
        "stable": stable,
        "elapsed_seconds": time.monotonic() - started,
        "log": str(log),
    }
    shutil.rmtree(output, ignore_errors=True)
    _wait_gpu_idle()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-script", type=Path, required=True)
    parser.add_argument("--dataset-repo", required=True)
    parser.add_argument("--policy-repo", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--job-prefix", default="pi05_full")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--start-batch", type=int, default=2)
    parser.add_argument("--max-batch", type=int, default=128)
    parser.add_argument("--max-memory-fraction", type=float, default=0.90)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--poll-interval", type=float, default=5.0)
    parser.add_argument("--timeout", type=float, default=3600.0)
    args = parser.parse_args()
    if args.start_batch <= 0 or args.max_batch < args.start_batch:
        parser.error("invalid batch bounds")
    if not 0 < args.max_memory_fraction < 1:
        parser.error("--max-memory-fraction must be in (0, 1)")

    args.train_script = args.train_script.expanduser().resolve()
    args.work_dir = args.work_dir.expanduser().resolve()
    args.result = args.result.expanduser().resolve()
    args.work_dir.mkdir(parents=True, exist_ok=True)
    _wait_gpu_idle()

    attempts: list[dict[str, Any]] = []
    lower = 0
    candidate = args.start_batch
    upper = args.max_batch + 1
    while candidate <= args.max_batch:
        attempt = run_candidate(args, candidate)
        attempts.append(attempt)
        print(json.dumps(attempt, ensure_ascii=False), flush=True)
        if not attempt["stable"]:
            upper = candidate
            break
        lower = candidate
        if candidate == args.max_batch:
            upper = args.max_batch + 1
            break
        candidate = min(candidate * 2, args.max_batch)
    if lower == 0:
        raise RuntimeError(
            "full finetune is not stable even at the smallest batch; "
            f"last log:\n{_tail(Path(attempts[-1]['log']))}"
        )

    while upper - lower > 1:
        candidate = (lower + upper) // 2
        attempt = run_candidate(args, candidate)
        attempts.append(attempt)
        print(json.dumps(attempt, ensure_ascii=False), flush=True)
        if attempt["stable"]:
            lower = candidate
        else:
            upper = candidate

    payload = {
        "status": "complete",
        "selected_batch_size_per_gpu": lower,
        "selected_global_batch_size": lower * 2,
        "max_memory_fraction": args.max_memory_fraction,
        "attempts": attempts,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
