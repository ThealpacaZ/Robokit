#!/usr/bin/env python3
"""Verify that a converged marker's W&B run is externally readable and complete."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any


WANDB_PATTERN = re.compile(r"https://wandb\.ai/([^/\s]+)/([^/\s]+)/runs/([^/\s]+)")


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _verify_once(marker: dict[str, Any]) -> dict[str, Any]:
    import wandb

    url = marker.get("wandb_url")
    match = WANDB_PATTERN.fullmatch(str(url or ""))
    if match is None:
        raise RuntimeError(f"converged marker has no valid W&B URL: {url!r}")
    entity, project, run_id = match.groups()
    run = wandb.Api(timeout=60).run(f"{entity}/{project}/{run_id}")
    eval_rows: list[tuple[int, float]] = []
    train_rows = 0
    for row in run.scan_history():
        train_loss = row.get("train/loss")
        if train_loss is not None and math.isfinite(float(train_loss)):
            train_rows += 1
        eval_loss = row.get("eval/eval_loss")
        if eval_loss is not None and math.isfinite(float(eval_loss)):
            source_step = row.get("eval/source_step")
            eval_rows.append(
                (
                    int(source_step if source_step is not None else row["_step"]),
                    float(eval_loss),
                )
            )
    expected = [
        (int(item["step"]), float(item["eval_loss"]))
        for item in marker.get("evaluations", [])
    ]
    if train_rows == 0:
        raise RuntimeError("W&B run has no finite train/loss rows")
    if len(eval_rows) < len(expected):
        raise RuntimeError(
            f"W&B has {len(eval_rows)} eval points, expected at least {len(expected)}"
        )
    actual_by_step = dict(eval_rows)
    missing = [
        {"step": step, "expected": loss, "actual": actual_by_step.get(step)}
        for step, loss in expected
        if step not in actual_by_step
        or not math.isclose(actual_by_step[step], loss, rel_tol=5e-4, abs_tol=5e-5)
    ]
    if missing:
        raise RuntimeError(f"W&B eval history differs from trainer log: {missing[:5]}")
    return {
        "url": url,
        "entity": entity,
        "project": project,
        "run_id": run_id,
        "state": run.state,
        "train_rows": train_rows,
        "eval_rows": len(eval_rows),
        "summary_keys": sorted(run.summary.keys()),
    }


def _repair_missing_evals(marker: dict[str, Any]) -> int:
    import wandb

    url = marker.get("wandb_url")
    match = WANDB_PATTERN.fullmatch(str(url or ""))
    if match is None:
        raise RuntimeError(f"converged marker has no valid W&B URL: {url!r}")
    entity, project, run_id = match.groups()
    api_run = wandb.Api(timeout=60).run(f"{entity}/{project}/{run_id}")
    actual_steps: set[int] = set()
    for row in api_run.scan_history():
        value = row.get("eval/eval_loss")
        if value is None:
            continue
        source_step = row.get("eval/source_step")
        actual_steps.add(int(source_step if source_step is not None else row["_step"]))
    missing = [
        (int(item["step"]), float(item["eval_loss"]))
        for item in marker.get("evaluations", [])
        if int(item["step"]) not in actual_steps
    ]
    if not missing:
        return 0
    run = wandb.init(
        entity=entity,
        project=project,
        id=run_id,
        resume="must",
        settings=wandb.Settings(init_timeout=60),
    )
    for step, loss in missing:
        run.log(
            {
                "eval/eval_loss": loss,
                "eval/source_step": step,
            }
        )
    run.finish(exit_code=0)
    return len(missing)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--repair-missing-evals", action="store_true")
    args = parser.parse_args()
    marker = json.loads(args.marker.read_text(encoding="utf-8"))
    deadline = time.monotonic() + args.timeout
    last_error: Exception | None = None
    repair_attempted = False
    while time.monotonic() < deadline:
        try:
            marker["wandb_verification"] = _verify_once(marker)
            _atomic_json(args.marker, marker)
            print(json.dumps(marker["wandb_verification"], ensure_ascii=False, indent=2))
            return
        except Exception as exc:
            last_error = exc
            if args.repair_missing_evals and not repair_attempted:
                repaired = _repair_missing_evals(marker)
                repair_attempted = True
                if repaired:
                    print(f"repaired {repaired} missing W&B eval point(s)")
            time.sleep(args.poll_interval)
    raise TimeoutError(f"W&B verification did not succeed within {args.timeout}s: {last_error}")


if __name__ == "__main__":
    main()
