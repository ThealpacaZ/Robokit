#!/usr/bin/env python3
"""Download one task's HDF5 from a robokit dataset repo and stage one flat source directory."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import tempfile

from huggingface_hub import snapshot_download


DEFAULT_SOURCE_REPO = "shaohuan1/lememory"
DEFAULT_TASK_SLUG = "stack_cups"
DEFAULT_EXPECTED_EPISODES = 202


def _natural_key(path: Path) -> tuple[int, int | str]:
    return (0, int(path.stem)) if path.stem.isdigit() else (1, path.stem)


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


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


def _merge_reports(paths: list[Path], episode_names: list[str]) -> dict:
    selected = set(episode_names)
    episodes: dict[str, dict] = {}
    dataset_errors: list[str] = []
    dataset_warnings: list[str] = []
    for path in paths:
        report = _load_json(path)
        for name, result in report.get("episodes", {}).items():
            if name not in selected:
                continue
            if name in episodes:
                raise RuntimeError(f"duplicate episode {name!r} in clean reports")
            episodes[name] = result
        dataset = report.get("dataset", {})
        dataset_errors.extend(dataset.get("errors", []))
        dataset_warnings.extend(dataset.get("warnings", []))

    missing = sorted(set(episode_names) - set(episodes))
    extra = sorted(set(episodes) - set(episode_names))
    if missing or extra:
        raise RuntimeError(
            f"clean report/source mismatch: missing={missing[:8]}, extra={extra[:8]}"
        )
    statuses = Counter(result.get("status", "unknown") for result in episodes.values())
    return {
        "dataset": {
            "errors": dataset_errors,
            "warnings": dataset_warnings,
        },
        "episodes": dict(sorted(episodes.items(), key=lambda item: _natural_key(Path(item[0])))),
        "summary": {
            "total": len(episodes),
            **dict(sorted(statuses.items())),
        },
    }


def stage(args: argparse.Namespace) -> dict:
    if args.snapshot_dir is not None:
        snapshot = args.snapshot_dir.expanduser().resolve()
        if not snapshot.is_dir():
            raise FileNotFoundError(f"--snapshot-dir is not a directory: {snapshot}")
    # 仓库里有两种批次布局：新的 <slug>/<批次>/…（一个仓库多个任务，任务一层目录），
    # 老的 <slug>_<批次>/…（slug 和批次拼在一个目录名里）。两种都收，别漏掉早期批次。
    layouts = [f"{args.task_slug}/b*", f"{args.task_slug}_b*"]
    if args.snapshot_dir is None:
        snapshot = Path(
            snapshot_download(
                repo_id=args.source_repo,
                repo_type="dataset",
                revision=args.revision,
                allow_patterns=[
                    f"{layout}/{leaf}"
                    for layout in layouts
                    for leaf in ("hdf5/*.hdf5",
                                 "source_meta/clean_report.json",
                                 "source_meta/config.json")
                ],
            )
        )
    def _glob(leaf: str) -> list[Path]:
        return [p for layout in layouts for p in snapshot.glob(f"{layout}/{leaf}")]

    source_files = sorted(_glob("hdf5/*.hdf5"), key=_natural_key)
    if not source_files:
        raise RuntimeError(
            f"{args.source_repo}: no {args.task_slug}/b*/hdf5/*.hdf5 "
            f"(or {args.task_slug}_b*/hdf5/*.hdf5) files"
        )
    names = [path.name for path in source_files]
    duplicates = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicates:
        raise RuntimeError(f"duplicate HDF5 names across batches: {duplicates[:8]}")
    if args.max_episodes is not None:
        source_files = source_files[:args.max_episodes]
    if args.expected_episodes and len(source_files) != args.expected_episodes:
        raise RuntimeError(
            f"expected {args.expected_episodes} HDF5 episodes, found {len(source_files)}"
        )
    numeric_ids = [int(path.stem) for path in source_files if path.stem.isdigit()]
    if len(numeric_ids) != len(source_files) or numeric_ids != list(range(len(source_files))):
        raise RuntimeError(
            f"{args.task_slug} episode ids must be one contiguous zero-based range; "
            f"got first={numeric_ids[:5]}, last={numeric_ids[-5:]}"
        )

    reports = sorted(_glob("source_meta/clean_report.json"))
    configs = sorted(_glob("source_meta/config.json"))
    if not reports or not configs:
        raise RuntimeError("source_meta clean_report.json/config.json is missing")
    config_payloads = [_load_json(path) for path in configs]
    if any(payload != config_payloads[0] for payload in config_payloads[1:]):
        raise RuntimeError("batch source_meta/config.json files differ")
    merged_report = _merge_reports(reports, names)
    if merged_report["dataset"]["errors"]:
        raise RuntimeError(
            f"published clean reports contain dataset errors: "
            f"{merged_report['dataset']['errors'][:3]}"
        )
    bad = [
        name
        for name, result in merged_report["episodes"].items()
        if result.get("status") == "bad"
    ]
    if bad:
        raise RuntimeError(f"published source still contains bad episodes: {bad[:8]}")

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    expected_names = set(names) | {
        "clean_report.json",
        "config.json",
        "hf_source_manifest.json",
    }
    unexpected = sorted(path.name for path in output.iterdir() if path.name not in expected_names)
    if unexpected:
        raise RuntimeError(f"refusing to mix files into {output}: unexpected={unexpected[:8]}")

    for source in source_files:
        destination = output / source.name
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source.resolve():
                raise RuntimeError(f"{destination} already points to another source")
            continue
        destination.symlink_to(source.resolve())

    _atomic_json(output / "clean_report.json", merged_report)
    _atomic_json(output / "config.json", config_payloads[0])
    manifest = {
        "source_repo": args.source_repo,
        "revision": args.revision,
        "snapshot": str(snapshot),
        "episode_count": len(source_files),
        "episode_ids": [0, len(source_files) - 1],
        "source_bytes": sum(path.stat().st_size for path in source_files),
        "batches": sorted({path.parents[1].name for path in source_files}),
        "staged_directory": str(output),
    }
    _atomic_json(output / "hf_source_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", default=DEFAULT_SOURCE_REPO)
    parser.add_argument("--task-slug", default=DEFAULT_TASK_SLUG,
                        help="仓库里的任务目录名（新布局 <slug>/<批次>/，老布局 <slug>_<批次>/）")
    parser.add_argument("--revision")
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        help="Use an already-downloaded HF snapshot instead of making a Hub request",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-episodes", type=int, default=DEFAULT_EXPECTED_EPISODES)
    parser.add_argument(
        "--max-episodes",
        type=int,
        help="Use only the first N globally ordered episodes",
    )
    args = parser.parse_args()
    print(json.dumps(stage(args), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
