#!/usr/bin/env python3
"""Combine the five immutable HF TFDS batches into one zero-copy TFDS dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", default="robokit_stackcups_all")
    parser.add_argument("--expected-episodes", type=int, default=401)
    args = parser.parse_args()

    versions = []
    for batch_index in range(5):
        candidates = sorted((args.source / f"stack_cups_b{batch_index}" / "robokit_dataset").glob("*"))
        candidates = [path for path in candidates if path.is_dir() and (path / "dataset_info.json").is_file()]
        if len(candidates) != 1:
            raise RuntimeError(f"batch {batch_index}: expected one TFDS version, found {candidates}")
        versions.append(candidates[0].resolve())

    infos = [json.loads((path / "dataset_info.json").read_text(encoding="utf-8")) for path in versions]
    feature_bytes = [(path / "features.json").read_bytes() for path in versions]
    if len(set(feature_bytes)) != 1:
        raise RuntimeError("HF batches do not share an identical features.json")

    shards: list[tuple[Path, int]] = []
    total_bytes = 0
    for version, info in zip(versions, infos):
        splits = info.get("splits", [])
        if len(splits) != 1 or splits[0].get("name") != "train":
            raise RuntimeError(f"{version}: expected exactly one train split")
        lengths = [int(value) for value in splits[0]["shardLengths"]]
        files = sorted(version.glob("robokit_dataset-train.tfrecord-*"))
        if len(files) != len(lengths):
            raise RuntimeError(f"{version}: {len(files)} files != {len(lengths)} shard lengths")
        shards.extend(zip(files, lengths))
        total_bytes += sum(path.stat().st_size for path in files)

    total_episodes = sum(length for _, length in shards)
    if total_episodes != args.expected_episodes:
        raise RuntimeError(f"expected {args.expected_episodes} episodes, found {total_episodes}")

    output = args.output_root / args.dataset_name / "1.1.0"
    output.mkdir(parents=True, exist_ok=True)
    shard_count = len(shards)
    for index, (source, _) in enumerate(shards):
        destination = output / (
            f"{args.dataset_name}-train.tfrecord-{index:05d}-of-{shard_count:05d}"
        )
        if destination.exists() or destination.is_symlink():
            if destination.resolve() != source:
                raise RuntimeError(f"refusing to replace unexpected path: {destination}")
        else:
            os.symlink(source, destination)

    info = dict(infos[0])
    info["name"] = args.dataset_name
    info["moduleName"] = "robokit_stackcups_all_dataset_builder"
    info["splits"] = [
        {
            "filepathTemplate": "{DATASET}-{SPLIT}.{FILEFORMAT}-{SHARD_X_OF_Y}",
            "name": "train",
            "numBytes": str(total_bytes),
            "shardLengths": [str(length) for _, length in shards],
        }
    ]
    (output / "features.json").write_bytes(feature_bytes[0])
    (output / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(
        json.dumps(
            {
                "output": str(output),
                "episodes": total_episodes,
                "shards": shard_count,
                "tfrecord_bytes": total_bytes,
                "source_batches": [str(path) for path in versions],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
