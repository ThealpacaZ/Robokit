#!/usr/bin/env python3
"""Build a local PI0.5 base view with OpenPI's public PaliGemma tokenizer.

The PI0.5 weights are public, but the Transformers processor configuration
points at Google's gated PaliGemma repository just to load its tokenizer.
OpenPI itself loads the same SentencePiece model anonymously from
``gs://big_vision/paligemma_tokenizer.model``.  This helper downloads that
official public asset, verifies its fixed SHA-256, checks token IDs against
SentencePiece, and rewrites only the local processor's tokenizer path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import urllib.request

from huggingface_hub import snapshot_download


TOKENIZER_URL = "https://storage.googleapis.com/big_vision/paligemma_tokenizer.model"
TOKENIZER_SHA256 = "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6"
TOKENIZER_BYTES = 4_264_023
TOKENIZER_METADATA_REVISION = "39996beb6fb17c5d16a50d3ef8f7a96ad9d03986"
TOKENIZER_METADATA_BASE = (
    "https://huggingface.co/leo009/paligemma-3b-pt-224/resolve/"
    f"{TOKENIZER_METADATA_REVISION}"
)
TOKENIZER_FILES = {
    "tokenizer.model": (
        TOKENIZER_URL,
        TOKENIZER_BYTES,
        TOKENIZER_SHA256,
    ),
    # These files are a public byte-for-byte export of the Google repository's
    # tokenizer metadata. The Google gated tree reports the same file sizes;
    # every byte is pinned here, and token IDs are checked against the official
    # anonymous SentencePiece asset below.
    "added_tokens.json": (
        f"{TOKENIZER_METADATA_BASE}/added_tokens.json",
        24,
        "7d0bad90030d638a4bf89a82e91206e25f1fbe0012f14bca21666a64b643bc49",
    ),
    "config.json": (
        f"{TOKENIZER_METADATA_BASE}/config.json",
        1_027,
        "e00c72cdff16296bf1229c3267b99f5247d8c740cae84336866338d64fa7918f",
    ),
    "special_tokens_map.json": (
        f"{TOKENIZER_METADATA_BASE}/special_tokens_map.json",
        607,
        "5ef37093ae4236587b6e8266acb815b46e2db8ce656c66552bfa574d32880405",
    ),
    "tokenizer.json": (
        f"{TOKENIZER_METADATA_BASE}/tokenizer.json",
        17_549_604,
        "ef6773c135b77b834de1d13c75a4c98ab7a3684ffd602d1831e1f1bf5467c563",
    ),
    "tokenizer_config.json": (
        f"{TOKENIZER_METADATA_BASE}/tokenizer_config.json",
        39_968,
        "3259402b1d1802e02417d7bff75a889ec61d359d15be6050a957b307c48edbbe",
    ),
}
PREPROCESSOR_NAME = "policy_preprocessor.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_file(path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if path.stat().st_size != expected_bytes:
        raise RuntimeError(
            f"{path}: expected {expected_bytes} bytes, got {path.stat().st_size}"
        )
    actual = _sha256(path)
    if actual != expected_sha256:
        raise RuntimeError(f"{path}: SHA-256 {actual} != {expected_sha256}")


def _verify_tokenizer_files(output: Path) -> None:
    for name, (_, expected_bytes, expected_sha256) in TOKENIZER_FILES.items():
        _verify_file(output / name, expected_bytes, expected_sha256)


def _prepare_tokenizer(output: Path) -> None:
    import sentencepiece
    from transformers import AutoTokenizer

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        _verify_tokenizer_files(output)
    else:
        temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
        try:
            for name, (url, expected_bytes, expected_sha256) in TOKENIZER_FILES.items():
                destination = temporary / name
                urllib.request.urlretrieve(url, destination)
                _verify_file(destination, expected_bytes, expected_sha256)
            _write_json(
                temporary / "source_manifest.json",
                {
                    "official_sentencepiece_url": TOKENIZER_URL,
                    "official_sentencepiece_sha256": TOKENIZER_SHA256,
                    "metadata_revision": TOKENIZER_METADATA_REVISION,
                    "files": {
                        name: {
                            "url": url,
                            "bytes": expected_bytes,
                            "sha256": expected_sha256,
                        }
                        for name, (url, expected_bytes, expected_sha256) in TOKENIZER_FILES.items()
                    },
                },
            )
            os.replace(temporary, output)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    tokenizer = AutoTokenizer.from_pretrained(output, local_files_only=True, use_fast=True)
    sentencepiece_tokenizer = sentencepiece.SentencePieceProcessor(
        model_file=str(output / "tokenizer.model")
    )
    probe = "Task: stack cups, State: 0 1 2 3 4 5 6;\nAction: "
    expected = sentencepiece_tokenizer.encode(probe, add_bos=True)
    actual = tokenizer.encode(probe, add_special_tokens=True)
    if actual != expected:
        raise RuntimeError(
            "Transformers tokenizer IDs differ from the official SentencePiece model: "
            f"transformers={actual[:16]}, sentencepiece={expected[:16]}"
        )


def _load_preprocessor(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    tokenizer_steps = [
        step
        for step in payload.get("steps", [])
        if step.get("registry_name") == "tokenizer_processor"
    ]
    if len(tokenizer_steps) != 1:
        raise RuntimeError(f"{path}: expected one tokenizer_processor, got {len(tokenizer_steps)}")
    return payload


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _prepare_local_base(
    base_repo: str,
    output: Path,
    tokenizer_dir: Path,
    base_path: Path | None = None,
) -> Path:
    snapshot = (
        base_path.expanduser().resolve()
        if base_path is not None
        else Path(snapshot_download(repo_id=base_repo, repo_type="model"))
    )
    required = {
        "config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    }
    missing = sorted(name for name in required if not (snapshot / name).is_file())
    if missing:
        raise RuntimeError(f"{snapshot}: incomplete base model, missing {missing}")
    source_preprocessor = _load_preprocessor(snapshot / PREPROCESSOR_NAME)
    tokenizer_step = next(
        step
        for step in source_preprocessor["steps"]
        if step["registry_name"] == "tokenizer_processor"
    )
    tokenizer_step["config"]["tokenizer_name"] = str(tokenizer_dir.resolve())

    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        current = _load_preprocessor(output / PREPROCESSOR_NAME)
        current_step = next(
            step
            for step in current["steps"]
            if step["registry_name"] == "tokenizer_processor"
        )
        if current_step["config"].get("tokenizer_name") != str(tokenizer_dir.resolve()):
            raise RuntimeError(f"{output}: existing tokenizer override differs")
        for name in ("config.json", "model.safetensors", "policy_postprocessor.json"):
            if not (output / name).is_file():
                raise RuntimeError(f"{output}: missing {name}")
        return output

    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        for source in snapshot.iterdir():
            if source.name == PREPROCESSOR_NAME:
                continue
            (temporary / source.name).symlink_to(source.resolve())
        _write_json(temporary / PREPROCESSOR_NAME, source_preprocessor)
        _write_json(
            temporary / "tokenizer_override.json",
            {
                "base_repo": base_repo,
                "base_snapshot": str(snapshot),
                "tokenizer_url": TOKENIZER_URL,
                "tokenizer_sha256": TOKENIZER_SHA256,
                "tokenizer_bytes": TOKENIZER_BYTES,
                "tokenizer_dir": str(tokenizer_dir.resolve()),
            },
        )
        os.replace(temporary, output)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-repo", default="lerobot/pi05_base")
    parser.add_argument(
        "--base-path",
        type=Path,
        help="Verified local base snapshot; skips snapshot_download when provided",
    )
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    tokenizer_dir = args.tokenizer_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    _prepare_tokenizer(tokenizer_dir)
    local_base = _prepare_local_base(
        args.base_repo,
        output,
        tokenizer_dir,
        args.base_path,
    )
    print(
        json.dumps(
            {
                "base_repo": args.base_repo,
                "local_base": str(local_base),
                "tokenizer": str(tokenizer_dir),
                "tokenizer_sha256": TOKENIZER_SHA256,
                "token_id_probe": "matched official SentencePiece",
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
