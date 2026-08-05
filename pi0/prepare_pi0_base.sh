#!/usr/bin/env bash
set -euo pipefail

# Pinned public LeRobot PI0 base. aria2 is used because the AutoDL host's
# single-stream Hugging Face download is much slower; every downloaded byte is
# still verified against the Hub LFS digest/revision.
REVISION="${PI0_BASE_REVISION:-25c379b52ba2ff8788cab921758a3cc3fe3f77f2}"
OUTPUT="${1:-/root/autodl-tmp/models/pi0_base_raw}"
MODEL_BYTES=14005618584
MODEL_SHA256=8229fd9a7c3c2aafc1e223567b61b5fe3e25eef873bb4233928dbee4bd836303

if ! command -v aria2c >/dev/null; then
  echo "[pi0-base] aria2c is required for the verified multi-stream download" >&2
  exit 1
fi

mkdir -p "$OUTPUT"
base_url="https://huggingface.co/lerobot/pi0_base/resolve/$REVISION"

download() {
  local name="$1"
  aria2c --continue=true --max-connection-per-server=16 --split=16 \
    --min-split-size=1M --file-allocation=none --dir="$OUTPUT" \
    --out="$name" "$base_url/$name?download=true"
}

download model.safetensors
download config.json
download policy_preprocessor.json
download policy_postprocessor.json

actual_bytes="$(stat -c %s "$OUTPUT/model.safetensors")"
if [[ "$actual_bytes" != "$MODEL_BYTES" ]]; then
  echo "[pi0-base] model size $actual_bytes != $MODEL_BYTES" >&2
  exit 2
fi

printf '%s  %s\n' \
  "$MODEL_SHA256" "$OUTPUT/model.safetensors" \
  "700c59d206885faff307d63272558fb9ad9a6b9feecdae55626ed9e3f594feb9" "$OUTPUT/config.json" \
  "7bae7494caaf887fbcf5e09052d72bed5ef09272eb296202708298802a02a2c5" "$OUTPUT/policy_preprocessor.json" \
  "b9dd462a5ad3c5add0329460e88c80f76153b2306820e6fae1e7d5fc5fe11474" "$OUTPUT/policy_postprocessor.json" \
  | sha256sum --check

printf '%s\n' \
  "repo=lerobot/pi0_base" \
  "revision=$REVISION" \
  "model_bytes=$MODEL_BYTES" \
  "model_sha256=$MODEL_SHA256" \
  >"$OUTPUT/source_manifest.txt"
echo "[pi0-base] verified $OUTPUT"
