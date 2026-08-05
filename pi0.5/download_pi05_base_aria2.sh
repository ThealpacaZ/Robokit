#!/usr/bin/env bash
# Download the pinned PI0.5 base weight with the server's fastest verified path.
set -euo pipefail

HF_HOME="${HF_HOME:-/root/pi05_hf}"
REVISION="${REVISION:-b211f3d44c36b6acfcf7ae94a64e8e96f75a64ba}"
EXPECTED_SIZE=14467165872
EXPECTED_SHA256=0eb11ca9587678c1d2ef8cf32807c29f8ce53a2bfdfc1aa4a4c96f16fca59b0f
REPO_CACHE="$HF_HOME/hub/models--lerobot--pi05_base"
BLOB="$REPO_CACHE/blobs/$EXPECTED_SHA256"
PART="$BLOB.aria2-download"
SNAPSHOT="$REPO_CACHE/snapshots/$REVISION"
URL="https://huggingface.co/lerobot/pi05_base/resolve/$REVISION/model.safetensors?download=true"

command -v aria2c >/dev/null || {
  echo "aria2c is required" >&2
  exit 1
}
mkdir -p "$REPO_CACHE/blobs" "$SNAPSHOT" "$REPO_CACHE/refs"

verify_weight() {
  local path="$1"
  local size digest
  size=$(stat -c '%s' "$path")
  [[ "$size" -eq "$EXPECTED_SIZE" ]] || {
    echo "size mismatch: expected=$EXPECTED_SIZE actual=$size path=$path" >&2
    return 1
  }
  digest=$(sha256sum "$path" | awk '{print $1}')
  [[ "$digest" == "$EXPECTED_SHA256" ]] || {
    echo "SHA-256 mismatch: expected=$EXPECTED_SHA256 actual=$digest path=$path" >&2
    return 1
  }
}

if [[ -f "$BLOB" ]] && verify_weight "$BLOB"; then
  echo "[pi05-base] verified blob already present: $BLOB"
else
  rm -f -- "$BLOB"
  if [[ -r /etc/network_turbo && "${NETWORK_TURBO:-true}" == "true" ]]; then
    set +u
    # shellcheck disable=SC1091
    source /etc/network_turbo
    set -u
  fi

  echo "[pi05-base] aria2c x16 via network_turbo: $URL"
  aria2c \
    --continue=true \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=1M \
    --file-allocation=none \
    --auto-file-renaming=false \
    --allow-overwrite=true \
    --max-tries=0 \
    --retry-wait=5 \
    --connect-timeout=30 \
    --timeout=120 \
    --lowest-speed-limit=1K \
    --summary-interval=10 \
    --console-log-level=notice \
    --checksum="sha-256=$EXPECTED_SHA256" \
    --dir="$(dirname "$PART")" \
    --out="$(basename "$PART")" \
    "$URL"

  verify_weight "$PART"
  mv -f -- "$PART" "$BLOB"
fi

ln -sfn "../../blobs/$EXPECTED_SHA256" "$SNAPSHOT/model.safetensors"
printf '%s' "$REVISION" >"$REPO_CACHE/refs/main"
echo "[pi05-base] HF cache ready: $SNAPSHOT/model.safetensors"
