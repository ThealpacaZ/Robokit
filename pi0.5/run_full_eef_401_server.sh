#!/usr/bin/env bash
# Full PI0.5 training of the first 200 stack-cups episodes in shaohuan1/lememory.
set -euo pipefail

: "${HF_TOKEN:?Export HF_TOKEN before launching}"
: "${WANDB_API_KEY:?Export WANDB_API_KEY before launching}"

if [[ -r /etc/network_turbo && "${NETWORK_TURBO:-true}" == "true" ]]; then
  set +u
  # shellcheck disable=SC1091
  source /etc/network_turbo
  set -u
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${ROOT:-/root/autodl-tmp}"
PI05_ENV_BIN="${PI05_ENV_BIN:-$ROOT/envs/pi05/bin}"
export PATH="$PI05_ENV_BIN:$PATH"
PYTHON_BIN="${PYTHON_BIN:-$PI05_ENV_BIN/python}"
[[ -x "$PYTHON_BIN" ]] || { echo "PI0.5 environment missing: $PYTHON_BIN" >&2; exit 1; }

SOURCE_REPO="${SOURCE_REPO:-shaohuan1/lememory}"
SOURCE_DIR="${SOURCE_DIR:-$ROOT/pi05/source/lememory_stackcups_200}"
DATASET_REPO="${DATASET_REPO:-shaohuan1/stackcups_200_eef_pi05_lerobot_ctrl_eef}"
POLICY_REPO="${POLICY_REPO:-shaohuan1/stackcups_200_eef_pi05_full_ctrl_eef}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/stackcups_200_eef_pi05_full_ctrl_eef}"
FINAL_DIR="${FINAL_DIR:-$ROOT/models/stackcups_200_eef_pi05_full_ctrl_eef}"
PROBE_DIR="${PROBE_DIR:-$ROOT/probes/stackcups_200_eef_pi05_full_ctrl_eef}"
RUN_LOG="${RUN_LOG:-$ROOT/outputs/stackcups_200_eef_pi05_full_ctrl_eef.train.log}"
MARKER="${MARKER:-$OUTPUT_DIR/converged.json}"
# Keep the Hub cache on the independent container overlay. Full checkpoints
# and the CUDA environment stay on the AutoDL data disk, avoiding a single
# disk-capacity bottleneck during full-parameter training.
HF_HOME="${HF_HOME:-/root/pi05_hf}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
SOURCE_SNAPSHOT="${SOURCE_SNAPSHOT:-$HF_HOME/hub/datasets--shaohuan1--lememory/snapshots/666ef5521cb3daf42d2f074bf4160cdaf92781bf}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$ROOT/models/paligemma_tokenizer}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/pi05_base_local}"
PROMPT_SUFFIX="${PROMPT_SUFFIX:- <control mode> eef <control mode>}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-200}"
STEPS="${STEPS:-60000}"
BATCH_SIZE="${BATCH_SIZE:-16}"
EVAL_STEPS="${EVAL_STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"

export HF_HOME HF_LEROBOT_HOME HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export WANDB_ENABLE=true WANDB_PROJECT="${WANDB_PROJECT:-stackcups_200_eef_pi05_full}"
export JOB_NAME="${JOB_NAME:-stackcups_200_eef_pi05_full_ctrl_eef}"
export EVAL_SPLIT="${EVAL_SPLIT:-0.10}" MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-4096}"
export SAVE_CHECKPOINT_TO_HUB=false
export DATASET_REPO POLICY_REPO OUTPUT_DIR

gpu_count=$(nvidia-smi -L | wc -l)
[[ "$gpu_count" -eq 1 ]] || { echo "Expected one assigned GPU, found $gpu_count" >&2; exit 1; }

for path in "$OUTPUT_DIR" "$FINAL_DIR"; do
  [[ ! -e "$path" ]] || { echo "Refusing to overwrite existing path: $path" >&2; exit 1; }
done
mkdir -p "$SOURCE_DIR" "$HF_HOME" "$(dirname "$RUN_LOG")" "$PROBE_DIR"

DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO"
if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
SOURCE_ARGS=(
  --source-repo "$SOURCE_REPO"
  --output "$SOURCE_DIR"
  --expected-episodes "$EXPECTED_EPISODES"
  --max-episodes "$EXPECTED_EPISODES"
)
if [[ -d "$SOURCE_SNAPSHOT" ]]; then
  SOURCE_ARGS+=(--snapshot-dir "$SOURCE_SNAPSHOT")
fi
"$PYTHON_BIN" "$HERE/prepare_lememory.py" "${SOURCE_ARGS[@]}"

  "$PYTHON_BIN" "$HERE/convert_hdf5_to_lerobot.py" \
    --data "$SOURCE_DIR" \
    --repo-id "$DATASET_REPO" \
    --camera cam_high \
    --arm right_arm \
    --action-space eef_delta \
    --instruction-suffix "$PROMPT_SUFFIX" \
    --push-to-hub --private --upload-large-folder
fi

"$PYTHON_BIN" "$HERE/validate_lerobot.py" \
  --repo-id "$DATASET_REPO" --root "$DATASET_ROOT" --action-space eef_delta

"$PYTHON_BIN" - "$DATASET_ROOT" "$PROMPT_SUFFIX" <<'PY'
import sys
from pathlib import Path
import pyarrow.parquet as pq

root, suffix = map(Path, sys.argv[1:2]) if False else (Path(sys.argv[1]), sys.argv[2])
table = pq.read_table(root / "meta" / "tasks.parquet")
tasks = table.column("task").to_pylist()
if not tasks or any(not str(task).endswith(suffix) for task in tasks):
    raise SystemExit("Not every dataset task prompt ends with the required control-mode suffix")
print(f"[prompt-contract] {len(tasks)} unique prompts all end with {suffix!r}")
PY

# The HDF5 snapshot is now represented by a validated, private LeRobot v3
# dataset on the Hub. Reclaim its local cache before downloading the base
# model: retaining both copies can exhaust the container overlay during a
# full-parameter run. Restrict removal to this exact, known source repo.
if [[ -f "$SOURCE_DIR/hf_source_manifest.json" ]]; then
SOURCE_SNAPSHOT=$("$PYTHON_BIN" - "$SOURCE_DIR/hf_source_manifest.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1], encoding="utf-8"))["snapshot"])
PY
)
SOURCE_CACHE_ROOT="$(dirname "$(dirname "$SOURCE_SNAPSHOT")")"
EXPECTED_SOURCE_CACHE="$HF_HOME/hub/datasets--shaohuan1--lememory"
[[ "$SOURCE_CACHE_ROOT" == "$EXPECTED_SOURCE_CACHE" ]] || {
  echo "Refusing to remove unexpected source cache: $SOURCE_CACHE_ROOT" >&2
  exit 1
}
rm -rf -- "$SOURCE_DIR" "$SOURCE_CACHE_ROOT"
echo "[disk] removed validated raw-source cache: $SOURCE_CACHE_ROOT"
fi

"$PYTHON_BIN" "$HERE/prefetch_hf.py" --dataset "$DATASET_REPO" --base lerobot/pi05_base
"$PYTHON_BIN" "$HERE/prepare_paligemma_tokenizer.py" \
  --base-repo lerobot/pi05_base --tokenizer-dir "$TOKENIZER_DIR" --output "$BASE_MODEL"

# A three-step full-parameter smoke run prevents a long training job from
# failing later due to a one-GPU OOM. On failure retry at half the batch,
# preserving the largest successful single-GPU batch for the real run.
smoke_rc=1
while [[ "$BATCH_SIZE" -ge 1 ]]; do
  PROBE_OUTPUT="$PROBE_DIR/full_smoke_bs${BATCH_SIZE}"
  set +e
  DATASET_REPO="$DATASET_REPO" POLICY_REPO="$POLICY_REPO" OUTPUT_DIR="$PROBE_OUTPUT" \
  JOB_NAME="${JOB_NAME}_smoke_bs${BATCH_SIZE}" BASE_MODEL="$BASE_MODEL" STEPS=3 \
  BATCH_SIZE="$BATCH_SIZE" NUM_PROCESSES=1 GPU_IDS=0 EVAL_STEPS=103 SAVE_FREQ=103 \
  MAX_EVAL_SAMPLES=1 NUM_WORKERS=8 WANDB_ENABLE=false WANDB_MODE=disabled \
  PUSH_TO_HUB=false SAVE_CHECKPOINT=false SAVE_CHECKPOINT_TO_HUB=false \
  bash "$HERE/train_full.sh" >"$PROBE_DIR/full_smoke_bs${BATCH_SIZE}.log" 2>&1
  smoke_rc=$?
  set -e
  if [[ "$smoke_rc" -eq 0 ]]; then
    rm -rf "$PROBE_OUTPUT"
    break
  fi
  echo "[full-eef] smoke OOM/failure at batch=$BATCH_SIZE; retrying smaller batch" >&2
  BATCH_SIZE=$((BATCH_SIZE / 2))
done
[[ "$smoke_rc" -eq 0 && "$BATCH_SIZE" -ge 1 ]] || {
  echo "Full-finetune smoke failed down to batch=1; see $PROBE_DIR" >&2
  exit 1
}

train_frames=$("$PYTHON_BIN" - "$DATASET_ROOT/meta/info.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
print(int(p["total_frames"]) * 9 // 10)
PY
)
steps_per_epoch=$(((train_frames + BATCH_SIZE - 1) / BATCH_SIZE))

echo "[full-eef] starting full PI0.5: episodes=$EXPECTED_EPISODES per_gpu_batch=$BATCH_SIZE steps=$STEPS steps_per_epoch=$steps_per_epoch"
set +e
setsid env \
  DATASET_REPO="$DATASET_REPO" POLICY_REPO="$POLICY_REPO" OUTPUT_DIR="$OUTPUT_DIR" \
  JOB_NAME="$JOB_NAME" BASE_MODEL="$BASE_MODEL" STEPS="$STEPS" BATCH_SIZE="$BATCH_SIZE" \
  NUM_PROCESSES=1 GPU_IDS=0 EVAL_STEPS="$EVAL_STEPS" SAVE_FREQ="$SAVE_FREQ" \
  MAX_EVAL_SAMPLES="$MAX_EVAL_SAMPLES" WANDB_ENABLE=true WANDB_MODE=online \
  PUSH_TO_HUB=false SAVE_CHECKPOINT=true SAVE_CHECKPOINT_TO_HUB=false \
  bash "$HERE/train_full.sh" >"$RUN_LOG" 2>&1 &
train_pid=$!

"$PYTHON_BIN" "$HERE/monitor_full_finetune.py" \
  --log "$RUN_LOG" --output-dir "$OUTPUT_DIR" --pid "$train_pid" --marker "$MARKER" \
  --final-dir "$FINAL_DIR" --repo-id "$POLICY_REPO" --steps-per-epoch "$steps_per_epoch" \
  --batch-size "$BATCH_SIZE" --num-processes 1 --min-steps 30000 --min-evals 6 \
  --patience 3 --min-relative-improvement 0.005 --trend-window 3 \
  --trend-relative-improvement 0.005 --max-checkpoints 2 --poll-interval 1800
monitor_rc=$?
wait "$train_pid"
train_rc=$?
set -e

if [[ "$monitor_rc" -ne 0 || ! -f "$MARKER" ]]; then
  printf 'status=not_converged\ntrain_rc=%s\nmonitor_rc=%s\n' "$train_rc" "$monitor_rc" >"$OUTPUT_DIR.NOT_CONVERGED"
  echo "[full-eef] training ended without convergence evidence; server remains on" >&2
  exit 2
fi

"$PYTHON_BIN" -m wandb sync --sync-all "$OUTPUT_DIR/wandb" || true
printf 'status=converged\ntrain_rc=%s\nmonitor_rc=%s\nmarker=%s\n' \
  "$train_rc" "$monitor_rc" "$MARKER" >"$OUTPUT_DIR/COMPLETED"
sync
echo "[full-eef] converged after >=30k steps, verified and published; shutting down"
/usr/bin/shutdown -h now
