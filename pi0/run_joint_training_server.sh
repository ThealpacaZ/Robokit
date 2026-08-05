#!/usr/bin/env bash
set -euo pipefail

# End-to-end AutoDL job for PI0 joint-angle LoRA. Secrets are injected through
# the environment and are never copied into code or logs.
: "${HF_TOKEN:?Export HF_TOKEN before launching}"
: "${WANDB_API_KEY:?Export WANDB_API_KEY before launching}"

if [[ -r /etc/network_turbo && "${NETWORK_TURBO:-true}" == "true" ]]; then
  set +u
  # shellcheck disable=SC1091
  source /etc/network_turbo
  set -u
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUPPORT_DIR="$HERE/../pi0.5"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_REPO="${SOURCE_REPO:-shaohuan1/lememory}"
SOURCE_DIR="${SOURCE_DIR:-/root/autodl-tmp/pi05/source/stackcups}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/hf/lerobot}"
DATASET_REPO="${DATASET_REPO:-shaohuan1/stackcups_joint_pi0_lerobot}"
POLICY_REPO="${POLICY_REPO:-shaohuan1/stackcups_joint_pi0_h30}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/outputs/stackcups_joint_pi0_h30}"
RUN_LOG="${RUN_LOG:-/root/autodl-tmp/outputs/stackcups_joint_pi0_h30.train.log}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/root/autodl-tmp/models/paligemma_tokenizer}"
LOCAL_BASE_MODEL="${LOCAL_BASE_MODEL:-/root/autodl-tmp/models/pi0_base_local}"
BASE_REPO="${BASE_REPO:-lerobot/pi0_base}"
RAW_BASE_MODEL="${RAW_BASE_MODEL:-/root/autodl-tmp/models/pi0_base_raw}"
STEPS="${STEPS:-30000}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-30}"

if [[ "$STEPS" != "30000" || "$CHUNK_SIZE" != "50" || "$EXECUTION_HORIZON" != "30" ]]; then
  echo "[pi0-server-train] expected STEPS=30000, CHUNK_SIZE=50, EXECUTION_HORIZON=30; got $STEPS/$CHUNK_SIZE/$EXECUTION_HORIZON" >&2
  exit 2
fi

export HF_HOME HF_LEROBOT_HOME HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export DATASET_REPO POLICY_REPO OUTPUT_DIR STEPS CHUNK_SIZE EXECUTION_HORIZON
export WANDB_ENABLE=true
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-stackcups_joint_pi0_h30}"
export JOB_NAME="${JOB_NAME:-stackcups_joint_pi0_h30}"
export EVAL_SPLIT="${EVAL_SPLIT:-0.10}"
export EVAL_STEPS="${EVAL_STEPS:-1000}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2048}"
export SAVE_FREQ="${SAVE_FREQ:-1000}"
export LOG_FREQ="${LOG_FREQ:-50}"
export SAVE_CHECKPOINT_TO_HUB="${SAVE_CHECKPOINT_TO_HUB:-false}"

mkdir -p "$SOURCE_DIR" "$HF_HOME" "$HF_LEROBOT_HOME" "$(dirname "$RUN_LOG")"

"$PYTHON_BIN" "$SUPPORT_DIR/prepare_lememory.py" \
  --source-repo "$SOURCE_REPO" \
  --output "$SOURCE_DIR" \
  --expected-episodes 202

dataset_root="$HF_LEROBOT_HOME/$DATASET_REPO"
if [[ ! -f "$dataset_root/meta/info.json" ]]; then
  "$PYTHON_BIN" "$SUPPORT_DIR/convert_hdf5_to_lerobot.py" \
    --data "$SOURCE_DIR" \
    --repo-id "$DATASET_REPO" \
    --camera cam_high \
    --arm right_arm \
    --instruction "stack cups" \
    --action-space joint \
    --push-to-hub \
    --private \
    --upload-large-folder
fi

"$PYTHON_BIN" "$SUPPORT_DIR/validate_lerobot.py" \
  --repo-id "$DATASET_REPO" \
  --root "$dataset_root" \
  --action-space joint

bash "$HERE/prepare_pi0_base.sh" "$RAW_BASE_MODEL"

"$PYTHON_BIN" "$SUPPORT_DIR/prepare_paligemma_tokenizer.py" \
  --base-repo "$BASE_REPO" \
  --base-path "$RAW_BASE_MODEL" \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output "$LOCAL_BASE_MODEL"
export BASE_MODEL="$LOCAL_BASE_MODEL"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "[pi0-server-train] refusing to overwrite OUTPUT_DIR=$OUTPUT_DIR" >&2
  exit 1
fi

set +e
stdbuf -oL -eL bash "$HERE/train_joint_lora.sh" >"$RUN_LOG" 2>&1
train_rc=$?
set -e

if [[ "$train_rc" -ne 0 ]]; then
  printf '%s\n' \
    "status=failed" \
    "train_rc=$train_rc" \
    "steps=$STEPS" \
    "chunk_size=$CHUNK_SIZE" \
    "execution_horizon=$EXECUTION_HORIZON" \
    >"${OUTPUT_DIR}.FAILED"
  echo "[pi0-server-train] training failed; EEF service and server remain on" >&2
  exit "$train_rc"
fi

checkpoint="$OUTPUT_DIR/checkpoints/$(printf '%06d' "$STEPS")"
"$PYTHON_BIN" "$SUPPORT_DIR/publish_checkpoint.py" \
  --checkpoint "$checkpoint" \
  --repo-id "$POLICY_REPO" \
  --private

"$PYTHON_BIN" -m wandb sync --sync-all "$OUTPUT_DIR/wandb" || true
printf '%s\n' \
  "status=completed" \
  "checkpoint=$checkpoint" \
  "steps=$STEPS" \
  "chunk_size=$CHUNK_SIZE" \
  "execution_horizon=$EXECUTION_HORIZON" \
  "action_space=joint" \
  "policy_repo=$POLICY_REPO" \
  >"$OUTPUT_DIR/COMPLETED"
sync
echo "[pi0-server-train] complete; server intentionally remains on for EEF testing"
