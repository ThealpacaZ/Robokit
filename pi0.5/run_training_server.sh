#!/usr/bin/env bash
set -euo pipefail

# End-to-end AutoDL job. Secrets must be provided through the environment and
# are never read from or written into the repository.
: "${HF_TOKEN:?Export HF_TOKEN before launching}"
: "${WANDB_API_KEY:?Export WANDB_API_KEY before launching}"

# AutoDL provides an optional download accelerator. It only changes network
# routing; all caches and outputs below remain on the data disk.
if [[ -r /etc/network_turbo && "${NETWORK_TURBO:-true}" == "true" ]]; then
  # Some provider versions reference unset shell variables.
  set +u
  # shellcheck disable=SC1091
  source /etc/network_turbo
  set -u
fi

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
SOURCE_REPO="${SOURCE_REPO:-shaohuan1/lememory}"
SOURCE_DIR="${SOURCE_DIR:-/root/autodl-tmp/pi05/source/stackcups}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/hf/lerobot}"
DATASET_REPO="${DATASET_REPO:-shaohuan1/stackcups_one_task_lerobot}"
POLICY_REPO="${POLICY_REPO:-shaohuan1/stackcups_one_task}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/autodl-tmp/outputs/stackcups_one_task}"
RUN_LOG="${RUN_LOG:-/root/autodl-tmp/outputs/stackcups_one_task.train.log}"
CONVERGED_MARKER="${CONVERGED_MARKER:-$OUTPUT_DIR/converged.json}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-true}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/root/autodl-tmp/models/paligemma_tokenizer}"
LOCAL_BASE_MODEL="${LOCAL_BASE_MODEL:-/root/autodl-tmp/models/pi05_base_local}"

export HF_HOME HF_LEROBOT_HOME HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export DATASET_REPO POLICY_REPO OUTPUT_DIR
export WANDB_ENABLE=true
export WANDB_PROJECT="${WANDB_PROJECT:-stackcups_one_task}"
export JOB_NAME="${JOB_NAME:-stackcups_one_task}"
export EVAL_SPLIT="${EVAL_SPLIT:-0.10}"
export EVAL_STEPS="${EVAL_STEPS:-1000}"
export MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2048}"
export SAVE_FREQ="${SAVE_FREQ:-1000}"
export LOG_FREQ="${LOG_FREQ:-50}"
export SAVE_CHECKPOINT_TO_HUB="${SAVE_CHECKPOINT_TO_HUB:-true}"

mkdir -p "$SOURCE_DIR" "$HF_HOME" "$HF_LEROBOT_HOME" "$(dirname "$RUN_LOG")"

"$PYTHON_BIN" "$HERE/prepare_lememory.py" \
  --source-repo "$SOURCE_REPO" \
  --output "$SOURCE_DIR" \
  --expected-episodes 202

if [[ ! -f "$HF_LEROBOT_HOME/$DATASET_REPO/meta/info.json" ]]; then
  "$PYTHON_BIN" "$HERE/convert_hdf5_to_lerobot.py" \
    --data "$SOURCE_DIR" \
    --repo-id "$DATASET_REPO" \
    --camera cam_high \
    --arm right_arm \
    --instruction "stack cups" \
    --push-to-hub \
    --private \
    --upload-large-folder
fi

"$PYTHON_BIN" "$HERE/validate_lerobot.py" \
  --repo-id "$DATASET_REPO" \
  --root "$HF_LEROBOT_HOME/$DATASET_REPO"

"$PYTHON_BIN" "$HERE/prefetch_hf.py" \
  --dataset "$DATASET_REPO" \
  --base lerobot/pi05_base

"$PYTHON_BIN" "$HERE/prepare_paligemma_tokenizer.py" \
  --base-repo lerobot/pi05_base \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output "$LOCAL_BASE_MODEL"
export BASE_MODEL="$LOCAL_BASE_MODEL"

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "[server-train] refusing to overwrite existing OUTPUT_DIR=$OUTPUT_DIR" >&2
  exit 1
fi

set +e
stdbuf -oL -eL bash "$HERE/train_lora.sh" >"$RUN_LOG" 2>&1 &
train_pid=$!
"$PYTHON_BIN" "$HERE/monitor_convergence.py" \
  --log "$RUN_LOG" \
  --output-dir "$OUTPUT_DIR" \
  --pid "$train_pid" \
  --marker "$CONVERGED_MARKER" \
  --min-steps "${CONVERGENCE_MIN_STEPS:-10000}" \
  --min-evals "${CONVERGENCE_MIN_EVALS:-6}" \
  --patience "${CONVERGENCE_PATIENCE:-4}" \
  --min-relative-improvement "${CONVERGENCE_RELATIVE_DELTA:-0.005}" &
monitor_pid=$!

wait "$train_pid"
train_rc=$?
wait "$monitor_pid"
monitor_rc=$?
set -e

if [[ -f "$CONVERGED_MARKER" ]]; then
  read -r stop_step publish_step < <("$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); print(p["stop_step"], p.get("publish_step", p["best_step"]))' \
    "$CONVERGED_MARKER")
  checkpoint="$OUTPUT_DIR/checkpoints/$(printf '%06d' "$publish_step")"
  "$PYTHON_BIN" "$HERE/publish_checkpoint.py" \
    --checkpoint "$checkpoint" \
    --repo-id "$POLICY_REPO" \
    --private

  # Runs are online; this additionally flushes any recoverable offline files.
  "$PYTHON_BIN" -m wandb sync --sync-all "$OUTPUT_DIR/wandb" || true
  printf '%s\n' \
    "status=converged" \
    "checkpoint=$checkpoint" \
    "stop_step=$stop_step" \
    "publish_step=$publish_step" \
    "policy_repo=$POLICY_REPO" \
    "train_rc=$train_rc" \
    "monitor_rc=$monitor_rc" \
    >"$OUTPUT_DIR/COMPLETED"
  sync

  if [[ "$AUTO_SHUTDOWN" == "true" ]]; then
    echo "[server-train] converged checkpoint published; shutting down server"
    /usr/bin/shutdown -h now
  fi
  exit 0
fi

printf '%s\n' \
  "status=not_converged" \
  "train_rc=$train_rc" \
  "monitor_rc=$monitor_rc" \
  >"${OUTPUT_DIR}.NOT_CONVERGED"
echo "[server-train] training ended without convergence marker; server stays on" >&2
exit 2
