#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?Export HF_TOKEN before launching}"
: "${WANDB_API_KEY:?Export WANDB_API_KEY before launching}"
CURRENT_TRAIN_PID="${CURRENT_TRAIN_PID:-}"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PI05_ENV_BIN="${PI05_ENV_BIN:-/root/autodl-tmp/envs/pi05/bin}"
PYTHON_BIN="${PYTHON_BIN:-$PI05_ENV_BIN/python}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs}"
FINAL_ROOT="${FINAL_ROOT:-/root/autodl-tmp/models/full_finetunes}"
OUTPUT_DIR="$OUTPUT_ROOT/stackcups_202_400_eef_pi05_full"
FINAL_DIR="$FINAL_ROOT/stackcups_202_400_eef_pi05_full"
TRAIN_LOG="$OUTPUT_ROOT/stackcups_202_400_eef_pi05_full.train.log"
MARKER="$OUTPUT_DIR/converged.json"
DATASET_REPO="shaohuan1/stackcups_202_400_eef_pi05_lerobot"
POLICY_REPO="shaohuan1/stackcups_202_400_eef_pi05_full"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/pi05_base_local}"
WANDB_PROJECT="${WANDB_PROJECT:-stackcups_pi05_full}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_PROCESSES=2
EPOCH_STEPS=1168
TRANSITION_STEP="${TRANSITION_STEP:-2336}"
RESUME_STEP="${RESUME_STEP:-$TRANSITION_STEP}"
MIN_STEPS="${MIN_STEPS:-10000}"
HARD_MAX_STEPS="${HARD_MAX_STEPS:-20000}"
MAX_EVAL_SAMPLES=4150
RESUME_CHECKPOINT="$OUTPUT_DIR/checkpoints/$(printf '%06d' "$RESUME_STEP")"
MONITOR_STATE_FILE="${MONITOR_STATE_FILE:-$OUTPUT_DIR/monitor_state.json}"

export PATH="$PI05_ENV_BIN:$PATH"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
export HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export WANDB_PROJECT BASE_MODEL NUM_PROCESSES

checkpoint_is_stable() {
  local checkpoint="$1"
  local optimizer="$checkpoint/training_state/optimizer_state.safetensors"
  local first_size second_size
  local required=(
    "$checkpoint/pretrained_model/model.safetensors"
    "$checkpoint/pretrained_model/config.json"
    "$checkpoint/pretrained_model/train_config.json"
    "$checkpoint/pretrained_model/policy_preprocessor.json"
    "$checkpoint/pretrained_model/policy_postprocessor.json"
    "$checkpoint/training_state/training_step.json"
    "$optimizer"
    "$checkpoint/training_state/optimizer_param_groups.json"
    "$checkpoint/training_state/rng_state.safetensors"
    "$checkpoint/training_state/scheduler_state.json"
  )
  local path
  for path in "${required[@]}"; do
    [[ -s "$path" ]] || return 1
  done
  first_size="$(stat -c '%s' "$optimizer")"
  [[ "$first_size" -gt 1000000000 ]] || return 1
  sleep 10
  second_size="$(stat -c '%s' "$optimizer")"
  [[ "$first_size" -eq "$second_size" ]]
}

if [[ -n "$CURRENT_TRAIN_PID" ]]; then
  echo "[eef-20k] waiting for resumable checkpoint $RESUME_STEP"
  while ! checkpoint_is_stable "$RESUME_CHECKPOINT"; do
    if ! kill -0 "$CURRENT_TRAIN_PID" 2>/dev/null; then
      echo "[eef-20k] active trainer exited before checkpoint $RESUME_STEP" >&2
      exit 1
    fi
    sleep 30
  done

  echo "[eef-20k] checkpoint $RESUME_STEP is complete; switching to 20k schedule"
  kill -TERM -- "-$CURRENT_TRAIN_PID" 2>/dev/null || kill -TERM "$CURRENT_TRAIN_PID" 2>/dev/null || true
  for _ in $(seq 1 60); do
    kill -0 "$CURRENT_TRAIN_PID" 2>/dev/null || break
    sleep 2
  done
  if kill -0 "$CURRENT_TRAIN_PID" 2>/dev/null; then
    echo "[eef-20k] trainer did not stop after SIGTERM" >&2
    exit 1
  fi
elif ! checkpoint_is_stable "$RESUME_CHECKPOINT"; then
  echo "[eef-20k] recovery checkpoint $RESUME_STEP is incomplete" >&2
  exit 1
fi

# The first checkpoint no longer contains optimizer state and is superseded by
# the fully resumable transition checkpoint.
find "$OUTPUT_DIR/checkpoints" -mindepth 1 -maxdepth 1 -type d -name '[0-9]*' \
  ! -name "$(printf '%06d' "$RESUME_STEP")" -exec rm -rf {} +

printf '\n[eef-20k] resume from step %s; min=%s hard_max=%s\n' \
  "$RESUME_STEP" "$MIN_STEPS" "$HARD_MAX_STEPS" >>"$TRAIN_LOG"

set +e
setsid env \
  DATASET_REPO="$DATASET_REPO" \
  POLICY_REPO="$POLICY_REPO" \
  OUTPUT_DIR="$OUTPUT_DIR" \
  JOB_NAME="stackcups_202_400_eef_pi05_full_bs${BATCH_SIZE}_2gpu" \
  BASE_MODEL="$BASE_MODEL" \
  RESUME_CONFIG_PATH="$RESUME_CHECKPOINT/pretrained_model/train_config.json" \
  RESUME_SCHEDULER_WARMUP_STEPS=667 \
  RESUME_SCHEDULER_DECAY_STEPS="$HARD_MAX_STEPS" \
  STEPS="$HARD_MAX_STEPS" \
  BATCH_SIZE="$BATCH_SIZE" \
  EVAL_STEPS="$EPOCH_STEPS" \
  SAVE_FREQ="$EPOCH_STEPS" \
  MAX_EVAL_SAMPLES="$MAX_EVAL_SAMPLES" \
  WANDB_ENABLE=true \
  WANDB_MODE=online \
  PUSH_TO_HUB=false \
  SAVE_CHECKPOINT=true \
  SAVE_CHECKPOINT_TO_HUB=false \
  bash "$HERE/train_full.sh" >>"$TRAIN_LOG" 2>&1 &
train_pid=$!

"$PYTHON_BIN" "$HERE/checkpoint_janitor.py" \
  --checkpoints-dir "$OUTPUT_DIR/checkpoints" \
  --trainer-pid "$train_pid" \
  --max-retained 2 \
  --poll-interval 15 \
  >>"$OUTPUT_ROOT/stackcups_202_400_eef_pi05_full.janitor.log" 2>&1 &
janitor_pid=$!

"$PYTHON_BIN" "$HERE/monitor_full_finetune.py" \
  --log "$TRAIN_LOG" \
  --output-dir "$OUTPUT_DIR" \
  --pid "$train_pid" \
  --marker "$MARKER" \
  --final-dir "$FINAL_DIR" \
  --repo-id "$POLICY_REPO" \
  --steps-per-epoch "$EPOCH_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --num-processes "$NUM_PROCESSES" \
  --min-steps "$MIN_STEPS" \
  --min-evals 5 \
  --patience 3 \
  --min-relative-improvement 0.005 \
  --trend-window 3 \
  --trend-relative-improvement 0.005 \
  --hard-max-steps "$HARD_MAX_STEPS" \
  --state-file "$MONITOR_STATE_FILE" \
  --max-checkpoints 3
monitor_rc=$?
wait "$train_pid"
train_rc=$?
wait "$janitor_pid"
janitor_rc=$?
set -e

if [[ "$monitor_rc" -ne 0 || "$janitor_rc" -ne 0 || ! -f "$MARKER" ]]; then
  printf 'status=failed\ntrain_rc=%s\nmonitor_rc=%s\njanitor_rc=%s\n' \
    "$train_rc" "$monitor_rc" "$janitor_rc" >"$OUTPUT_DIR.FAILED"
  echo "[eef-20k] training did not reach a publishable plateau or hard limit" >&2
  exit 1
fi

"$PYTHON_BIN" -m wandb sync --sync-all "$OUTPUT_DIR/wandb" || true
"$PYTHON_BIN" "$HERE/verify_wandb_run.py" \
  --marker "$MARKER" \
  --repair-missing-evals
[[ -s "$FINAL_DIR/model.safetensors" ]]
printf 'status=complete\ntrain_rc=%s\nmonitor_rc=%s\n' \
  "$train_rc" "$monitor_rc" >"$OUTPUT_DIR/COMPLETED"

"$PYTHON_BIN" - "$OUTPUT_ROOT" "$FINAL_ROOT" <<'PY'
import json,sys
from pathlib import Path
outputs,finals=map(Path,sys.argv[1:])
slugs=["stackcups_150_400_joint_pi05_full","stackcups_202_400_eef_pi05_full"]
summary={}
for slug in slugs:
    marker=outputs/slug/"converged.json"
    final=finals/slug/"model.safetensors"
    payload=json.loads(marker.read_text())
    if payload.get("status")!="converged" or "wandb_verification" not in payload or not final.is_file():
        raise SystemExit(f"incomplete final gate for {slug}")
    summary[slug]={"marker":str(marker),"final":str(final),"publish":payload["publish"],"wandb":payload["wandb_verification"]}
(outputs/"stackcups_pi05_full_two_tasks.COMPLETED.json").write_text(
    json.dumps(summary,ensure_ascii=False,indent=2)+"\n"
)
PY
sync
echo "[eef-20k] both tasks complete; shutting down"
/usr/bin/shutdown -h now
