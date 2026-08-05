#!/usr/bin/env bash
# Full-parameter PI0.5 fine-tune on the 201 "Stack one cup on top of another
# cup" episodes (shaohuan1/Memoryvla, folder Stack_one_cup_on_top_of_another_cup_b0).
#
# The raw HDF5 episodes are downloaded and staged separately (aria2c + academic
# accelerator, the only fast path on this box); this script starts from the
# staged directory, so it never re-downloads 34.5 GB on a resume.
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

# ACTION_SPACE selects the whole contract: eef_delta (local EEF deltas) or
# joint (next-frame absolute joint targets in radians). CONTROL_MODE is the
# token the deployment side reads back out of the prompt, so the two must never
# drift apart — everything below is derived from this one pair.
ACTION_SPACE="${ACTION_SPACE:-eef_delta}"
case "$ACTION_SPACE" in
  eef_delta) CONTROL_MODE="${CONTROL_MODE:-eef}" ;;
  joint)     CONTROL_MODE="${CONTROL_MODE:-joint}" ;;
  *) echo "Unsupported ACTION_SPACE=$ACTION_SPACE (use eef_delta or joint)" >&2; exit 1 ;;
esac
RUN_TAG="${RUN_TAG:-stack_one_cup_201_${CONTROL_MODE}_pi05_full}"

SOURCE_DIR="${SOURCE_DIR:-$ROOT/pi05/stage201}"
RAW_DIR="${RAW_DIR:-$ROOT/pi05/src201}"
DATASET_REPO="${DATASET_REPO:-shaohuan1/stack_one_cup_201_${CONTROL_MODE}_pi05_lerobot}"
POLICY_REPO="${POLICY_REPO:-shaohuan1/$RUN_TAG}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/$RUN_TAG}"
FINAL_DIR="${FINAL_DIR:-$ROOT/models/$RUN_TAG}"
PROBE_DIR="${PROBE_DIR:-$ROOT/probes/$RUN_TAG}"
RUN_LOG="${RUN_LOG:-$ROOT/outputs/$RUN_TAG.train.log}"
MARKER="${MARKER:-$OUTPUT_DIR/converged.json}"

# The PI0.5 base is already cached on the data disk; keeping HF_HOME there
# avoids re-downloading 14.4 GB and keeps the 30 GB container overlay free.
HF_HOME="${HF_HOME:-$ROOT/hf}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-$HF_HOME/lerobot}"
TOKENIZER_DIR="${TOKENIZER_DIR:-$ROOT/models/paligemma_tokenizer}"
BASE_SNAPSHOT="${BASE_SNAPSHOT:-$HF_HOME/hub/models--lerobot--pi05_base/snapshots/7de663972b7817d2c4cf2d84c821153dfea772e9}"
BASE_MODEL="${BASE_MODEL:-$ROOT/models/pi05_base_201_local}"

PROMPT_SUFFIX="${PROMPT_SUFFIX:- <control mode> $CONTROL_MODE <control mode>}"
EXPECTED_EPISODES="${EXPECTED_EPISODES:-201}"
STEPS="${STEPS:-60000}"
MIN_STEPS="${MIN_STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
# Keep save_freq == eval_steps: the monitor pairs each eval_loss with the
# checkpoint written at the same step to pick the best one.
EVAL_STEPS="${EVAL_STEPS:-5000}"
SAVE_FREQ="${SAVE_FREQ:-5000}"
MAX_RETAINED="${MAX_RETAINED:-1}"
POLL_INTERVAL="${POLL_INTERVAL:-1800}"

export HF_HOME HF_LEROBOT_HOME HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export WANDB_ENABLE=true WANDB_PROJECT="${WANDB_PROJECT:-stack_one_cup_201_${CONTROL_MODE}_pi05_full}"
export JOB_NAME="${JOB_NAME:-$RUN_TAG}"
export EVAL_SPLIT="${EVAL_SPLIT:-0.10}" MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-4096}"
export SAVE_CHECKPOINT_TO_HUB=false
export DATASET_REPO POLICY_REPO OUTPUT_DIR

gpu_count=$(nvidia-smi -L | wc -l)
[[ "$gpu_count" -eq 1 ]] || { echo "Expected one assigned GPU, found $gpu_count" >&2; exit 1; }

mkdir -p "$(dirname "$RUN_LOG")" "$PROBE_DIR" "$ROOT/models"

# ---------------------------------------------------------------- dataset ----
DATASET_ROOT="$HF_LEROBOT_HOME/$DATASET_REPO"
if [[ ! -f "$DATASET_ROOT/meta/info.json" ]]; then
  staged=$(ls -1 "$SOURCE_DIR"/*.hdf5 2>/dev/null | wc -l)
  [[ "$staged" -eq "$EXPECTED_EPISODES" ]] || {
    echo "Expected $EXPECTED_EPISODES staged episodes in $SOURCE_DIR, found $staged" >&2
    exit 1
  }
  # Kept local: uploading ~4 GB at this box's 2.5 MB/s uplink would delay the
  # run by half an hour for no training benefit. Publish after training starts.
  "$PYTHON_BIN" "$HERE/convert_hdf5_to_lerobot.py" \
    --data "$SOURCE_DIR" \
    --repo-id "$DATASET_REPO" \
    --camera cam_high \
    --arm right_arm \
    --action-space "$ACTION_SPACE" \
    --instruction-suffix "$PROMPT_SUFFIX" \
    --image-writer-threads "${IMAGE_WRITER_THREADS:-32}"
fi

"$PYTHON_BIN" "$HERE/validate_lerobot.py" \
  --repo-id "$DATASET_REPO" --root "$DATASET_ROOT" --action-space "$ACTION_SPACE"

"$PYTHON_BIN" - "$DATASET_ROOT" "$PROMPT_SUFFIX" <<'PY'
import sys
from pathlib import Path
import pyarrow.parquet as pq

root, suffix = Path(sys.argv[1]), sys.argv[2]
table = pq.read_table(root / "meta" / "tasks.parquet")
tasks = table.column("task").to_pylist()
if not tasks or any(not str(task).endswith(suffix) for task in tasks):
    raise SystemExit("Not every dataset task prompt ends with the required control-mode suffix")
print(f"[prompt-contract] {len(tasks)} unique prompts all end with {suffix!r}")
for task in tasks:
    print(f"[prompt-contract]   {task!r}")
PY

# The validated LeRobot dataset fully represents the raw episodes. Reclaim the
# 34.5 GB of HDF5 so full-parameter checkpoints have room on the data disk.
if [[ "${KEEP_RAW:-false}" != "true" && -d "$RAW_DIR" ]]; then
  case "$RAW_DIR" in
    "$ROOT"/pi05/src201) rm -rf -- "$RAW_DIR" "$SOURCE_DIR"
      echo "[disk] removed raw HDF5 source after dataset validation: $RAW_DIR" ;;
    *) echo "Refusing to remove unexpected raw dir: $RAW_DIR" >&2; exit 1 ;;
  esac
fi

# ------------------------------------------------------------------- base ----
"$PYTHON_BIN" "$HERE/prepare_paligemma_tokenizer.py" \
  --base-repo lerobot/pi05_base --base-path "$BASE_SNAPSHOT" \
  --tokenizer-dir "$TOKENIZER_DIR" --output "$BASE_MODEL"

# ------------------------------------------------------------------ smoke ----
# The smoke run must exercise *eval and checkpoint save*, not just training
# steps. The 2026-08-04 01:55 run trained 5,000 clean steps and then died at its
# first eval: torch.compile recompiled for the eval shapes, Triton autotuning
# blew past this GPU's 101,376-byte shared-memory limit, and the fallback kernel
# hit an illegal memory access. A steps-only smoke cannot see that, and because
# the crash landed before the first save, all 5,000 steps were lost.
#
# COMPILE_MODEL is off by default: inductor codegen is not reliable on this
# sm_120 card. Set it to true to buy back ~30% step time (1.95s vs 2.86s) at the
# risk of the eval-time crash above; the caller is then responsible for falling
# back. Either way the smoke below now proves the eval path before committing.
COMPILE_MODEL="${COMPILE_MODEL:-false}"
export COMPILE_MODEL

# The smoke deliberately asks for a NON-multiple of the batch size
# (2*batch + 5) so the eval loader must produce a short final batch. That extra
# shape is what makes inductor recompile mid-eval, and it is the most likely
# trigger of the 2026-08-04 crash — the old smoke used 64 samples, which divides
# evenly by 32 and therefore never exercised it.
#
# A compile blow-up and a real OOM both surface as a non-zero smoke exit, but
# only OOM is fixed by a smaller batch. Halving the batch on an inductor fault
# just burns four more smoke runs before failing at batch=1, so tell them apart.
COMPILE_FAULT_RE='illegal memory access|No valid triton configs|_inductor|cudagraph'
compile_fault=false

# On a resume the batch is already proven and the weights are on disk; a second
# smoke would only cost time.
smoke_rc=1
while [[ "${SKIP_SMOKE:-false}" != "true" && "$BATCH_SIZE" -ge 1 ]]; do
  PROBE_OUTPUT="$PROBE_DIR/full_smoke_bs${BATCH_SIZE}"
  rm -rf "$PROBE_OUTPUT"
  set +e
  DATASET_REPO="$DATASET_REPO" POLICY_REPO="$POLICY_REPO" OUTPUT_DIR="$PROBE_OUTPUT" \
  JOB_NAME="${JOB_NAME}_smoke_bs${BATCH_SIZE}" BASE_MODEL="$BASE_MODEL" STEPS=4 \
  BATCH_SIZE="$BATCH_SIZE" NUM_PROCESSES=1 GPU_IDS=0 EVAL_STEPS=2 SAVE_FREQ=2 \
  MAX_EVAL_SAMPLES=$((BATCH_SIZE * 2 + 5)) NUM_WORKERS=8 WANDB_ENABLE=false WANDB_MODE=disabled \
  PUSH_TO_HUB=false SAVE_CHECKPOINT=true SAVE_CHECKPOINT_TO_HUB=false \
  COMPILE_MODEL="$COMPILE_MODEL" \
  bash "$HERE/train_full.sh" >"$PROBE_DIR/full_smoke_bs${BATCH_SIZE}.log" 2>&1
  smoke_rc=$?
  set -e
  if [[ "$smoke_rc" -eq 0 ]]; then
    # Record what a checkpoint actually costs on disk before the real run
    # starts, so the janitor's retention can be trusted.
    du -sh "$PROBE_OUTPUT/checkpoints"/* 2>/dev/null | tail -2 \
      | sed 's/^/[stack-one-cup] smoke checkpoint size: /'
    rm -rf "$PROBE_OUTPUT"
    break
  fi
  if grep -qE "$COMPILE_FAULT_RE" "$PROBE_DIR/full_smoke_bs${BATCH_SIZE}.log" 2>/dev/null; then
    compile_fault=true
    echo "[stack-one-cup] smoke hit an inductor/compile fault at batch=$BATCH_SIZE (not OOM)" >&2
    grep -oE "$COMPILE_FAULT_RE" "$PROBE_DIR/full_smoke_bs${BATCH_SIZE}.log" | sort -u | head -3 >&2
    break
  fi
  echo "[stack-one-cup] smoke failed at batch=$BATCH_SIZE; retrying at half" >&2
  tail -5 "$PROBE_DIR/full_smoke_bs${BATCH_SIZE}.log" >&2 || true
  BATCH_SIZE=$((BATCH_SIZE / 2))
done
if [[ "$compile_fault" == "true" ]]; then
  echo "SMOKE_COMPILE_FAULT" >&2
  exit 3
fi
if [[ "${SKIP_SMOKE:-false}" == "true" ]]; then
  echo "[stack-one-cup] SKIP_SMOKE=true; using batch=$BATCH_SIZE as given"
else
  [[ "$smoke_rc" -eq 0 && "$BATCH_SIZE" -ge 1 ]] || {
    echo "Full-finetune smoke failed down to batch=1; see $PROBE_DIR" >&2
    exit 1
  }
fi

train_frames=$("$PYTHON_BIN" - "$DATASET_ROOT/meta/info.json" <<'PY'
import json,sys
p=json.load(open(sys.argv[1],encoding="utf-8"))
print(int(p["total_frames"]) * 9 // 10)
PY
)
steps_per_epoch=$(((train_frames + BATCH_SIZE - 1) / BATCH_SIZE))

# ------------------------------------------------------------------ train ----
echo "[stack-one-cup] starting full PI0.5: episodes=$EXPECTED_EPISODES batch=$BATCH_SIZE steps=$STEPS steps_per_epoch=$steps_per_epoch"
set +e
setsid env \
  DATASET_REPO="$DATASET_REPO" POLICY_REPO="$POLICY_REPO" OUTPUT_DIR="$OUTPUT_DIR" \
  JOB_NAME="$JOB_NAME" BASE_MODEL="$BASE_MODEL" STEPS="$STEPS" BATCH_SIZE="$BATCH_SIZE" \
  NUM_PROCESSES=1 GPU_IDS=0 EVAL_STEPS="$EVAL_STEPS" SAVE_FREQ="$SAVE_FREQ" \
  MAX_EVAL_SAMPLES="$MAX_EVAL_SAMPLES" WANDB_ENABLE=true WANDB_MODE=online \
  PUSH_TO_HUB=false SAVE_CHECKPOINT=true SAVE_CHECKPOINT_TO_HUB=false \
  COMPILE_MODEL="$COMPILE_MODEL" RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-}" \
  bash "$HERE/train_full.sh" >>"$RUN_LOG" 2>&1 &
train_pid=$!
echo "$train_pid" > "$OUTPUT_DIR.train.pid"

setsid "$PYTHON_BIN" "$HERE/checkpoint_janitor.py" \
  --checkpoints-dir "$OUTPUT_DIR/checkpoints" --trainer-pid "$train_pid" \
  --max-retained "$MAX_RETAINED" --poll-interval 60 \
  >"$OUTPUT_DIR.janitor.log" 2>&1 &

"$PYTHON_BIN" "$HERE/monitor_full_finetune.py" \
  --log "$RUN_LOG" --output-dir "$OUTPUT_DIR" --pid "$train_pid" --marker "$MARKER" \
  --final-dir "$FINAL_DIR" --repo-id "$POLICY_REPO" --steps-per-epoch "$steps_per_epoch" \
  --batch-size "$BATCH_SIZE" --num-processes 1 --min-steps "$MIN_STEPS" --min-evals 6 \
  --patience 3 --min-relative-improvement 0.005 --trend-window 3 \
  --trend-relative-improvement 0.005 --max-checkpoints "$MAX_RETAINED" \
  --poll-interval "$POLL_INTERVAL"
monitor_rc=$?
wait "$train_pid"
train_rc=$?
set -e

if [[ "$monitor_rc" -ne 0 || ! -f "$MARKER" ]]; then
  printf 'status=not_converged\ntrain_rc=%s\nmonitor_rc=%s\n' "$train_rc" "$monitor_rc" >"$OUTPUT_DIR.NOT_CONVERGED"
  echo "[stack-one-cup] training ended without convergence evidence; server remains on" >&2
  exit 2
fi

printf 'status=converged\ntrain_rc=%s\nmonitor_rc=%s\nmarker=%s\n' \
  "$train_rc" "$monitor_rc" "$MARKER" >"$OUTPUT_DIR/COMPLETED"
sync
echo "[stack-one-cup] converged after >=$MIN_STEPS steps, verified and published; shutting down"
/usr/bin/shutdown -h now
