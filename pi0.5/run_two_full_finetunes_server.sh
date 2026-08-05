#!/usr/bin/env bash
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
PI05_ENV_BIN="${PI05_ENV_BIN:-/root/autodl-tmp/envs/pi05/bin}"
export PATH="$PI05_ENV_BIN:$PATH"
PYTHON_BIN="${PYTHON_BIN:-$PI05_ENV_BIN/python}"
HF_HOME="${HF_HOME:-/root/autodl-tmp/hf}"
HF_LEROBOT_HOME="${HF_LEROBOT_HOME:-/root/autodl-tmp/hf/lerobot}"
RAW_SOURCE="${RAW_SOURCE:-/root/autodl-tmp/pi05/source/stackcups_150_400_raw}"
JOINT_SOURCE="${JOINT_SOURCE:-/root/autodl-tmp/pi05/source/stackcups_150_400}"
EEF_SOURCE="${EEF_SOURCE:-/root/autodl-tmp/pi05/source/stackcups_202_400}"
TOKENIZER_DIR="${TOKENIZER_DIR:-/root/autodl-tmp/models/paligemma_tokenizer}"
BASE_MODEL="${BASE_MODEL:-/root/autodl-tmp/models/pi05_base_local}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/root/autodl-tmp/outputs}"
FINAL_ROOT="${FINAL_ROOT:-/root/autodl-tmp/models/full_finetunes}"
PROBE_ROOT="${PROBE_ROOT:-/root/autodl-tmp/probes/pi05_full}"
WANDB_PROJECT="${WANDB_PROJECT:-stackcups_pi05_full}"
NUM_PROCESSES=2
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-true}"
QUICK_START="${QUICK_START:-true}"
BATCH_SIZE_OVERRIDE="${BATCH_SIZE_OVERRIDE:-16}"

JOINT_DATASET="shaohuan1/stackcups_150_400_joint_pi05_lerobot"
EEF_DATASET="shaohuan1/stackcups_202_400_eef_pi05_lerobot"
JOINT_POLICY="shaohuan1/stackcups_150_400_joint_pi05_full"
EEF_POLICY="shaohuan1/stackcups_202_400_eef_pi05_full"
JOINT_FRAMES=53343
EEF_FRAMES=41495

export HF_HOME HF_LEROBOT_HOME HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export WANDB_PROJECT BASE_MODEL NUM_PROCESSES
mkdir -p "$OUTPUT_ROOT" "$FINAL_ROOT" "$PROBE_ROOT" "$JOINT_SOURCE" "$EEF_SOURCE"

stage_source() {
  local start_id="$1"
  local end_id="$2"
  local destination="$3"
  local expected=$((end_id - start_id + 1))
  local actual=0
  local episode
  for episode in $(seq "$start_id" "$end_id"); do
    [[ -s "$RAW_SOURCE/$episode.hdf5" ]] || {
      echo "[full-supervisor] missing $RAW_SOURCE/$episode.hdf5" >&2
      return 1
    }
    ln -sfn "$RAW_SOURCE/$episode.hdf5" "$destination/$episode.hdf5"
    actual=$((actual + 1))
  done
  [[ "$actual" -eq "$expected" ]]
  cp "$RAW_SOURCE/clean_report.json" "$destination/clean_report.json"
  cp "$RAW_SOURCE/config.json" "$destination/config.json"
  cp "$RAW_SOURCE/source_manifest.sha256" "$destination/source_manifest.sha256"
}

validate_manifest() {
  local root="$1"
  local action_space="$2"
  local episodes="$3"
  local frames="$4"
  "$PYTHON_BIN" - "$root/conversion_manifest.json" "$action_space" "$episodes" "$frames" <<'PY'
import json,sys
path,action_space,episodes,frames=sys.argv[1:]
p=json.load(open(path,encoding="utf-8"))
expected={"action_space":action_space,"episode_count":int(episodes),"frame_count":int(frames)}
actual={key:p.get(key) for key in expected}
if actual != expected:
    raise SystemExit(f"conversion manifest mismatch: expected={expected}, actual={actual}")
print(json.dumps(actual,ensure_ascii=False))
PY
}

prepare_dataset() {
  local source="$1"
  local repo="$2"
  local action_space="$3"
  local episodes="$4"
  local frames="$5"
  local root="$HF_LEROBOT_HOME/$repo"
  if [[ ! -f "$root/meta/info.json" ]]; then
    "$PYTHON_BIN" "$HERE/convert_hdf5_to_lerobot.py" \
      --data "$source" \
      --repo-id "$repo" \
      --camera cam_high \
      --arm right_arm \
      --instruction "stack cups" \
      --action-space "$action_space" \
      --push-to-hub \
      --private \
      --upload-large-folder
  fi
  validate_manifest "$root" "$action_space" "$episodes" "$frames"
  "$PYTHON_BIN" "$HERE/validate_lerobot.py" \
    --repo-id "$repo" \
    --root "$root" \
    --action-space "$action_space"
}

JOINT_DATASET_ROOT="$HF_LEROBOT_HOME/$JOINT_DATASET"
EEF_DATASET_ROOT="$HF_LEROBOT_HOME/$EEF_DATASET"

# A supervisor restart must not require the raw HDF5 staging tree after the
# already-published LeRobot encodings have been validated.
if [[ -f "$JOINT_DATASET_ROOT/meta/info.json" ]]; then
  validate_manifest "$JOINT_DATASET_ROOT" joint 251 "$JOINT_FRAMES"
  "$PYTHON_BIN" "$HERE/validate_lerobot.py" \
    --repo-id "$JOINT_DATASET" \
    --root "$JOINT_DATASET_ROOT" \
    --action-space joint
else
  stage_source 150 400 "$JOINT_SOURCE"
  prepare_dataset "$JOINT_SOURCE" "$JOINT_DATASET" joint 251 "$JOINT_FRAMES"
fi

EEF_DATASET_READY=false
if [[ -f "$EEF_DATASET_ROOT/meta/info.json" ]]; then
  validate_manifest "$EEF_DATASET_ROOT" eef_delta 199 "$EEF_FRAMES"
  "$PYTHON_BIN" "$HERE/validate_lerobot.py" \
    --repo-id "$EEF_DATASET" \
    --root "$EEF_DATASET_ROOT" \
    --action-space eef_delta
  EEF_DATASET_READY=true
else
  stage_source 202 400 "$EEF_SOURCE"
fi

"$PYTHON_BIN" "$HERE/prefetch_hf.py" \
  --dataset "$JOINT_DATASET" \
  --base lerobot/pi05_base
"$PYTHON_BIN" "$HERE/prepare_paligemma_tokenizer.py" \
  --base-repo lerobot/pi05_base \
  --tokenizer-dir "$TOKENIZER_DIR" \
  --output "$BASE_MODEL"

PROBE_RESULT="$PROBE_ROOT/result.json"
if [[ "$QUICK_START" == "true" ]]; then
  BATCH_SIZE="$BATCH_SIZE_OVERRIDE"
  QUICK_SMOKE="$PROBE_ROOT/joint_quick_smoke"
  if [[ ! -f "$PROBE_ROOT/joint_quick_smoke.complete" ]]; then
    quick_smoke_rc=1
    while [[ "$BATCH_SIZE" -ge 1 ]]; do
      rm -rf "$QUICK_SMOKE"
      set +e
      DATASET_REPO="$JOINT_DATASET" \
      POLICY_REPO="$JOINT_POLICY" \
      OUTPUT_DIR="$QUICK_SMOKE" \
      JOB_NAME="joint_150_400_full_quick_smoke_bs${BATCH_SIZE}" \
      STEPS=3 \
      BATCH_SIZE="$BATCH_SIZE" \
      EVAL_STEPS=103 \
      SAVE_FREQ=103 \
      MAX_EVAL_SAMPLES=1 \
      WANDB_ENABLE=false \
      WANDB_MODE=disabled \
      PUSH_TO_HUB=false \
      SAVE_CHECKPOINT=false \
      SAVE_CHECKPOINT_TO_HUB=false \
      bash "$HERE/train_full.sh" >"$PROBE_ROOT/joint_quick_smoke_bs${BATCH_SIZE}.log" 2>&1
      quick_smoke_rc=$?
      set -e
      [[ "$quick_smoke_rc" -eq 0 ]] && break
      BATCH_SIZE=$((BATCH_SIZE / 2))
    done
    [[ "$quick_smoke_rc" -eq 0 && "$BATCH_SIZE" -ge 1 ]] || {
      echo "[full-supervisor] full finetune smoke failed down to batch 1" >&2
      exit 1
    }
    rm -rf "$QUICK_SMOKE"
    printf 'batch_size_per_gpu=%s\nglobal_batch_size=%s\nsteps=3\n' \
      "$BATCH_SIZE" "$((BATCH_SIZE * 2))" >"$PROBE_ROOT/joint_quick_smoke.complete"
  fi
elif [[ ! -f "$PROBE_RESULT" ]]; then
  "$PYTHON_BIN" "$HERE/probe_full_batch.py" \
    --train-script "$HERE/train_full.sh" \
    --dataset-repo "$JOINT_DATASET" \
    --policy-repo "$JOINT_POLICY" \
    --base-model "$BASE_MODEL" \
    --work-dir "$PROBE_ROOT" \
    --result "$PROBE_RESULT" \
    --job-prefix joint_150_400_full
  BATCH_SIZE="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_batch_size_per_gpu"])' "$PROBE_RESULT")"
else
  BATCH_SIZE="$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected_batch_size_per_gpu"])' "$PROBE_RESULT")"
fi
[[ "$BATCH_SIZE" -gt 0 ]]

run_task() {
  local slug="$1"
  local dataset_repo="$2"
  local policy_repo="$3"
  local frame_count="$4"
  local min_steps_override="${5:-0}"
  local max_steps_override="${6:-0}"
  local hard_max_steps="${7:-0}"
  local output_dir="$OUTPUT_ROOT/$slug"
  local final_dir="$FINAL_ROOT/$slug"
  local log="$OUTPUT_ROOT/$slug.train.log"
  local marker="$output_dir/converged.json"
  local train_frames epoch_steps min_steps max_steps max_eval
  train_frames=$((frame_count * 9 / 10))
  epoch_steps=$(((train_frames + BATCH_SIZE * NUM_PROCESSES - 1) / (BATCH_SIZE * NUM_PROCESSES)))
  min_steps=$((epoch_steps * 5))
  max_steps=$((epoch_steps * 10))
  if [[ "$min_steps_override" -gt 0 ]]; then
    min_steps="$min_steps_override"
  fi
  if [[ "$max_steps_override" -gt 0 ]]; then
    max_steps="$max_steps_override"
  fi
  max_eval=$((frame_count - train_frames))

  if [[ -f "$marker" ]]; then
    echo "[full-supervisor] $slug already converged; verifying retained artifacts"
    [[ -s "$final_dir/model.safetensors" ]]
    "$PYTHON_BIN" "$HERE/verify_wandb_run.py" --marker "$marker"
    return
  fi
  if [[ -e "$output_dir" || -e "$final_dir" ]]; then
    echo "[full-supervisor] refusing to overwrite incomplete task artifacts for $slug" >&2
    return 1
  fi

  echo "[full-supervisor] start $slug: per_gpu_batch=$BATCH_SIZE global_batch=$((BATCH_SIZE * 2)) epoch_steps=$epoch_steps max_steps=$max_steps"
  set +e
  setsid env \
    DATASET_REPO="$dataset_repo" \
    POLICY_REPO="$policy_repo" \
    OUTPUT_DIR="$output_dir" \
    JOB_NAME="${slug}_bs${BATCH_SIZE}_2gpu" \
    BASE_MODEL="$BASE_MODEL" \
    STEPS="$max_steps" \
    BATCH_SIZE="$BATCH_SIZE" \
    EVAL_STEPS="$epoch_steps" \
    SAVE_FREQ="$epoch_steps" \
    MAX_EVAL_SAMPLES="$max_eval" \
    WANDB_ENABLE=true \
    WANDB_MODE=online \
    PUSH_TO_HUB=false \
    SAVE_CHECKPOINT=true \
    SAVE_CHECKPOINT_TO_HUB=false \
    bash "$HERE/train_full.sh" >"$log" 2>&1 &
  local train_pid=$!

  "$PYTHON_BIN" "$HERE/monitor_full_finetune.py" \
    --log "$log" \
    --output-dir "$output_dir" \
    --pid "$train_pid" \
    --marker "$marker" \
    --final-dir "$final_dir" \
    --repo-id "$policy_repo" \
    --steps-per-epoch "$epoch_steps" \
    --batch-size "$BATCH_SIZE" \
    --num-processes "$NUM_PROCESSES" \
    --min-steps "$min_steps" \
    --min-evals 5 \
    --patience 3 \
    --min-relative-improvement 0.005 \
    --trend-window 3 \
    --trend-relative-improvement 0.005 \
    --hard-max-steps "$hard_max_steps" \
    --max-checkpoints 3
  local monitor_rc=$?
  wait "$train_pid"
  local train_rc=$?
  set -e

  if [[ "$monitor_rc" -ne 0 || ! -f "$marker" ]]; then
    printf 'status=failed\ntrain_rc=%s\nmonitor_rc=%s\n' \
      "$train_rc" "$monitor_rc" >"$output_dir.FAILED"
    echo "[full-supervisor] $slug did not converge; server stays on" >&2
    return 1
  fi

  "$PYTHON_BIN" -m wandb sync --sync-all "$output_dir/wandb" || true
  "$PYTHON_BIN" "$HERE/verify_wandb_run.py" --marker "$marker"
  [[ -s "$final_dir/model.safetensors" ]]
  printf 'status=converged\ntrain_rc=%s\nmonitor_rc=%s\n' \
    "$train_rc" "$monitor_rc" >"$output_dir/COMPLETED"
  sync
}

# Prepare the second dataset concurrently with the first GPU run. Its encoder
# is CPU-bound and the result is required only after task one converges.
eef_prepare_pid=""
if [[ "$EEF_DATASET_READY" != "true" ]]; then
  (
    prepare_dataset "$EEF_SOURCE" "$EEF_DATASET" eef_delta 199 "$EEF_FRAMES"
  ) >"$OUTPUT_ROOT/stackcups_202_400_eef_pi05_full.prepare.log" 2>&1 &
  eef_prepare_pid=$!
fi

run_task \
  "stackcups_150_400_joint_pi05_full" \
  "$JOINT_DATASET" \
  "$JOINT_POLICY" \
  "$JOINT_FRAMES"

if [[ -n "$eef_prepare_pid" ]]; then
  wait "$eef_prepare_pid"
fi

# The joint dataset is already verified on the Hub and no longer needed by the
# second task. Removing only this task-created local encoding leaves the final
# joint model and the source HDF5 intact while reserving checkpoint headroom.
rm -rf "$HF_LEROBOT_HOME/$JOINT_DATASET"

run_task \
  "stackcups_202_400_eef_pi05_full" \
  "$EEF_DATASET" \
  "$EEF_POLICY" \
  "$EEF_FRAMES" \
  10000 \
  20000 \
  20000

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
(outputs/"stackcups_pi05_full_two_tasks.COMPLETED.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
PY
sync

if [[ "$AUTO_SHUTDOWN" == "true" ]]; then
  echo "[full-supervisor] both full finetunes converged and verified; shutting down"
  /usr/bin/shutdown -h now
fi
