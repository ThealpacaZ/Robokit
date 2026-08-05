#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_REPO:=shaohuan1/stackcups_joint_pi0_lerobot}"
: "${POLICY_REPO:=shaohuan1/stackcups_joint_pi0_h30}"

BASE_MODEL="${BASE_MODEL:-lerobot/pi0_base}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/stackcups_joint_pi0_h30}"
JOB_NAME="${JOB_NAME:-stackcups_joint_pi0_h30}"
STEPS="${STEPS:-30000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-32}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-4}"
DECAY_LEARNING_RATE="${DECAY_LEARNING_RATE:-2.5e-5}"
SAVE_FREQ="${SAVE_FREQ:-1000}"
NUM_WORKERS="${NUM_WORKERS:-4}"
LOG_FREQ="${LOG_FREQ:-50}"
EVAL_SPLIT="${EVAL_SPLIT:-0.10}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2048}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-stackcups_joint_pi0_h30}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
PUSH_TO_HUB="${PUSH_TO_HUB:-true}"
POLICY_PRIVATE="${POLICY_PRIVATE:-true}"
SAVE_CHECKPOINT_TO_HUB="${SAVE_CHECKPOINT_TO_HUB:-false}"
CHUNK_SIZE="${CHUNK_SIZE:-50}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-30}"

if [[ "$STEPS" != "30000" ]]; then
  echo "[pi0-train] refusing STEPS=$STEPS; the EEF comparison budget is 30000" >&2
  exit 2
fi
if [[ "$CHUNK_SIZE" != "50" || "$EXECUTION_HORIZON" != "30" ]]; then
  echo "[pi0-train] expected CHUNK_SIZE=50 and EXECUTION_HORIZON=30; got $CHUNK_SIZE/$EXECUTION_HORIZON" >&2
  exit 2
fi

exec lerobot-train \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.eval_split="$EVAL_SPLIT" \
  --dataset.video_backend=pyav \
  --policy.path="$BASE_MODEL" \
  --policy.repo_id="$POLICY_REPO" \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.chunk_size="$CHUNK_SIZE" \
  --policy.n_action_steps="$EXECUTION_HORIZON" \
  --policy.use_relative_actions=false \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model=true \
  --policy.device=cuda \
  --policy.optimizer_lr="$LEARNING_RATE" \
  --policy.scheduler_decay_lr="$DECAY_LEARNING_RATE" \
  --policy.push_to_hub="$PUSH_TO_HUB" \
  --policy.private="$POLICY_PRIVATE" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --log_freq="$LOG_FREQ" \
  --eval_steps="$EVAL_STEPS" \
  --max_eval_samples="$MAX_EVAL_SAMPLES" \
  --save_freq="$SAVE_FREQ" \
  --save_checkpoint_to_hub="$SAVE_CHECKPOINT_TO_HUB" \
  --wandb.enable="$WANDB_ENABLE" \
  --wandb.project="$WANDB_PROJECT" \
  --wandb.mode="$WANDB_MODE" \
  --wandb.disable_artifact="$WANDB_DISABLE_ARTIFACT" \
  --peft.method_type=LORA \
  --peft.r="$LORA_RANK" \
  --peft.lora_alpha="$LORA_ALPHA"
