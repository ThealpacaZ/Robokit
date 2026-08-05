#!/usr/bin/env bash
set -euo pipefail

: "${DATASET_REPO:?Set DATASET_REPO}"
: "${POLICY_REPO:?Set POLICY_REPO}"
: "${OUTPUT_DIR:?Set OUTPUT_DIR}"
: "${JOB_NAME:?Set JOB_NAME}"
: "${STEPS:?Set STEPS}"
: "${BATCH_SIZE:?Set per-GPU BATCH_SIZE}"

BASE_MODEL="${BASE_MODEL:-lerobot/pi05_base}"
NUM_PROCESSES="${NUM_PROCESSES:-2}"
GPU_IDS="${GPU_IDS:-0,1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_FREQ="${LOG_FREQ:-50}"
EVAL_SPLIT="${EVAL_SPLIT:-0.10}"
EVAL_STEPS="${EVAL_STEPS:-1000}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-2048}"
SAVE_FREQ="${SAVE_FREQ:-$EVAL_STEPS}"
SEED="${SEED:-1000}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-5}"
DECAY_LEARNING_RATE="${DECAY_LEARNING_RATE:-2.5e-6}"
SCHEDULER_WARMUP_STEPS="${SCHEDULER_WARMUP_STEPS:-1000}"
SCHEDULER_DECAY_STEPS="${SCHEDULER_DECAY_STEPS:-30000}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-stackcups_pi05_full}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_DISABLE_ARTIFACT="${WANDB_DISABLE_ARTIFACT:-true}"
PUSH_TO_HUB="${PUSH_TO_HUB:-false}"
POLICY_PRIVATE="${POLICY_PRIVATE:-true}"
SAVE_CHECKPOINT="${SAVE_CHECKPOINT:-true}"
SAVE_CHECKPOINT_TO_HUB="${SAVE_CHECKPOINT_TO_HUB:-false}"
COMPILE_MODEL="${COMPILE_MODEL:-true}"
ACCELERATE_BIN="${ACCELERATE_BIN:-accelerate}"
LEROBOT_TRAIN_BIN="${LEROBOT_TRAIN_BIN:-lerobot-train}"
LEROBOT_TRAIN_PATH="$(command -v "$LEROBOT_TRAIN_BIN")"
RESUME_CONFIG_PATH="${RESUME_CONFIG_PATH:-}"

MODEL_SOURCE_ARGS=(--policy.path="$BASE_MODEL")
RESUME_SCHEDULER_ARGS=()
if [[ -n "$RESUME_CONFIG_PATH" ]]; then
  MODEL_SOURCE_ARGS=(
    --resume=true
    --config_path="$RESUME_CONFIG_PATH"
  )
  if [[ -n "${RESUME_SCHEDULER_WARMUP_STEPS:-}" ]]; then
    RESUME_SCHEDULER_ARGS+=(
      --scheduler.num_warmup_steps="$RESUME_SCHEDULER_WARMUP_STEPS"
    )
  fi
  if [[ -n "${RESUME_SCHEDULER_DECAY_STEPS:-}" ]]; then
    RESUME_SCHEDULER_ARGS+=(
      --scheduler.num_decay_steps="$RESUME_SCHEDULER_DECAY_STEPS"
    )
  fi
fi

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export WANDB_NAME="$JOB_NAME"

# accelerate rejects --multi_gpu with a single process, so only pass it when
# more than one process is actually launched.
LAUNCH_ARGS=()
if [[ "$NUM_PROCESSES" -gt 1 ]]; then
  LAUNCH_ARGS+=(--multi_gpu)
fi

exec "$ACCELERATE_BIN" launch \
  "${LAUNCH_ARGS[@]}" \
  --num_machines=1 \
  --num_processes="$NUM_PROCESSES" \
  --gpu_ids="$GPU_IDS" \
  --mixed_precision=bf16 \
  "$LEROBOT_TRAIN_PATH" \
  --dataset.repo_id="$DATASET_REPO" \
  --dataset.eval_split="$EVAL_SPLIT" \
  --dataset.video_backend=pyav \
  "${MODEL_SOURCE_ARGS[@]}" \
  --policy.repo_id="$POLICY_REPO" \
  --policy.input_features=null \
  --policy.output_features=null \
  --policy.dtype=bfloat16 \
  --policy.gradient_checkpointing=true \
  --policy.compile_model="$COMPILE_MODEL" \
  --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=false \
  --policy.device=cuda \
  --policy.optimizer_lr="$LEARNING_RATE" \
  --policy.scheduler_warmup_steps="$SCHEDULER_WARMUP_STEPS" \
  --policy.scheduler_decay_steps="$SCHEDULER_DECAY_STEPS" \
  --policy.scheduler_decay_lr="$DECAY_LEARNING_RATE" \
  --policy.push_to_hub="$PUSH_TO_HUB" \
  --policy.private="$POLICY_PRIVATE" \
  --output_dir="$OUTPUT_DIR" \
  --job_name="$JOB_NAME" \
  --seed="$SEED" \
  --steps="$STEPS" \
  --batch_size="$BATCH_SIZE" \
  --num_workers="$NUM_WORKERS" \
  --log_freq="$LOG_FREQ" \
  --eval_steps="$EVAL_STEPS" \
  --max_eval_samples="$MAX_EVAL_SAMPLES" \
  --save_freq="$SAVE_FREQ" \
  --save_checkpoint="$SAVE_CHECKPOINT" \
  --save_checkpoint_to_hub="$SAVE_CHECKPOINT_TO_HUB" \
  --wandb.enable="$WANDB_ENABLE" \
  --wandb.project="$WANDB_PROJECT" \
  --wandb.mode="$WANDB_MODE" \
  --wandb.disable_artifact="$WANDB_DISABLE_ARTIFACT" \
  "${RESUME_SCHEDULER_ARGS[@]}"
