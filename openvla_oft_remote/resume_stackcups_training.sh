#!/usr/bin/env bash
set -euo pipefail

: "${HF_TOKEN:?HF_TOKEN is required}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"

if [[ -r /etc/network_turbo ]]; then
  set +u
  source /etc/network_turbo
  set -u
fi

ROOT=/root/autodl-tmp
ENV="$ROOT/envs/openvla_oft"
CODE="$ROOT/openvla-oft"
DATA="$ROOT/openvla_oft_rlds"
BASE_MODEL="$ROOT/openvla-7b-oft-base"
RUNS="$ROOT/openvla_oft_runs"
RUN_ID=stackcups_401_openvla_oft_lora
RUN_DIR="$RUNS/$RUN_ID"
TRAIN_LOG="$RUNS/$RUN_ID.resume-15000.train.log"
POLICY_REPO=shaohuan1/stackcups_openvla_oft_lora
WANDB_RUN_ID=stackcups401oft20260801
RESUME_STEP=15000

export HF_HOME="$ROOT/hf"
export HF_HUB_DISABLE_XET=1
export WANDB_DIR="$RUNS/wandb"
export WANDB_MODE=online
export WANDB_RUN_ID WANDB_RESUME=must
export WANDB_NAME="ft+$RUN_ID"
export TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

required=(
  "$RUN_DIR/lora_adapter/adapter_model.safetensors"
  "$RUN_DIR/lora_adapter/adapter_config.json"
  "$RUN_DIR/action_head--latest_checkpoint.pt"
  "$RUN_DIR/dataset_statistics.json"
  "$BASE_MODEL/model-00001-of-00003.safetensors"
  "$DATA/robokit_stackcups_all/1.1.0/dataset_info.json"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "missing resume input: $path" >&2; exit 1; }
done
if pgrep -af 'vla-scripts/finetune.py' >/dev/null; then
  echo "trainer already running; refusing duplicate launch" >&2
  exit 1
fi

cd "$CODE"
setsid "$ENV/bin/python" -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 1 \
  vla-scripts/finetune.py \
  --vla_path "$BASE_MODEL" \
  --resume True \
  --resume_step "$RESUME_STEP" \
  --resume_checkpoint_dir "$RUN_DIR" \
  --data_root_dir "$DATA" \
  --dataset_name robokit_stackcups_all \
  --run_root_dir "$RUNS" \
  --use_l1_regression True \
  --use_diffusion False \
  --use_film False \
  --num_images_in_input 1 \
  --use_proprio False \
  --batch_size 16 \
  --grad_accumulation_steps 1 \
  --learning_rate 5e-4 \
  --num_steps_before_decay 75000 \
  --max_steps 150000 \
  --use_val_set False \
  --save_freq 5000 \
  --save_latest_checkpoint_only True \
  --merge_lora_during_training False \
  --image_aug True \
  --lora_rank 32 \
  --shuffle_buffer_size 50000 \
  --wandb_entity yangshaohuan720-university \
  --wandb_project stackcups_openvla_oft_lora \
  --run_id_override "$RUN_ID" \
  --wandb_log_freq 10 \
  >"$TRAIN_LOG" 2>&1 &
train_pid=$!
echo "$train_pid" >"$RUNS/$RUN_ID.train.pid"

nohup "$ENV/bin/python" "$CODE/robokit_tools/monitor_and_shutdown.py" \
  --pid "$train_pid" \
  --run-path "yangshaohuan720-university/stackcups_openvla_oft_lora/$WANDB_RUN_ID" \
  --run-dir "$RUN_DIR" \
  --train-log "$TRAIN_LOG" \
  --repo-id "$POLICY_REPO" \
  --poll-seconds 1800 \
  >"$RUNS/$RUN_ID.monitor.log" 2>&1 &
echo $! >"$RUNS/$RUN_ID.monitor.pid"

echo "trainer_pid=$train_pid monitor_pid=$(cat "$RUNS/$RUN_ID.monitor.pid") log=$TRAIN_LOG"
