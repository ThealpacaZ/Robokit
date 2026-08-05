#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
ENV="$ROOT/envs/openvla_oft"
CODE="$ROOT/openvla-oft"
BASE="$ROOT/openvla-7b-oft-base"
SOURCE="$ROOT/openvla_oft_hf_source"
DATA="$ROOT/openvla_oft_rlds"
RUNS="$ROOT/openvla_oft_runs"
LOG="$RUNS/stackcups_401_openvla_oft_lora.supervisor.log"
SECRET=/root/autodl-tmp/secrets/pi05.env

mkdir -p "$RUNS" "$BASE" "$SOURCE" "$DATA"
exec > >(tee -a "$LOG") 2>&1
echo "[$(date -Is)] OpenVLA-OFT StackCups supervisor starting"

[[ -r "$SECRET" ]] || { echo "missing secret file: $SECRET"; exit 1; }
source "$SECRET"
: "${HF_TOKEN:?HF_TOKEN is required}"
: "${WANDB_API_KEY:?WANDB_API_KEY is required}"
if [[ -r /etc/network_turbo ]]; then
  set +u
  source /etc/network_turbo
  set -u
fi
export HF_TOKEN WANDB_API_KEY HF_HUB_DISABLE_XET=1
export HF_HOME="$ROOT/hf"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$CODE${PYTHONPATH:+:$PYTHONPATH}"

declare -A MODEL_SIZE=(
  [model-00001-of-00003.safetensors]=6948961960
  [model-00002-of-00003.safetensors]=6971232040
  [model-00003-of-00003.safetensors]=1162406824
)
declare -A MODEL_SHA=(
  [model-00001-of-00003.safetensors]=10d8636256018712c5e5c823d12e22b5797f99bb721bd123bf6bf2379892be85
  [model-00002-of-00003.safetensors]=2050b14f21d48904d269f48d5a980fecea87cd7b36641d9b0f015e72d1fe216a
  [model-00003-of-00003.safetensors]=ea65305a1577f36f721965bf84c8caec0a948ce7ce84d754701637376c531fef
)

for name in model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors; do
  path="$BASE/$name"
  if [[ -e "$path" ]] \
      && [[ "$(stat -c %s "$path")" == "${MODEL_SIZE[$name]}" ]] \
      && echo "${MODEL_SHA[$name]}  $path" | sha256sum -c -; then
    continue
  fi
  if [[ -e "$path" ]]; then
    quarantine="$ROOT/failed/openvla-base-corrupt-$(date +%Y%m%dT%H%M%S)"
    mkdir -p "$quarantine"
    mv "$path" "$quarantine/$name"
    [[ -e "$path.aria2" ]] && mv "$path.aria2" "$quarantine/$name.aria2"
    echo "quarantined invalid shard at $quarantine/$name"
  fi
  aria2c -c -x 16 -s 16 -k 1M -d "$BASE" -o "$name" \
    "https://huggingface.co/openvla/openvla-7b/resolve/main/$name?download=true"
  [[ "$(stat -c %s "$path")" == "${MODEL_SIZE[$name]}" ]] || {
    echo "wrong model shard size: $name"; exit 1;
  }
  echo "${MODEL_SHA[$name]}  $path" | sha256sum -c -
done

"$ENV/bin/python" - "$SOURCE" <<'PY'
import os, sys, time
from huggingface_hub import snapshot_download

target = sys.argv[1]
patterns = [f"stack_cups_b{i}/robokit_dataset/**" for i in range(5)]
last = None
for attempt in range(1, 5):
    try:
        snapshot_download(
            repo_id="shaohuan1/lememory",
            repo_type="dataset",
            revision="666ef5521cb3daf42d2f074bf4160cdaf92781bf",
            allow_patterns=patterns,
            local_dir=target,
            token=os.environ["HF_TOKEN"],
            max_workers=8,
        )
        break
    except Exception as error:
        last = error
        print(f"dataset download attempt {attempt} failed: {error!r}", flush=True)
        time.sleep(30)
else:
    raise RuntimeError("dataset download failed after four attempts") from last
PY

rm -rf "$DATA/robokit_stackcups_all"
"$ENV/bin/python" "$CODE/robokit_tools/combine_rlds_batches.py" \
  --source "$SOURCE" --output-root "$DATA" --expected-episodes 401

"$ENV/bin/python" - "$DATA/robokit_stackcups_all/1.1.0/dataset_info.json" <<'PY'
import json, sys
info = json.load(open(sys.argv[1], encoding="utf-8"))
episodes = sum(int(n) for split in info["splits"] for n in split["shardLengths"])
assert episodes == 401, episodes
assert info["name"] == "robokit_stackcups_all", info["name"]
print(f"validated combined dataset: {episodes} episodes")
PY

cd "$CODE"

probe_batch() {
  local batch="$1"
  local probe_root="$ROOT/openvla_oft_probes/b$batch"
  local probe_log="$ROOT/openvla_oft_probes/b$batch.log"
  rm -rf "$probe_root"
  mkdir -p "$probe_root"
  echo "[$(date -Is)] probing batch=$batch"
  set +e
  WANDB_MODE=disabled TOKENIZERS_PARALLELISM=false TF_CPP_MIN_LOG_LEVEL=2 \
    "$ENV/bin/python" -m torch.distributed.run --standalone --nnodes 1 --nproc-per-node 1 \
    "$CODE/vla-scripts/finetune.py" \
    --vla_path "$BASE" \
    --data_root_dir "$DATA" \
    --dataset_name robokit_stackcups_all \
    --run_root_dir "$probe_root" \
    --use_l1_regression True --use_diffusion False --use_film False \
    --num_images_in_input 1 --use_proprio False \
    --batch_size "$batch" --grad_accumulation_steps 1 \
    --learning_rate 5e-4 --num_steps_before_decay 75000 --max_steps 2 \
    --use_val_set False --save_freq 999999 --image_aug True \
    --lora_rank 32 --merge_lora_during_training False \
    --shuffle_buffer_size 50000 --run_id_override "probe_b$batch" \
    --wandb_log_freq 1 >"$probe_log" 2>&1
  local rc=$?
  set -e
  if [[ "$rc" == 0 ]]; then
    return 0
  fi
  if grep -Eqi "CUDA out of memory|torch.OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED" "$probe_log"; then
    echo "batch=$batch does not fit VRAM"
    return 1
  fi
  echo "batch=$batch probe failed for a non-OOM reason; leaving server on"
  tail -100 "$probe_log"
  exit "$rc"
}

selected=""
for batch in 24 16 12 8; do
  if probe_batch "$batch"; then
    selected="$batch"
    break
  fi
done
[[ -n "$selected" ]] || { echo "no tested batch fits"; exit 1; }
echo "$selected" >"$RUNS/stackcups_401_openvla_oft_lora.batch_size"
echo "[$(date -Is)] selected maximum tested stable batch=$selected"

cd "$CODE"
BATCH_SIZE="$selected" BASE_MODEL="$BASE" WANDB_RUN_ID=stackcups401oft20260801 \
  bash robokit_tools/run_stackcups_training.sh
