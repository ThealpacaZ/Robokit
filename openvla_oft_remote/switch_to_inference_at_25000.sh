#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp
RUNS="$ROOT/openvla_oft_runs"
RUN="$RUNS/stackcups_401_openvla_oft_lora"
TRAIN_LOG="$RUNS/stackcups_401_openvla_oft_lora.resume-15000.train.log"
DEPLOY_ROOT="$ROOT/openvla_oft_deploy"
DEPLOY="$DEPLOY_ROOT/stackcups-step-025000"
SWITCH_LOG="$RUNS/stackcups_401_openvla_oft_lora.switch-to-inference.log"
SERVER_LOG="$DEPLOY_ROOT/stackcups-step-025000.server.log"
SERVER_PID="$DEPLOY_ROOT/stackcups-step-025000.server.pid"

exec >>"$SWITCH_LOG" 2>&1
echo "$(date -Is) waiting for complete global step 25000 checkpoint"

while true; do
  if grep -aq "Saving Model Checkpoint for Step 25000" "$TRAIN_LOG"; then
    # Seeing a later optimizer step proves the synchronous checkpoint function returned.
    local_step=$(tr '\r' '\n' <"$TRAIN_LOG" | sed -n 's/.*| *\([0-9][0-9]*\)\/150000.*/\1/p' | tail -n 1)
    if [[ -n "$local_step" && "$local_step" -gt 10000 ]]; then
      break
    fi
  fi
  trainer_pid=$(cat "$RUNS/stackcups_401_openvla_oft_lora.train.pid")
  kill -0 "$trainer_pid" 2>/dev/null || {
    echo "$(date -Is) trainer exited before complete step 25000 checkpoint" >&2
    exit 1
  }
  sleep 30
done

mkdir -p "$DEPLOY_ROOT"
if [[ -e "$DEPLOY" ]]; then
  echo "refusing to overwrite existing deployment snapshot: $DEPLOY" >&2
  exit 1
fi
temporary=$(mktemp -d "$DEPLOY_ROOT/.stackcups-step-025000.XXXXXX")
cp -a "$RUN/lora_adapter" "$temporary/"
for name in \
  action_head--latest_checkpoint.pt dataset_statistics.json processor_config.json \
  preprocessor_config.json processing_prismatic.py tokenizer_config.json tokenizer.json \
  tokenizer.model special_tokens_map.json added_tokens.json training_config.json; do
  [[ -s "$RUN/$name" ]] && cp -a "$RUN/$name" "$temporary/"
done
printf '{"global_step":25000,"source":"%s","saved_at":"%s"}\n' \
  "$RUN" "$(date -Is)" >"$temporary/deployment_checkpoint.json"

"$ROOT/envs/openvla_oft/bin/python" - "$temporary" <<'PY'
from pathlib import Path
import sys, torch
from safetensors import safe_open
p = Path(sys.argv[1])
with safe_open(p / "lora_adapter/adapter_model.safetensors", framework="pt", device="cpu") as f:
    assert len(list(f.keys())) > 0
head = torch.load(p / "action_head--latest_checkpoint.pt", weights_only=True, map_location="cpu")
assert head
print(f"verified adapter and action head: {p}")
PY
mv "$temporary" "$DEPLOY"
sync
echo "$(date -Is) frozen deployment snapshot $DEPLOY"

monitor_pid=$(cat "$RUNS/stackcups_401_openvla_oft_lora.monitor.pid")
kill -TERM "$monitor_pid" 2>/dev/null || true
trainer_pid=$(cat "$RUNS/stackcups_401_openvla_oft_lora.train.pid")
kill -TERM -- "-$trainer_pid" 2>/dev/null || kill -TERM "$trainer_pid" 2>/dev/null || true
for _ in $(seq 1 60); do
  pgrep -f 'vla-scripts/finetune.py' >/dev/null || break
  sleep 1
done
if pgrep -f 'vla-scripts/finetune.py' >/dev/null; then
  echo "trainer did not stop after SIGTERM" >&2
  exit 1
fi
echo "$(date -Is) trainer stopped"

port_is_open() {
  "$ROOT/envs/openvla_oft/bin/python" - <<'PY'
import socket
try:
    with socket.create_connection(("127.0.0.1", 8080), timeout=0.2):
        pass
except OSError:
    raise SystemExit(1)
PY
}

if port_is_open; then
  echo "port 8080 is already occupied; refusing to replace an unknown service" >&2
  exit 1
fi

cd /root/robokit
setsid "$ROOT/envs/openvla_oft/bin/python" scripts/deploy_server.py \
  --port 8080 --action-space eef_delta --policy openvla_oft \
  --policy-arg checkpoint=/root/autodl-tmp/openvla_oft_deploy/stackcups-step-025000 \
  --policy-arg base=/root/autodl-tmp/openvla-7b-oft-base \
  --policy-arg codebase=/root/autodl-tmp/openvla-oft \
  --policy-arg robot_platform=piper \
  >"$SERVER_LOG" 2>&1 &
server_pid=$!
echo "$server_pid" >"$SERVER_PID"

for _ in $(seq 1 180); do
  kill -0 "$server_pid" 2>/dev/null || {
    echo "inference server exited during load" >&2
    tail -n 120 "$SERVER_LOG" >&2
    exit 1
  }
  if port_is_open; then
    echo "$(date -Is) inference ready pid=$server_pid port=8080 checkpoint=$DEPLOY"
    exit 0
  fi
  sleep 1
done
echo "inference server did not listen on port 8080 within 180s" >&2
exit 1
