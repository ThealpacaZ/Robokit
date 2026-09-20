#!/usr/bin/env bash
# GPU 服务器上起/停/看推理服务。用 pidfile 而不是 pgrep -f：
# 通过 ssh 跑的远端命令行本身就含有模型名和 serve_policy.py，任何 pgrep -f/pkill -f
# 的模式都会匹配到发起命令的那个 shell 自己，一 kill 就把自己杀了。
#
#   scripts/serve_ctl.sh start pi05-joint rtc
#   scripts/serve_ctl.sh start pi05-eef   rtc
#   scripts/serve_ctl.sh status
#   scripts/serve_ctl.sh stop  pi05-joint rtc
#   scripts/serve_ctl.sh log   pi05-joint rtc [行数]
set -uo pipefail

ROOT="${ROOT:-/root/autodl-tmp}"
REPO="${REPO:-/root/robokit}"
# 两套环境：MemoryVLA 全量权重（lamem-*）用 /root/miniconda3/envs/memvla（transformers 4.40），
# pi0/pi05 系列用 $ROOT/envs/pi05（LeRobot）。按模型名前缀选，PYTHON_BIN 可显式覆盖。
pick_python() {  # $1=model
  if [ -n "${PYTHON_BIN:-}" ]; then echo "$PYTHON_BIN"; return; fi
  case "$1" in
    lamem-*|memoryvla*) echo /root/miniconda3/envs/memvla/bin/python ;;
    # OpenVLA-OFT 有自己的一套（torch 2.8 + transformers 4.40 + tf 2.15 + flash-attn），
    # 和 LeRobot 那套装不到一个环境里
    openvla*) echo "$ROOT/envs/openvla_oft/bin/python" ;;
    *) echo "$ROOT/envs/pi05/bin/python" ;;
  esac
}
SERVICE_DIR="${SERVICE_DIR:-$ROOT/outputs/eval-services}"

usage() { sed -n '2,12p' "$0"; exit 2; }

paths() {  # $1=model $2=mode
  TAG="$1-$2"
  PIDFILE="$SERVICE_DIR/$TAG.pid"
  LOGFILE="$SERVICE_DIR/$TAG.log"
}

# 只在 pidfile 指向的进程确实是我们的服务时才认它 —— PID 会被复用，
# 认错了就会 kill 掉别人的训练进程。
running() {  # $1=pidfile ; echoes pid
  local pid
  [ -f "$1" ] || return 1
  pid=$(cat "$1" 2>/dev/null) || return 1
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  tr '\0' ' ' < "/proc/$pid/cmdline" 2>/dev/null | grep -q "serve_policy.py" || return 1
  echo "$pid"
}

cmd_start() {
  paths "$1" "$2"
  mkdir -p "$SERVICE_DIR"
  if pid=$(running "$PIDFILE"); then
    echo "already running: $TAG pid=$pid"
    return 0
  fi
  cd "$REPO" || exit 1
  # PYTHONUNBUFFERED：不加的话 log() 的输出全卡在缓冲区里，
  # 真机跑的时候日志是空的，看不到每次推理的延迟。
  local py; py=$(pick_python "$1")
  [ -x "$py" ] || { echo "python 不存在: $py（先建环境或 PYTHON_BIN= 指定）" >&2; return 1; }
  PYTHONUNBUFFERED=1 HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}" setsid nohup "$py" scripts/serve_policy.py \
    --model "$1" --mode "$2" "${@:3}" > "$LOGFILE" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  echo "started $TAG pid=$(cat "$PIDFILE") log=$LOGFILE"
}

cmd_stop() {
  paths "$1" "$2"
  if pid=$(running "$PIDFILE"); then
    kill "$pid"
    for _ in $(seq 20); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
    echo "stopped $TAG pid=$pid"
  else
    echo "not running: $TAG"
  fi
  rm -f "$PIDFILE"
}

cmd_status() {
  printf '%-24s %-8s %s\n' SERVICE PID STATE
  for pidfile in "$SERVICE_DIR"/*.pid; do
    [ -e "$pidfile" ] || continue
    tag=$(basename "$pidfile" .pid)
    if pid=$(running "$pidfile"); then
      printf '%-24s %-8s %s\n' "$tag" "$pid" running
    else
      printf '%-24s %-8s %s\n' "$tag" "-" stopped
    fi
  done
  echo
  echo "listening ports:"
  python3 - <<'PY'
import socket
for p in (8080, 8081, 8082, 8083, 8084, 8085):
    s = socket.socket(); s.settimeout(1.5)
    try:
        s.connect(("127.0.0.1", p)); print(f"  {p}: LISTENING")
    except Exception as exc:
        print(f"  {p}: {type(exc).__name__}")
    finally:
        s.close()
PY
}

cmd_log() {
  paths "$1" "$2"
  tail -n "${3:-40}" "$LOGFILE"
}

[ $# -ge 1 ] || usage
action="$1"; shift
case "$action" in
  start)  [ $# -ge 2 ] || usage; cmd_start "$@" ;;
  stop)   [ $# -ge 2 ] || usage; cmd_stop "$1" "$2" ;;
  status) cmd_status ;;
  log)    [ $# -ge 2 ] || usage; cmd_log "$@" ;;
  *) usage ;;
esac
