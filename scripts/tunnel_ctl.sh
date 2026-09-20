#!/usr/bin/env bash
# 机器人端到 GPU 服务器的 SSH 隧道，断了自动重连。
#
#   export ROBOKIT_SSH_PASS=...          # 不设就走交互式密码（前台）
#   scripts/tunnel_ctl.sh start
#   scripts/tunnel_ctl.sh status
#   scripts/tunnel_ctl.sh stop
#
# 为什么要守护：裸 ssh -N -L 掉线后不会自己回来，端口就此消失，而真机端只有在下一次
# 请求时才会发现——表现为跑到一半 ConnectionRefusedError 中止。AutoDL 这条链路
# 经代理，掉线不算罕见。
set -uo pipefail

# 默认 westb（~/.ssh/config 里的别名，含 socks ProxyCommand）；换机器用 ROBOKIT_SSH_HOST 覆盖
HOST="${ROBOKIT_SSH_HOST:-westb}"
PORTS="${ROBOKIT_TUNNEL_PORTS:-8080 8081 8082 8083 8084 8085}"
RUN_DIR="${ROBOKIT_RUN_DIR:-$HOME/.robokit}"
PIDFILE="$RUN_DIR/tunnel.pid"
LOGFILE="$RUN_DIR/tunnel.log"

forwards() {
  for p in $PORTS; do printf -- '-L %s:127.0.0.1:%s ' "$p" "$p"; done
}

running() {
  local pid
  [ -f "$PIDFILE" ] || return 1
  pid=$(cat "$PIDFILE" 2>/dev/null) || return 1
  [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null || return 1
  echo "$pid"
}

probe() {
  python3 - "$PORTS" <<'PY'
import socket, sys
for port in sys.argv[1].split():
    s = socket.socket(); s.settimeout(2)
    try:
        s.connect(("127.0.0.1", int(port))); print(f"  {port}: OK")
    except Exception as exc:
        print(f"  {port}: {type(exc).__name__}")
    finally:
        s.close()
PY
}

cmd_start() {
  mkdir -p "$RUN_DIR"
  if pid=$(running); then echo "already running pid=$pid"; probe; return 0; fi

  local ssh_cmd=(ssh -o StrictHostKeyChecking=no -o ConnectTimeout=20 -o ServerAliveInterval=15
                 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes -N)
  if [ -n "${ROBOKIT_SSH_PASS:-}" ]; then
    command -v sshpass >/dev/null || { echo "需要 sshpass" >&2; exit 1; }
    # sshpass -e 读的是 SSHPASS 这个名字。没设时它打的是一整屏 usage 加
    # "At most one of -f, -d, -p or -e should be used"，看起来像参数给重了，
    # 其实是变量没传进来 —— 必须在这里显式转一道。
    export SSHPASS="$ROBOKIT_SSH_PASS"
    ssh_cmd=(sshpass -e "${ssh_cmd[@]}")
  fi

  # 用子 shell 跑重连循环，pidfile 记的是循环本身；kill 它会连带带走 ssh。
  setsid nohup bash -c '
    while true; do
      echo "[$(date "+%F %T")] connecting..."
      '"${ssh_cmd[*]}"' '"$(forwards)"' '"$HOST"'
      echo "[$(date "+%F %T")] tunnel exited ($?), retry in 3s"
      sleep 3
    done
  ' >> "$LOGFILE" 2>&1 < /dev/null &
  echo $! > "$PIDFILE"
  sleep 6
  echo "started pid=$(cat "$PIDFILE") log=$LOGFILE"
  probe
}

cmd_stop() {
  if pid=$(running); then
    # 先杀整个进程组，否则循环会立刻把 ssh 拉回来。
    kill -- "-$pid" 2>/dev/null || kill "$pid"
    sleep 1
    kill -0 "$pid" 2>/dev/null && kill -9 -- "-$pid" 2>/dev/null
    echo "stopped pid=$pid"
  else
    echo "not running"
  fi
  rm -f "$PIDFILE"
}

case "${1:-status}" in
  start)  cmd_start ;;
  stop)   cmd_stop ;;
  status)
    if pid=$(running); then echo "keeper running pid=$pid"; else echo "keeper not running"; fi
    probe
    ;;
  *) sed -n '2,10p' "$0"; exit 2 ;;
esac
