#!/usr/bin/env bash
# 本机 <-> AutoDL 服务器的代码与记忆同步。
#
#   ./scripts/sync_server.sh push        代码 + 记忆 本机 -> 服务器（默认）
#   ./scripts/sync_server.sh pull        记忆 + runs/  服务器 -> 本机
#   ./scripts/sync_server.sh status      两边 git/记忆差异
#
# 密码不写在文件里，通过环境变量传：
#   export ROBOKIT_SSH_PASS=...；未设置时走交互式 ssh 密码提示。
set -euo pipefail

# 2026-09-11 起指向 westb 4090 推理机：连接细节（端口、socks ProxyCommand）在 ~/.ssh/config 的
# `Host westb` 里；直连 kex 阶段会被断，必须走本机 socks 7899。
HOST=${ROBOKIT_SSH_HOST:-westb}
LOCAL_REPO=/home/ysh/robokit
# MemoryVLA 代码库现在 vendor 在仓库里（vendor/MemoryVLA-openvla-codebase），随代码一起推
REMOTE_REPO=/root/robokit

SSH="ssh -o StrictHostKeyChecking=no"
RUN=()
if [[ -n "${ROBOKIT_SSH_PASS:-}" ]]; then
  command -v sshpass >/dev/null || { echo "需要 sshpass" >&2; exit 1; }
  RUN=(sshpass -p "${ROBOKIT_SSH_PASS}")
fi

EXCLUDES=(--exclude='__pycache__/' --exclude='datasets/' --exclude='hf/' --exclude='*.hdf5'
          --exclude='.git/' --exclude='*.egg-info/' --exclude='*.pt' --exclude='*.pth')

push() {
  echo "==> 代码 -> ${REMOTE_REPO}"
  "${RUN[@]}" rsync -az --delete "${EXCLUDES[@]}" -e "${SSH}" \
    "${LOCAL_REPO}/" "${HOST}:${REMOTE_REPO}/"
  echo "==> 完成"
}

pull() {
  echo "==> 记忆与 runs <- 服务器"
  "${RUN[@]}" rsync -az -e "${SSH}" \
    "${HOST}:${REMOTE_REPO}/PROJECT_MEMORY.md" "${HOST}:${REMOTE_REPO}/TASK_STATE.md" \
    "${LOCAL_REPO}/"
  "${RUN[@]}" rsync -az -e "${SSH}" "${HOST}:${REMOTE_REPO}/runs/" "${LOCAL_REPO}/runs/"
  echo "==> 完成（记得 git diff 复核后再提交）"
}

status() {
  echo "==> 本机 git"
  git -C "${LOCAL_REPO}" log --oneline -1
  git -C "${LOCAL_REPO}" status --short
  echo "==> 服务器文件校验和"
  "${RUN[@]}" ${SSH} "${HOST}" \
    "cd ${REMOTE_REPO} && sha256sum PROJECT_MEMORY.md TASK_STATE.md scripts/serve_policy.py robokit/policies/memoryvla.py 2>/dev/null"
  echo "==> 本机对应校验和"
  (cd "${LOCAL_REPO}" && sha256sum PROJECT_MEMORY.md TASK_STATE.md scripts/serve_policy.py robokit/policies/memoryvla.py)
}

case "${1:-push}" in
  push)   push ;;
  pull)   pull ;;
  status) status ;;
  *) echo "用法: $0 [push|pull|status]" >&2; exit 1 ;;
esac
