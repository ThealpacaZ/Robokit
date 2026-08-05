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

HOST=root@connect.weste.seetacloud.com
PORT=41815
LOCAL_REPO=/home/ysh/robokit
LOCAL_POLICY=/home/ysh/Test_piper/policy/MemoryVLA-openvla-codebase
REMOTE_REPO=/root/robokit
REMOTE_POLICY=/root/autodl-tmp/MemoryVLA-openvla-codebase

SSH="ssh -p ${PORT} -o StrictHostKeyChecking=no"
RUN=()
if [[ -n "${ROBOKIT_SSH_PASS:-}" ]]; then
  command -v sshpass >/dev/null || { echo "需要 sshpass" >&2; exit 1; }
  RUN=(sshpass -p "${ROBOKIT_SSH_PASS}")
fi

EXCLUDES=(--exclude='__pycache__/' --exclude='datasets/' --exclude='*.hdf5'
          --exclude='.git/' --exclude='*.egg-info/' --exclude='*.pt' --exclude='*.pth')

push() {
  echo "==> 代码 -> ${REMOTE_REPO}"
  "${RUN[@]}" rsync -az --delete "${EXCLUDES[@]}" -e "${SSH}" \
    "${LOCAL_REPO}/" "${HOST}:${REMOTE_REPO}/"
  if [[ -d "${LOCAL_POLICY}" ]]; then
    # pretrained/ 下是服务器独有的 33 GB checkpoint 及其同目录元数据，本机没有对应文件。
    # 必须先 --filter 保护再 --delete，否则整个 laMem-VLA 目录会被当成「多余文件」删掉。
    echo "==> MemoryVLA 代码库 -> ${REMOTE_POLICY}（保护 pretrained/）"
    "${RUN[@]}" rsync -az --delete --filter='protect pretrained/***' \
      "${EXCLUDES[@]}" --exclude='wandb/' -e "${SSH}" \
      "${LOCAL_POLICY}/" "${HOST}:${REMOTE_POLICY}/"
  fi
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
    "cd ${REMOTE_REPO} && sha256sum PROJECT_MEMORY.md TASK_STATE.md scripts/deploy_server.py robokit/policies/memoryvla.py 2>/dev/null"
  echo "==> 本机对应校验和"
  (cd "${LOCAL_REPO}" && sha256sum PROJECT_MEMORY.md TASK_STATE.md scripts/deploy_server.py robokit/policies/memoryvla.py)
}

case "${1:-push}" in
  push)   push ;;
  pull)   pull ;;
  status) status ;;
  *) echo "用法: $0 [push|pull|status]" >&2; exit 1 ;;
esac
