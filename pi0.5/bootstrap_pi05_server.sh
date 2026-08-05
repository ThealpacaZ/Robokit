#!/usr/bin/env bash
# Install the exact LeRobot revision used by the PI0.5 training scripts.
set -euo pipefail

ROOT="${ROOT:-/root/autodl-tmp}"
ENV_DIR="${ENV_DIR:-$ROOT/envs/pi05}"
LEROBOT_DIR="${LEROBOT_DIR:-$ROOT/src/lerobot}"
LEROBOT_REVISION="${LEROBOT_REVISION:-f37be3edbee60f3a09a5183788b91eb19f0c07d1}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$ROOT/conda_pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$ROOT/pip_cache}"

if [[ -r /etc/network_turbo && "${NETWORK_TURBO:-true}" == "true" ]]; then
  set +u
  # shellcheck disable=SC1091
  source /etc/network_turbo
  set -u
fi

[[ -x "$CONDA_BIN" ]] || { echo "Missing conda at $CONDA_BIN" >&2; exit 1; }
mkdir -p "$ROOT/src" "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR"

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$CONDA_BIN" create -y -p "$ENV_DIR" python=3.12
fi

if [[ ! -d "$LEROBOT_DIR/.git" ]]; then
  git clone https://github.com/huggingface/lerobot.git "$LEROBOT_DIR"
fi
git -C "$LEROBOT_DIR" fetch --depth=1 origin "$LEROBOT_REVISION"
git -C "$LEROBOT_DIR" checkout --detach "$LEROBOT_REVISION"

"$ENV_DIR/bin/python" -m pip install --upgrade pip
"$ENV_DIR/bin/python" -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 --no-deps torch torchvision
"$ENV_DIR/bin/python" -m pip install -e "$LEROBOT_DIR[pi,peft,training]" h5py scipy sentencepiece

"$ENV_DIR/bin/python" - <<'PY'
import lerobot, torch
print(f"lerobot={lerobot.__version__}")
print(f"torch={torch.__version__} cuda={torch.version.cuda} available={torch.cuda.is_available()}")
assert torch.cuda.is_available(), "CUDA is unavailable from PyTorch"
PY
