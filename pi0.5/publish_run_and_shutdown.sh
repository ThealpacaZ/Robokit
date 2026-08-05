#!/usr/bin/env bash
# Publish a finished PI0.5 run's weights and power the box down.
#
# The supervisor only auto-publishes when its convergence criteria fire. On a
# dataset this small eval_loss rises monotonically, so convergence never
# triggers and the weights would otherwise sit on a box that is about to be
# shut down. This finisher runs regardless of the convergence verdict.
#
# Retention order is deliberate and matches what was asked for: the final-step
# checkpoint is published FIRST, the best-eval checkpoint second, so a failure
# part-way through still leaves the higher-priority artifact on the Hub.
set -uo pipefail

: "${HF_TOKEN:?Export HF_TOKEN}"
RUN_TAG="${RUN_TAG:?Set RUN_TAG}"
FINAL_STEP="${FINAL_STEP:?Set FINAL_STEP, e.g. 30000}"
ROOT="${ROOT:-/root/autodl-tmp}"
PYTHON_BIN="${PYTHON_BIN:-$ROOT/envs/pi05/bin/python}"
HF_BIN="${HF_BIN:-$ROOT/envs/pi05/bin/hf}"
SHUTDOWN="${SHUTDOWN:-true}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/$RUN_TAG}"
FINAL_DIR="${FINAL_DIR:-$ROOT/models/$RUN_TAG}"
STEP_DIR="$OUTPUT_DIR/checkpoints/$(printf '%06d' "$FINAL_STEP")/pretrained_model"
REPO_STEP="${REPO_STEP:-shaohuan1/${RUN_TAG}_step${FINAL_STEP}}"
REPO_BEST="${REPO_BEST:-shaohuan1/$RUN_TAG}"

export HF_HOME="${HF_HOME:-$ROOT/hf}" HF_HUB_DISABLE_XET=1
if [[ -r /etc/network_turbo ]]; then
  set +u; source /etc/network_turbo >/dev/null 2>&1; set -u
fi

REQUIRED=(model.safetensors config.json train_config.json
          policy_preprocessor.json policy_postprocessor.json)

check_complete() {
  local dir="$1" name
  [[ -d "$dir" ]] || { echo "MISSING DIR: $dir" >&2; return 1; }
  for name in "${REQUIRED[@]}"; do
    [[ -f "$dir/$name" ]] || { echo "MISSING FILE: $dir/$name" >&2; return 1; }
  done
  return 0
}

publish() { # local_dir repo label
  local dir="$1" repo="$2" label="$3"
  echo "=== publishing $label -> $repo ==="
  check_complete "$dir" || { echo "SKIP $label: incomplete"; return 1; }
  "$HF_BIN" upload "$repo" "$dir" . --repo-type=model --private \
    --commit-message="PI0.5 $RUN_TAG — $label"
  local rc=$?
  echo "RC[$label]=$rc"
  return $rc
}

rc_step=1; rc_best=1
publish "$STEP_DIR"  "$REPO_STEP" "step $FINAL_STEP" && rc_step=0
publish "$FINAL_DIR" "$REPO_BEST" "best eval_loss"   && rc_best=0

echo "=== read-back verification ==="
STEP_DIR="$STEP_DIR" FINAL_DIR="$FINAL_DIR" REPO_STEP="$REPO_STEP" REPO_BEST="$REPO_BEST" \
"$PYTHON_BIN" - <<'PY'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
REQUIRED = {"model.safetensors", "config.json", "train_config.json",
            "policy_preprocessor.json", "policy_postprocessor.json"}
ok = True
for repo, local, label in ((os.environ["REPO_STEP"], os.environ["STEP_DIR"], "final step"),
                           (os.environ["REPO_BEST"], os.environ["FINAL_DIR"], "best")):
    try:
        info = api.repo_info(repo, repo_type="model", files_metadata=True)
    except Exception as exc:
        print(f"[{label}] {repo}: repo_info failed: {exc}"); ok = False; continue
    remote = {s.rfilename: s.size for s in info.siblings}
    missing = sorted(REQUIRED - set(remote))
    if missing:
        print(f"[{label}] {repo}: MISSING {missing}"); ok = False; continue
    bad = []
    for name in REQUIRED:
        lp = os.path.join(local, name)
        if os.path.exists(lp) and os.path.getsize(lp) != remote[name]:
            bad.append((name, os.path.getsize(lp), remote[name]))
    if bad:
        print(f"[{label}] {repo}: SIZE MISMATCH {bad}"); ok = False; continue
    print(f"[{label}] {repo}: OK ({len(remote)} files, "
          f"model.safetensors={remote['model.safetensors']})")
print("ALL_VERIFIED" if ok else "VERIFY_FAILED")
sys.exit(0 if ok else 1)
PY
verify_rc=$?

if [[ "$rc_step" -eq 0 && "$rc_best" -eq 0 && "$verify_rc" -eq 0 ]]; then
  echo "PUBLISH_OK"
  if [[ "$SHUTDOWN" == "true" ]]; then
    sync
    echo "shutting down"
    /usr/bin/shutdown -h now
  fi
else
  # Never power off weights that are not provably on the Hub.
  echo "PUBLISH_FAILED rc_step=$rc_step rc_best=$rc_best verify=$verify_rc; box stays on" >&2
  exit 1
fi
