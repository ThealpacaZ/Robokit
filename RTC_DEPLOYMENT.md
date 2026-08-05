# Piper PI0 / PI0.5 RTC deployment

This implementation follows Physical Intelligence's open-source RTC design:
inference runs while the current chunk is being executed, the previous
normalized chunk is used as the model-space inpainting prefix, and actions
whose controller timesteps elapsed during inference are skipped when the new
chunk is installed.

Official references:

- <https://www.pi.website/research/real_time_chunking>
- <https://github.com/Physical-Intelligence/real-time-chunking-kinetix>
- <https://huggingface.co/docs/lerobot/rtc>

## Contracts

| `--model` | Robot action | Model H | default `s_min` | Port |
|---|---|---:|---:|---:|
| `pi05-joint` | six absolute joint radians + absolute gripper | 50 | 15 | 8081 |
| `pi05-eef` | local SE(3) EEF delta + absolute gripper | 50 | 25 | 8080 |

Both ends use the same two entry points; `--model` selects the checkpoint and
its action space. See `configs/models.yaml`.

Both servers keep the previous chunk private in normalized model space.
The robot receives only physical-unit actions and an opaque chunk id. The
client measures warm-up latency, converts it to controller steps, maintains a
rolling maximum, and enforces `d <= s_min <= H-d` before motion.

Absolute joint targets need no relative re-anchoring. The EEF checkpoint was trained
with sequential local SE(3) increments and `use_relative_actions=false`; it
must not use LeRobot's ordinary `absolute_action-current_state` relative-action
helper. Local transform increments are already frame-relative. Guided overlap
provides the chunk transition, so EEF RTC uses `chunk_base=continuous`; clearing
the base at every high-frequency swap accumulates small IK/FK residuals.

## 1. Deterministic no-CAN digital twin

This runs real HDF5 trajectories through the RTC state machine, injected
3/5/7-step delays, ActionGuard, the production Piper action path, and
Pinocchio IK:

```bash
python scripts/validate_rtc_policy_path.py \
  --episodes 2,5,7 \
  --max-actions 180
```

Report:
`runs/validation/rtc-policy-path-digital-twin.json`.

The chosen episodes remain inside the current configured workspace. Episodes
0 and 1 are useful negative controls: their recorded EEF x coordinate itself
goes below `-0.10 m`, so the current safety configuration correctly rejects
them. Do not widen the workspace merely to make that test pass.

## 2. Start the GPU RTC service

```bash
# absolute joint (RTC is semantically valid here)
python scripts/serve_policy.py --model pi05-joint --mode rtc

# local EEF delta
python scripts/serve_policy.py --model pi05-eef --mode rtc
```

The defaults are five flow steps, guidance weight 5, and exponential prefix
attention. Override with `--num-inference-steps`, `--max-guidance-weight`, or
`--schedule`. Each model has its own port, so both can run at once.

## 3. Model-in-the-loop zero-motion test

After forwarding the selected GPU port, run the robot-side client without
`--execute`. Arms connect read-only and no CAN motion command is sent:

```bash
python scripts/run_policy.py --model pi05-joint --mode rtc \
  --dry-run --max-steps 200

python scripts/run_policy.py --model pi05-eef --mode rtc \
  --dry-run --max-steps 200
```

Add `--save-obs runs/obs/rtc-dry` (or `--show-image`) to check what the model
is actually being shown; the server side takes `--save-obs` too and additionally
writes the post-preprocessing tensor.

PI0.5 can also be inspected in the existing visual digital twin:

```bash
/home/ysh/miniconda3/envs/robokit/bin/python \
  /home/ysh/piper_data_reviewer/run.py \
  --data "/home/ysh/robokit/datasets/stack cups" \
  --robokit-root /home/ysh/robokit \
  --robokit-config configs/piper_single.yaml \
  --policy-host 127.0.0.1 --policy-port 8080 \
  --policy-mode rtc --instruction "stack cups" \
  --port 0 --idle-timeout 30 --open-browser
```

Accept the model-in-the-loop gate only if:

- the server handshake is `rtc` and the expected `joint`/`eef_delta` action
  space;
- measured `d_init` satisfies the RTC deadline inequality;
- every response is finite `(50, 7)`;
- trace contains swaps but no deadline, workspace, action, IK, or controller
  abort;
- swap-boundary joint/EEF steps are comparable to within-chunk steps, with no
  single-step spike.

## 4. Real motion

Drop `--dry-run`. Only do this after the digital-twin and GPU dry-run gates pass:

```bash
python scripts/run_policy.py --model pi05-joint --mode rtc --max-steps 50
```

Software guards are off by default (`--safety off`); pass `--safety on` for the
protected thresholds. Start with a cleared workspace and an operator at the
emergency stop. Keep the short step limit for the first run; inspect the
generated JSONL trace before raising it or dropping to the default
`--max-steps 0` (unlimited).

For `--model pi05-eef --mode rtc`, note the open issue recorded in README
("RTC 暂停真机"): LeRobot's `RTCProcessor` treats the action vector as
chunk-alignable trajectory coordinates, which does not hold for step-local SE(3)
deltas. The client prints that warning at startup rather than blocking, so the
decision is explicit.
