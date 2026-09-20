#!/usr/bin/env python3
"""不接机械臂，用真实数据集帧对推理服务做一次 model-in-the-loop 自检。

服务起来了不等于服务是对的：prompt 可能对不上训练时的 task 字符串、端口可能接到了
另一个动作空间的服务、RTC 引导可能根本没生效。这些在真机上全都表现为「动作看起来
不太对」，很难当场定位。这个脚本在任何 CAN 报文之前把它们区分开。

    python scripts/probe_policy_server.py --model pi05-joint --mode rtc

默认从登记表拿端口，从对应的 LeRobot 数据集里取一帧 held-out 观测（最后 10% 的
episode，训练没见过），走和真机完全相同的 TCP 协议发过去。RTC 模式下会再发一次带
previous_chunk_id 的请求，验证 ΠGDM 引导确实被触发。

退出码 0 = 全部通过；1 = 有检查未通过（细节打在输出里）。
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robokit.comm import RequestClient
from robokit.deploy.registry import resolve_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="configs/models.yaml 里的模型名")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--mode", choices=("sync", "rtc"), default="rtc",
                        help="期望服务端处于哪种模式；不一致会判失败")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=None, help="缺省取登记表")
    parser.add_argument("--dataset", default=None,
                        help="LeRobot repo id；缺省按 action_space 推 "
                             "shaohuan1/stack_one_cup_201_{joint,eef}_pi05_lerobot")
    parser.add_argument("--dataset-root", default=None)
    parser.add_argument("--hdf5", default=None,
                        help="改用本地 robokit HDF5 的一帧当观测（不需要 lerobot），"
                             "图像按真机端同样的 JPEG 协议发送")
    parser.add_argument("--episode", type=int, default=None,
                        help="用第几个 episode 的帧；缺省取 held-out 段的第一个")
    parser.add_argument("--frame", type=int, default=20, help="episode 内第几帧")
    parser.add_argument("--instruction", "--L", "-L", default=None,
                        help="发这条指令而不是登记表里的那条（-L / --L 同义）；"
                             "上真机前先在这里试新 prompt")
    parser.add_argument("--executed", type=int, default=None,
                        help="第二次请求声明已执行掉多少步；缺省用登记表的 s_min")
    parser.add_argument("--delay", type=int, default=5, help="第二次请求声明的推理延迟步数")
    parser.add_argument("--timeout", type=float, default=120.0)
    return parser.parse_args()


def load_hdf5_observation(args, spec) -> tuple[dict, int]:
    """从 robokit 采集的 HDF5 取一帧（scripts/collect.py 的布局），走真机端同样的 JPEG 编码。"""
    import h5py

    from robokit.comm import encode_image_jpeg

    arm = spec.arm or "right_arm"
    with h5py.File(args.hdf5, "r") as f:
        image = np.asarray(f[f"observations/images/{spec.camera}"][args.frame], dtype=np.uint8)
        group = f[f"observations/{arm}"]
        eef_pose = np.asarray(group["eef_pose"][args.frame], dtype=np.float32)
        joint = np.asarray(group["joint"][args.frame], dtype=np.float32)
        gripper = float(np.asarray(group["gripper"][args.frame]).reshape(-1)[0])
    observation = {
        "images": {spec.camera: encode_image_jpeg(image)},
        "state": {arm: {"joint": joint, "eef_pose": eef_pose, "gripper": gripper}},
        "instruction": args.instruction or spec.instruction,
    }
    return observation, -1


def load_observation(args, spec) -> tuple[dict, int]:
    """从 LeRobot 数据集取一帧，拼成真机端会发的那种 obs 字典。"""
    if args.hdf5:
        return load_hdf5_observation(args, spec)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    repo_id = args.dataset or (
        "shaohuan1/stack_one_cup_201_"
        f"{'joint' if spec.action_space == 'joint' else 'eef'}_pi05_lerobot"
    )
    dataset = LeRobotDataset(repo_id, root=args.dataset_root, video_backend="pyav")
    # 训练用的是前 90% 的 episode，这里默认取被留出的那 10% 的第一个。
    episode = args.episode
    if episode is None:
        episode = dataset.num_episodes - math.ceil(dataset.num_episodes * 0.10)
    start = int(dataset.meta.episodes["dataset_from_index"][episode])
    item = dataset[start + args.frame]

    image = item[f"observation.images.{spec.camera}"]
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3):      # CHW -> HWC
        image = np.transpose(image, (1, 2, 0))
    if image.dtype != np.uint8:
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

    state = np.asarray(item["observation.state"], dtype=np.float32)
    arm = spec.arm or "right_arm"
    if spec.action_space == "joint":
        arm_state = {"joint": state[:6].copy(), "gripper": float(state[6])}
    else:
        arm_state = {"eef_pose": state[:6].copy(), "gripper": float(state[6])}

    observation = {
        "images": {spec.camera: image},
        "state": {arm: arm_state},
        "instruction": args.instruction or spec.instruction,
    }
    return observation, episode


def check(label: str, ok: bool, detail: str = "") -> bool:
    print(f"  [{'PASS' if ok else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    return ok


def inspect_reply(reply, spec, mode: str, expect_guided: bool | None) -> bool:
    if not isinstance(reply, dict):
        return check("回包是 dict", False, f"got {type(reply).__name__}")
    if "error" in reply:
        print(f"  [FAIL] 服务端报错 — {reply['error']}")
        print(reply.get("traceback", ""))
        return False

    ok = True
    ok &= check("server_mode", reply.get("server_mode") == mode,
                f"expected {mode}, got {reply.get('server_mode')!r}")
    ok &= check("action_space", reply.get("action_space") == spec.action_space,
                f"expected {spec.action_space}, got {reply.get('action_space')!r}")

    chunk = np.asarray(reply.get("action_chunk"))
    ok &= check("chunk 形状", chunk.ndim == 2 and chunk.shape[1] == 7,
                f"{chunk.shape}")
    if mode == "rtc":
        ok &= check("RTC 返回完整 H", chunk.shape[0] == spec.chunk_size,
                    f"expected {spec.chunk_size}, got {chunk.shape[0]}")
    ok &= check("动作有限", bool(np.isfinite(chunk).all()),
                f"min={np.nanmin(chunk):.4f} max={np.nanmax(chunk):.4f}")
    if expect_guided is not None:
        ok &= check("ΠGDM 引导已触发", bool(reply.get("rtc_guided")) == expect_guided,
                    f"rtc_guided={reply.get('rtc_guided')!r}")
    print(f"         服务端延迟 {float(reply.get('latency_s', 0)):.3f}s"
          f"{', chunk_id=' + str(reply.get('chunk_id')) if reply.get('chunk_id') else ''}")
    return bool(ok)


def main() -> int:
    args = parse_args()
    spec = resolve_model(args.model, args.registry)
    port = args.port if args.port is not None else spec.port
    s_min = args.executed if args.executed is not None else spec.min_execution_horizon

    print(f"model={spec.name} action_space={spec.action_space} H={spec.chunk_size} "
          f"s_min={s_min} -> {args.host}:{port} ({args.mode})")
    prompt = args.instruction or spec.instruction
    print(f"prompt={prompt!r}"
          + ("" if args.instruction is None else f"（覆盖登记表 {spec.instruction!r}）"))

    observation, episode = load_observation(args, spec)
    if args.hdf5:
        print(f"观测来自 {args.hdf5} 帧 {args.frame}（JPEG 协议）")
    else:
        print(f"观测来自 episode {episode} 帧 {args.frame}"
              f"（held-out 段），图像 {observation['images'][spec.camera].shape}")

    client = RequestClient(args.host, port, timeout=args.timeout)
    passed = True
    try:
        started = time.monotonic()
        first = client.request(dict(observation))
        print(f"\n第一次请求（冷启动，无 previous_chunk_id），往返 "
              f"{time.monotonic() - started:.3f}s：")
        passed &= inspect_reply(first, spec, args.mode,
                                expect_guided=False if args.mode == "rtc" else None)

        if args.mode == "rtc" and isinstance(first, dict) and "chunk_id" in first:
            # 论文的执行期约束：推理耗时折算成的步数 d 必须满足 d <= s_min <= H-d。
            measured = float(first.get("latency_s", 0.0))
            request = dict(observation)
            request["rtc"] = {
                "previous_chunk_id": first["chunk_id"],
                "executed_at_start": int(s_min),
                "inference_delay": int(args.delay),
            }
            started = time.monotonic()
            second = client.request(request)
            print(f"\n第二次请求（previous_chunk_id={first['chunk_id']}, "
                  f"s={s_min}, d={args.delay}），往返 {time.monotonic() - started:.3f}s：")
            passed &= inspect_reply(second, spec, args.mode, expect_guided=True)

            guided_latency = float(second.get("latency_s", measured)) if isinstance(second, dict) else measured
            control_hz = 30.0
            d_steps = math.ceil(guided_latency * control_hz)
            passed &= check(
                f"RTC 时限 d <= s_min <= H-d（d={d_steps} @30Hz）",
                d_steps <= s_min <= spec.chunk_size - d_steps,
                f"{d_steps} <= {s_min} <= {spec.chunk_size - d_steps}",
            )
    finally:
        client.close()

    print("\n" + ("全部通过，可以进入零运动 dry-run" if passed else "有检查未通过，不要接真机"))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
