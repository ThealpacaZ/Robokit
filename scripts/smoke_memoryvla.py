#!/usr/bin/env python3
"""GPU 服务器上不经 TCP、不接机械臂，直接加载登记表里的 MemoryVLA 模型跑几帧。

    python scripts/smoke_memoryvla.py --model lamem-v2 --frames /root/frames.npz
    python scripts/smoke_memoryvla.py --model lamem-v2            # 没有真实帧就用随机图

frames.npz 由 scripts/smoke_memoryvla.py --pack <hdf5> 在本机生成（几帧 cam_high + 状态），
用真实画面比随机噪声更能暴露预处理/通道序的问题。输出每次推理耗时、动作块形状、
每维范围与夹爪取值，最后按 (N, 7) 与有限性判 PASS/FAIL；退出码 0/1。
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robokit.deploy.registry import resolve_model


def pack(hdf5: str, out: str, camera: str, arm: str, indices) -> None:
    import h5py

    with h5py.File(hdf5, "r") as f:
        images = np.stack([f[f"observations/images/{camera}"][i] for i in indices])
        eef = np.stack([f[f"observations/{arm}/eef_pose"][i] for i in indices])
        joint = np.stack([f[f"observations/{arm}/joint"][i] for i in indices])
        gripper = np.stack([np.asarray(f[f"observations/{arm}/gripper"][i]).reshape(-1)[0] for i in indices])
        task = f.attrs.get("task_name", "")
    np.savez_compressed(out, images=images, eef_pose=eef, joint=joint, gripper=gripper,
                        task=str(task), source=str(hdf5), indices=np.asarray(indices))
    print(f"packed {len(indices)} frames from {hdf5} ({task!r}) -> {out}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="lamem-v2")
    parser.add_argument("--registry", default=None)
    parser.add_argument("--checkpoint", default=None, help="覆盖登记表里的 checkpoint 路径")
    parser.add_argument("--frames", default=None, help="--pack 生成的 npz；缺省用随机图")
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--instruction", "-L", default=None)
    parser.add_argument("--pack", default=None, metavar="HDF5", help="本机：从 HDF5 打包几帧到 --out")
    parser.add_argument("--out", default="frames.npz")
    parser.add_argument("--indices", default="0,60,120,180")
    parser.add_argument("--camera", default="cam_high")
    parser.add_argument("--arm", default="right_arm")
    args = parser.parse_args()

    if args.pack:
        pack(args.pack, args.out, args.camera, args.arm, [int(x) for x in args.indices.split(",")])
        return 0

    spec = resolve_model(args.model, args.registry)
    if spec.family != "memoryvla":
        raise SystemExit(f"{spec.name} 的 family={spec.family}，这个冒烟脚本只跑 memoryvla")
    instruction = args.instruction or spec.instruction

    if args.frames:
        blob = np.load(args.frames, allow_pickle=False)
        images = blob["images"]
        states = [{"joint": blob["joint"][i], "eef_pose": blob["eef_pose"][i], "gripper": float(blob["gripper"][i])}
                  for i in range(len(images))]
        print(f"frames: {args.frames} task={blob['task']!s} n={len(images)} shape={images.shape[1:]}")
    else:
        rng = np.random.default_rng(0)
        images = rng.integers(0, 255, size=(args.rounds, 480, 640, 3), dtype=np.uint8)
        states = [{"joint": np.zeros(6, np.float32), "eef_pose": np.zeros(6, np.float32), "gripper": 0.5}] * args.rounds
        print("frames: 随机噪声（只验通路，不验语义）")

    from robokit.policies.memoryvla import MemoryVLAPolicy

    t0 = time.time()
    policy = MemoryVLAPolicy(
        checkpoint=args.checkpoint or spec.checkpoint,
        camera=spec.camera, action_space=spec.action_space, device=spec.device,
        **spec.policy_args,
    )
    print(f"loaded in {time.time() - t0:.0f}s: {policy.describe()}")
    print(f"instruction={instruction!r}")

    ok = True
    policy.reset()
    for i in range(min(args.rounds, len(images))):
        obs = {"images": {spec.camera: images[i]}, "state": {args.arm: states[i]}, "instruction": instruction}
        t1 = time.time()
        chunk = policy.infer(obs)
        dt = time.time() - t1
        finite = bool(np.isfinite(chunk).all())
        ok &= finite and chunk.ndim == 2 and chunk.shape[1] == 7 and chunk.shape[0] == spec.chunk_size
        np.set_printoptions(precision=4, suppress=True, linewidth=140)
        print(f"\n[{i}] {dt:.3f}s chunk {chunk.shape} finite={finite}")
        print(f"    xyz  min {chunk[:, :3].min(0)} max {chunk[:, :3].max(0)}  (m)")
        print(f"    rpy  min {chunk[:, 3:6].min(0)} max {chunk[:, 3:6].max(0)}  (rad)")
        print(f"    grip {chunk[:, 6]}")
    import torch

    print(f"\nGPU 显存峰值 {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
