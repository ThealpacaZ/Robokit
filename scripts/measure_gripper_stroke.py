#!/usr/bin/env python3
"""只读连接 Piper，实时打印夹爪行程（mm），用来量满行程。

    python scripts/measure_gripper_stroke.py --port can0 --seconds 20

跑起来后用示教/手动把夹爪张到最大、再合到最小，结束时打印最大/最小读数。
把最大值填到机器人配置 arms.<name>.gripper_full_mm，之后采集的 gripper 就落在 [0,1]。
本脚本 read_only 连接，不向 CAN 发任何控制帧。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robokit.arms.piper import Piper


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", default="can0")
    parser.add_argument("--seconds", type=float, default=20.0)
    args = parser.parse_args()

    arm = Piper("right_arm", {"port": args.port, "dof": 6})
    arm.connect(read_only=True)
    lo, hi = float("inf"), float("-inf")
    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            raw = arm.sdk.GetArmGripperMsgs().gripper_state.grippers_angle
            mm = raw / 1000.0
            lo, hi = min(lo, mm), max(hi, mm)
            print(f"\r夹爪 {mm:7.2f} mm   (min {lo:.2f}, max {hi:.2f})", end="", flush=True)
            time.sleep(0.05)
    finally:
        print()
        arm.disconnect()
    print(f"满行程建议值 gripper_full_mm: {hi:.1f}（最小 {lo:.2f} mm）")


if __name__ == "__main__":
    main()
