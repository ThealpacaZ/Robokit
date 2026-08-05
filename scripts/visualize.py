"""episode 可视化：多相机拼接视频 + 状态/时间戳曲线图。

用法:
    python scripts/visualize.py --data datasets/任务名                # 全部 episode
    python scripts/visualize.py --data datasets/任务名 --episode 3 7  # 指定编号

输出到 <data>/viz/：
    <N>_video.mp4   所有相机横向拼接，叠加帧号与各相机帧龄（图像-动作错位一眼可见）
    <N>_curves.png  每臂关节/夹爪/EEF 曲线 + 各相机帧龄曲线
"""
import argparse
import glob
import os
import sys

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robokit.episode import EpisodeFile
from robokit.utils import log

TILE_HEIGHT = 360  # 视频里每路相机统一缩放到的高度


def render_video(ep: EpisodeFile, out_path):
    freq = ep.freq
    frame_ts = ep.frame_ts()
    receive = {cam: ep.cam_ts(cam, "receive") for cam in ep.cams}

    writer = None
    for t in range(ep.length):
        tiles = []
        for cam in ep.cams:
            img = ep.images(cam)[t]
            h, w = img.shape[:2]
            img = cv2.resize(img, (int(w * TILE_HEIGHT / h), TILE_HEIGHT))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            age_ms = (frame_ts[t] - receive[cam][t]) * 1000
            cv2.putText(img, f"{cam}  age {age_ms:.0f}ms", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            tiles.append(img)
        canvas = np.hstack(tiles)
        cv2.putText(canvas, f"frame {t}/{ep.length}", (8, TILE_HEIGHT - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        if writer is None:
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                     freq, (canvas.shape[1], canvas.shape[0]))
        writer.write(canvas)
    if writer is not None:
        writer.release()


def render_curves(ep: EpisodeFile, out_path):
    frame_ts = ep.frame_ts()
    time_axis = frame_ts - frame_ts[0]

    n_rows = 0
    for arm in ep.arms:
        n_rows += 2  # joints + gripper
        if ep.eef_pose(arm) is not None:
            n_rows += 2  # xyz + rpy
    n_rows += 1  # 相机帧龄

    fig, axes = plt.subplots(n_rows, 1, figsize=(12, 2.6 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)
    row = 0

    for arm in ep.arms:
        joint = ep.joint(arm)
        for j in range(joint.shape[1]):
            axes[row].plot(time_axis, joint[:, j], label=f"j{j + 1}", linewidth=0.9)
        axes[row].set_ylabel(f"{arm}\njoint (rad)")
        axes[row].legend(ncol=joint.shape[1], fontsize=7)
        row += 1

        axes[row].plot(time_axis, ep.gripper(arm), color="tab:purple")
        axes[row].set_ylabel(f"{arm}\ngripper")
        row += 1

        eef = ep.eef_pose(arm)
        if eef is not None:
            for i, lbl in enumerate(["x", "y", "z"]):
                axes[row].plot(time_axis, eef[:, i], label=lbl, linewidth=0.9)
            axes[row].set_ylabel(f"{arm}\nEEF pos (m)")
            axes[row].legend(ncol=3, fontsize=7)
            row += 1
            for i, lbl in enumerate(["roll", "pitch", "yaw"]):
                axes[row].plot(time_axis, eef[:, 3 + i], label=lbl, linewidth=0.9)
            axes[row].set_ylabel(f"{arm}\nEEF rot (rad)")
            axes[row].legend(ncol=3, fontsize=7)
            row += 1

    for cam in ep.cams:
        age_ms = (frame_ts - ep.cam_ts(cam, "receive")) * 1000
        axes[row].plot(time_axis, age_ms, label=cam, linewidth=0.9)
    axes[row].axhline(2000.0 / ep.freq, color="red", linestyle="--", linewidth=0.8,
                      label="stale threshold")
    axes[row].set_ylabel("frame age (ms)")
    axes[row].set_xlabel("time (s)")
    axes[row].legend(fontsize=7)

    fig.suptitle(os.path.basename(ep.path))
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="任务目录（含 N.hdf5）")
    parser.add_argument("--episode", type=int, nargs="*", default=None, help="指定 episode 编号")
    parser.add_argument("--no-video", action="store_true", help="只画曲线不导视频")
    args = parser.parse_args()

    if args.episode is not None:
        files = [os.path.join(args.data, f"{i}.hdf5") for i in args.episode]
    else:
        files = sorted(glob.glob(os.path.join(args.data, "*.hdf5")))
    files = [p for p in files if os.path.exists(p)]
    if not files:
        log("viz", f"no .hdf5 found in {args.data}", "ERROR")
        return

    out_dir = os.path.join(args.data, "viz")
    os.makedirs(out_dir, exist_ok=True)

    for path in files:
        stem = os.path.splitext(os.path.basename(path))[0]
        with EpisodeFile(path) as ep:
            curves_path = os.path.join(out_dir, f"{stem}_curves.png")
            render_curves(ep, curves_path)
            log("viz", f"{stem}: curves -> {curves_path}", "INFO")
            if not args.no_video and ep.cams:
                video_path = os.path.join(out_dir, f"{stem}_video.mp4")
                render_video(ep, video_path)
                log("viz", f"{stem}: video -> {video_path}", "INFO")


if __name__ == "__main__":
    main()
