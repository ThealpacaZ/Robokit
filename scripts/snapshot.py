#!/usr/bin/env python3
"""真机端：用配置里的相机拍一张现场照片存到本地并打印路径。

    # 默认：configs/piper_single.yaml 里的全部相机，各拍一张到 runs/snapshots/
    python scripts/snapshot.py

    # 只拍某一台，换目录，多丢几帧等自动曝光 / 白平衡收敛（缺省 120 帧 ≈ 2 s）
    python scripts/snapshot.py --camera cam_high --out runs/snapshots/scene-a --warmup 240

只打开相机，不碰机械臂 / CAN。相机的构造、取帧、关闭走的是 collect.py / run_policy.py
同一条代码路径（robokit.cameras.create_camera → connect / wait_for_frame / disconnect），
不另写 pyrealsense2 调用，所以这里拍到的就是采集与推理时送出去的那种帧。

退出必须经过相机的正常 stop 路径：SIGTERM 杀正在流的 RealSense 进程跳过了
pipeline.stop()，曾把这台机器的 xHCI 控制器搞死（鼠标一起掉）。因此
* 相机关闭放在 finally 里，Ctrl-C 也走它；
* SIGTERM 被转成普通异常，同样从 finally 走出去，而不是直接被杀；
* 不要在外面套 `timeout` 命令。

相机被别的进程占着（run_policy / collect 正在跑）时给出中文说明并以退出码 2 结束，
不吐 traceback。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import os
from pathlib import Path
import signal
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robokit.cameras import create_camera
from robokit.utils import load_config, log

TAG = "snapshot"
FIRST_FRAME_TIMEOUT = 10.0   # 与 collect.wait_cameras_ready 一致：首帧最多等这么久
FRAME_TIMEOUT = 2.0          # 已出首帧之后，两帧之间超过这个间隔就当断流


class Terminated(Exception):
    """SIGTERM 到达。抛异常而不是默认被杀，让 finally 里的相机 stop 有机会跑。"""


def _on_sigterm(signum, frame):
    raise Terminated()


def _who_holds_video_devices() -> list[str]:
    """扫 /proc 找出打开了 /dev/video* 的其他进程，给「设备被占用」的报错指名道姓。

    只看得到本用户的进程（别人的 /proc/<pid>/fd 读不了），读不到就跳过；
    这是诊断信息，任何失败都不能让报错本身失败。
    """
    # 只认 RealSense 的 v4l2 节点：本机还有内置摄像头（/dev/video0/1），会议软件
    # 常年占着它，列出来只会误导。
    rs_nodes = set()
    for node in Path("/sys/class/video4linux").glob("video*"):
        try:
            if "realsense" in (node / "name").read_text().lower():
                rs_nodes.add(f"/dev/{node.name}")
        except OSError:
            continue
    holders = []
    me = os.getpid()
    for entry in os.scandir("/proc"):
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        fd_dir = Path(entry.path) / "fd"
        try:
            targets = {os.readlink(fd) for fd in fd_dir.iterdir()}
        except OSError:
            continue
        if not any(t in rs_nodes for t in targets):
            continue
        try:
            cmdline = (Path(entry.path) / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace").strip()
        except OSError:
            cmdline = "?"
        holders.append(f"PID {entry.name}: {cmdline[:120]}")
    return holders


def _explain_connect_error(name: str, cfg: dict, exc: Exception) -> str:
    """把 pyrealsense2 的 RuntimeError 翻成人能看懂的中文原因；认不出的原样附上。"""
    msg = str(exc)
    low = msg.lower()
    if any(key in low for key in ("busy", "power state", "in use", "already")):
        lines = [f"相机 {name} 已被别的进程占用，无法再次打开。",
                 "RealSense 同一时刻只允许一个进程开流；最常见的是 run_policy.py 或 collect.py 正在跑。",
                 "等那个进程正常退出（回车 / Ctrl-C，不要 kill -9）后再拍。"]
        holders = _who_holds_video_devices()
        if holders:
            lines.append("当前打开了 /dev/video* 的进程：")
            lines.extend("  " + h for h in holders)
        lines.append(f"底层错误：{msg}")
        return "\n".join(lines)
    if "no device connected" in low or "no device" in low:
        serial = cfg.get("serial")
        lines = [f"相机 {name} 没找到设备。"]
        if serial:
            # 见记忆：serial 不匹配时 librealsense 报的也是 "No device connected"，
            # 现象和没插线一模一样，所以把当前在线的序列号列出来对照。
            lines.append(f"配置里指定 serial={serial}；这个报错既可能是没插好，也可能是换过相机而序列号没改。")
            try:
                import pyrealsense2 as rs
                seen = [d.get_info(rs.camera_info.serial_number) for d in rs.context().query_devices()]
                lines.append(f"当前在线的 RealSense 序列号：{seen or '无'}")
            except Exception:  # 诊断信息，拿不到就算了
                pass
        lines.append(f"底层错误：{msg}")
        return "\n".join(lines)
    return f"相机 {name} 打开失败：{type(exc).__name__}: {msg}"


def grab_settled_frame(cam, warmup: int) -> dict:
    """丢掉前 warmup 帧再取一帧，让自动曝光 / 白平衡收敛到现场光线。

    支持帧时钟的后端（RealSense）逐帧等新帧到达，保证真的丢掉了 warmup 张新帧；
    不支持的后端（mock / ros2）只能按 receive_ts 变化数帧。
    """
    if cam.supports_frame_clock():
        seq = -1
        timeout = FIRST_FRAME_TIMEOUT
        for _ in range(warmup + 1):
            frame = cam.wait_for_frame(seq, timeout)
            if frame is None:
                err = getattr(cam, "error", None)
                raise RuntimeError(f"{timeout:.0f}s 内没有新帧" + (f"：{err}" if err else ""))
            seq = frame["seq"]
            timeout = FRAME_TIMEOUT
        return frame

    deadline = time.time() + FIRST_FRAME_TIMEOUT
    last_ts, seen, frame = None, 0, None
    while seen < warmup + 1:
        frame = cam.read()
        if frame is not None and frame["receive_ts"] != last_ts:
            last_ts = frame["receive_ts"]
            seen += 1
            deadline = time.time() + FRAME_TIMEOUT
        elif time.time() > deadline:
            raise RuntimeError("相机没有产出新帧")
        else:
            time.sleep(0.005)
    return frame


def save_jpeg(image: np.ndarray, path: Path) -> None:
    """相机给的是 RGB（Camera.read 约定），cv2 按 BGR 写盘，不转就是蓝红对调。"""
    import cv2

    ok = cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                     [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise RuntimeError(f"cv2.imwrite 写入失败：{path}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="用配置里的相机拍一张现场照片，不碰机械臂",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--config", default="configs/piper_single.yaml",
                        help="机器人配置，只读其中 robot.cameras")
    parser.add_argument("--camera", action="append", default=None,
                        help="只拍这台相机（可重复给多次）；缺省全部")
    parser.add_argument("--out", default="runs/snapshots",
                        help="输出目录，文件名 YYYYmmdd-HHMMSS-<camera>.jpg")
    # 相机参数与采集 demo 完全一致（同一个 RealSenseCamera 类、同一份配置：自动曝光 +
    # 自动白平衡），差别只在采集时相机已经流了很久、白平衡早收敛了，而这里是冷启动。
    # 实测 60 fps 下只丢 10 帧（0.17 s）拍出来 R 通道均值 62、B 通道 109，明显偏蓝；
    # 白平衡要 1–2 s 才收敛，所以缺省丢 120 帧（2 s）。
    parser.add_argument("--warmup", type=int, default=120,
                        help="丢掉前 N 帧等自动曝光 / 白平衡收敛（60 fps 下 120 帧 ≈ 2 s）")
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup 不能为负")

    config = load_config(args.config)
    cam_cfgs = config.get("robot", {}).get("cameras", {}) or {}
    if not cam_cfgs:
        log(TAG, f"{args.config} 里没有配置相机（robot.cameras 为空）", "ERROR")
        return 1
    names = args.camera or sorted(cam_cfgs)
    unknown = [n for n in names if n not in cam_cfgs]
    if unknown:
        log(TAG, f"配置里没有这些相机：{unknown}，可选：{sorted(cam_cfgs)}", "ERROR")
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")   # 多台相机共用同一时间戳，便于配对

    # 这两条信号默认会直接杀掉进程、跳过 finally；转成异常后同样走相机的 stop 路径。
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGHUP, _on_sigterm)

    saved: list[Path] = []
    status = 0
    for name in names:
        cam = create_camera(name, cam_cfgs[name])
        connected = False
        try:
            try:
                cam.connect()
                connected = True
            except RuntimeError as exc:
                log(TAG, _explain_connect_error(name, cam_cfgs[name], exc), "ERROR")
                status = 2
                continue
            frame = grab_settled_frame(cam, args.warmup)
            image = frame["image"]
            path = out_dir / f"{stamp}-{name}.jpg"
            save_jpeg(image, path)
            saved.append(path)
            # 亮度给个数，好和 deploy.camera_min/max_luminance（60–220）对照，判断现场光线
            luminance = float(np.asarray(image, dtype=np.float32).mean())
            log(TAG, f"{name}: {image.shape[1]}x{image.shape[0]} 平均亮度 {luminance:.1f} → {path.resolve()}")
        except (KeyboardInterrupt, Terminated):
            log(TAG, "收到中断，正在关闭相机……", "WARNING")
            status = 130
            break
        except RuntimeError as exc:
            log(TAG, f"{name}: 取帧失败：{exc}", "ERROR")
            status = status or 1
        finally:
            if connected:
                # 无论上面怎么退出都要走到这里：pipeline.stop() 不跑，USB 控制器可能挂死。
                cam.disconnect()

    for path in saved:
        print(path.resolve())
    return status


if __name__ == "__main__":
    sys.exit(main())
