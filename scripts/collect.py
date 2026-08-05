"""数据采集入口。

用法:
    python scripts/collect.py --config configs/piper_single.yaml --episodes 5
    python scripts/collect.py --config configs/mock.yaml --episodes 2 --auto-seconds 3   # 无人值守/无硬件测试

交互流程（每个 episode）:
    回车 → 开始录制 → 再次回车 → 停止并落盘。
    开了 show_image 时，回车在终端里按或在预览窗口里按都算，不用来回切焦点。
相机断联(camera_timeout 内无新帧)会自动中止并丢弃当前 episode。
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robokit.image import lememory_preprocess
from robokit.recorder import EpisodeRecorder, next_episode_index
from robokit.robot import Robot
from robokit.utils import is_enter_pressed, load_config, log


# episode 开头让预览静默几 tick，避开 recorder 建表/分配缓冲的那阵开销（见 record_episode）
PREVIEW_WARMUP_TICKS = 5


def screen_size(default=(1920, 1080)):
    """探屏幕分辨率。

    用 xdpyinfo/xrandr 而不是 tkinter：只读一行文本，不新建任何 GUI 对象 ——
    为了探个分辨率反倒再抢一次焦点就本末倒置了。探不到就按 default 走。
    """
    import re
    import subprocess

    for cmd, pattern in ((("xdpyinfo",), r"dimensions:\s+(\d+)x(\d+)"),
                         (("xrandr",), r"current\s+(\d+)\s*x\s*(\d+)")):
        try:
            out = subprocess.run(cmd, capture_output=True, text=True, timeout=2).stdout
        except (OSError, subprocess.SubprocessError):
            continue
        match = re.search(pattern, out)
        if match:
            return int(match.group(1)), int(match.group(2))
    return default


class Preview:
    """采集时的相机预览窗口。

    有两处是专门为了「开了预览也能一直按回车」：

    * 窗口在采集开始前一次性建好。GTK/Qt 后端只在**新建**窗口时把焦点抢过去，
      之后 imshow 刷新同一个窗口不会再抢；把这一次抢焦点提前到第一次提示之前，
      后面终端焦点就不会被打断。
    * 回车同时认终端 stdin 和窗口按键。窗口一旦拿到焦点，回车就进不了 stdin，
      不认窗口按键的话每采一条都得先用鼠标点回终端 —— 这正是要修的毛病。

    放大是自己 cv2.resize 出来的，不是把窗口拉大：实测本机 Qt5 后端即使
    resizeWindow 到 988x666，画面仍按图像原始尺寸居中画，窗口拉大只是加黑边。
    插值用 INTER_NEAREST —— 预览是拿来看「模型吃到的那张图」的，平滑插值会把
    拉伸产生的锯齿抹掉，反而掩盖问题；真要看得清楚就把 preview_image_size 调大，
    让像素来自传感器而不是插值（模型自己吃的仍是 224x224）。
    """

    ENTER_KEYS = (10, 13, 141)   # LF / CR / 小键盘回车，不同 highgui 后端给的码不一样

    def __init__(self, enabled, cam_names, image_size=None, preprocess=None,
                 screen_fraction=0.5, every=1):
        self.enabled = bool(enabled)
        self.every = max(1, int(every))
        self._tick = -1
        self._cv2 = None
        self._titles = {}
        self._image_size = None
        self._preprocess = preprocess
        self._fraction = float(screen_fraction)
        if not 0.0 < self._fraction <= 1.0:
            raise ValueError(
                f"collect.preview_screen_fraction must be in (0, 1], got {screen_fraction!r}"
            )
        self._box = None          # 单个窗口的目标画面尺寸 (宽, 高)
        self._display = {}        # 相机名 → 放大后的 (宽, 高)，按首帧算一次
        self._slots = {}          # 相机名 → 窗口左上角坐标
        self._moves = {}          # 相机名 → 还要再摆几次位置
        if preprocess not in (None, "lememory"):
            raise ValueError(
                f"collect.preview_preprocess must be 'lememory' or omitted, got {preprocess!r}"
            )
        if preprocess == "lememory" and image_size is None:
            image_size = [224, 224]
        if image_size is not None:
            if not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
                raise ValueError(
                    f"collect.preview_image_size must be [height, width], got {image_size!r}"
                )
            self._image_size = tuple(int(value) for value in image_size)
            if any(value <= 0 for value in self._image_size):
                raise ValueError(
                    f"collect.preview_image_size must be positive, got {image_size!r}"
                )
        if not self.enabled:
            return
        import cv2

        self._cv2 = cv2
        screen_w, screen_h = screen_size()
        self._box = (int(screen_w * self._fraction), int(screen_h * self._fraction))
        per_row = max(1, int(screen_w // self._box[0]))
        for slot, name in enumerate(cam_names):
            title = f"collect:{name}"
            # GUI_NORMAL 去掉 Qt 后端默认的工具栏/状态栏，窗口才紧贴画面而不是多出百来像素
            cv2.namedWindow(title, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
            # 从屏幕左上角开始平铺，排满一行再换下一行
            self._slots[name] = ((slot % per_row) * self._box[0],
                                 (slot // per_row) * self._box[1])
            self._moves[name] = 2
            cv2.moveWindow(title, *self._slots[name])
            self._titles[name] = title
        geometry = ""
        if self._image_size is not None and self._preprocess == "lememory":
            height, width = self._image_size
            geometry = (f"；leMemory train_aligned 预览 {width}x{height}"
                        "（整帧拉伸+0.9 中心裁剪，HDF5 仍存原图）")
        geometry += (f"；显示放大到 ≤{self._box[0]}x{self._box[1]}"
                     f"（屏幕 {screen_w}x{screen_h} 的 {self._fraction:g}），左上角起平铺")
        if self.every > 1:
            geometry += f"；每 {self.every} 帧画一次（不占采集节拍）"
        log("collect", f"预览窗口已打开{geometry}；回车在终端或预览窗口里按都算", "INFO")

    def pump(self, cams):
        """刷新画面并处理窗口事件，返回「预览窗口里按了回车」。

        cams 为 {相机名: 帧} —— 直接吃 robot.get_obs()["cams"] 的结构，尚无帧的相机
        取到 None，跳过即可。
        """
        if not self.enabled:
            return False
        # 预览是诊断功能，不该跟采集节拍抢时间：640x480 源做 train_aligned 实测出
        # 480x480 要 11.0ms/帧、出 224x224 要 6.9ms。隔 every 个 tick 画一次把这笔
        # 开销摊薄——每帧都画时 episode 中段还会偶发一个长间隔，隔帧画就只剩开头
        # 固定的那一个（见 record_episode 的 PREVIEW_WARMUP_TICKS）。
        self._tick += 1
        if self._tick % self.every:
            return False
        cv2 = self._cv2
        for name, frame in cams.items():
            title = self._titles.get(name)
            if title is None or not frame:
                continue
            image = frame["image"]
            if self._preprocess == "lememory":
                height, width = self._image_size
                image = np.asarray(
                    lememory_preprocess(image, size=(width, height), mode="train_aligned")
                )
            cv2.imshow(title, cv2.resize(cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                         self._fit(name, image),
                                         interpolation=cv2.INTER_NEAREST))
            if self._moves[name] > 0:
                # 建窗口时那一次 moveWindow 会被窗口管理器的自动摆放盖掉：AUTOSIZE
                # 窗口要等首帧到了才定尺寸，实测得等窗口真正映射出来再摆才管用。
                # 摆两次就收手，之后不再动，免得跟用户自己拖窗口打架。
                cv2.moveWindow(title, *self._slots[name])
                self._moves[name] -= 1
        return (cv2.waitKey(1) & 0xFF) in self.ENTER_KEYS

    def _fit(self, name, image):
        """按首帧算一次「等比塞进 1/4 屏那个框」的目标尺寸，之后查表。"""
        size = self._display.get(name)
        if size is None:
            height, width = image.shape[:2]
            scale = min(self._box[0] / width, self._box[1] / height)
            size = (max(1, int(width * scale)), max(1, int(height * scale)))
            self._display[name] = size
        return size

    def close(self):
        if not self.enabled:
            return
        try:
            self._cv2.destroyAllWindows()
            self._cv2.waitKey(1)        # 让窗口真正关掉，不留僵尸窗
        except Exception:
            pass


def wait_for_enter(prompt, robot=None, preview=None):
    """等回车。等待期间预览画面照常刷新，窗口里按的回车同样算数。"""
    log("collect", prompt, "INFO")
    while True:
        if is_enter_pressed():
            return
        if preview is not None and preview.enabled and robot is not None:
            if preview.pump({name: cam.read() for name, cam in robot.cameras.items()}):
                return
        time.sleep(0.03)


def wait_cameras_ready(robot, timeout=10.0):
    """等待所有相机产出第一帧，超时报错退出。"""
    start = time.time()
    while time.time() - start < timeout:
        pending = [name for name, cam in robot.cameras.items() if cam.read() is None]
        if not pending:
            return
        time.sleep(0.2)
    raise RuntimeError(f"cameras produced no frame within {timeout}s: {pending}")


def pick_frame_clock(robot, freq):
    """选出给采集循环当节拍源的相机，并算出每个 tick 该跨过几帧。

    为什么要有这一步：相机和主机是两个独立晶振。实测该 D435i 自由运行在 59.53Hz
    （周期 16.80ms），而按主机时钟睡出来的 tick 是 30.00Hz（周期 33.35ms），比值
    33.35/16.80 = 1.985 不是整数。每个 tick 只能跨过整数个新帧，于是必然是
    「2,2,2,…,1,2,2,…」——实测 stack cups 3.50%、test 3.01% 的 tick 只前进 1 帧
    （或被节拍抖动推成 3 帧）。结果是图像真实间隔在 16.8/33.6/50.4ms 之间跳，而
    timestamps/frame 一律声称 33.3ms。**只要节拍来自主机时钟这个拍频就消不掉**，
    与「读缓存最新帧」的做法无关。等帧到达而不是等时钟，间隔才是传感器周期的整数倍。

    返回 (相机名, 相机对象, decimate)；没有可用帧时钟时返回 (None, None, 0)。
    """
    for name in sorted(robot.cameras):
        cam = robot.cameras[name]
        if not cam.supports_frame_clock():
            continue
        cam_fps = float(cam.cfg.get("fps", freq))
        decimate = max(1, int(round(cam_fps / freq)))
        return name, cam, decimate
    return None, None, 0


def record_episode(robot, recorder, cfg, auto_seconds=None, preview=None):
    """录制一个 episode。返回 (帧数, 中止原因或 None)。"""
    freq = cfg["freq"]
    period = 1.0 / freq
    camera_timeout = cfg.get("camera_timeout", 1.0)
    stale_threshold = 2.0 * period

    clock_name, clock_cam, decimate = pick_frame_clock(robot, freq)
    if clock_cam is not None:
        log("collect", f"节拍源 = {clock_name}（每 tick 跨 {decimate} 帧），"
                       f"图像间隔锁到相机晶振", "INFO")
    elif robot.cameras:
        log("collect", "所有相机都不支持帧时钟，回退到主机时钟节拍："
                       "图像间隔会因两个时钟拍频而不均匀", "WARNING")

    frames = 0
    stale_count = 0
    skew_count = 0
    last_warn = 0.0
    start_time = time.monotonic()
    next_tick = start_time
    want_seq = None

    while True:
        if clock_cam is not None:
            # 帧时钟：等到下一帧真的到了才走一个 tick，间隔严格是传感器周期的整数倍
            frame = clock_cam.wait_for_frame(
                after_seq=-1 if want_seq is None else want_seq - 1,
                timeout=camera_timeout)
            if frame is None:
                return frames, f"camera disconnected: [{clock_name}]"
            seq = frame["seq"]
            if want_seq is not None and seq > want_seq:
                # 主机没跟上，跨过的帧数多于 decimate。仍是整数倍，但这一 tick 的间隔偏长
                skew_count += 1
            want_seq = seq + decimate
        else:
            # 无帧时钟的后端（mock/ros2）：维持原来的固定节拍，落后过多时跳帧对齐
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()

        frame_ts = time.time()
        obs = robot.get_obs()

        # 相机断联 → 中止本 episode
        stale_cams = robot.stale_cameras(camera_timeout)
        if stale_cams:
            return frames, f"camera disconnected: {stale_cams}"

        # 陈旧帧统计：相机帧采集时刻明显早于当前 tick，说明图像-动作错位
        for cam_name, cam_frame in obs["cams"].items():
            if frame_ts - cam_frame["receive_ts"] > stale_threshold:
                stale_count += 1
                if time.time() - last_warn > 2.0:
                    log("collect", f"{cam_name} frame is stale "
                        f"({frame_ts - cam_frame['receive_ts']:.3f}s old)", "WARNING")
                    last_warn = time.time()

        recorder.append({"frame_ts": frame_ts, "arms": obs["arms"], "cams": obs["cams"]})
        frames += 1

        # episode 开头几 tick 不画预览：这时 recorder 写线程在建 HDF5 数据集、分配
        # 64 帧缓冲，主线程再插一次预览必然吃掉一个相机周期。这一下躲不掉（试过挪到
        # tick 0/1/5，长间隔就跟着挪到哪），但让它落在开头、且只发生一次：实测
        # 每个 episode 恰好 1 个 50.4ms 间隔在这里，其余全程 33.60ms（关预览则 0 个）。
        stop_from_window = (preview.pump(obs["cams"])
                            if preview is not None and frames > PREVIEW_WARMUP_TICKS
                            else False)

        if auto_seconds is not None:
            if time.monotonic() - start_time >= auto_seconds:
                break
        elif stop_from_window or is_enter_pressed():
            break

    if stale_count:
        log("collect", f"episode had {stale_count} stale camera reads (see clean.py report)", "WARNING")
    if skew_count:
        log("collect", f"episode had {skew_count}/{frames} ticks 跨帧数多于 {decimate}"
                       "（主机没跟上相机），这些 tick 的图像间隔偏长", "WARNING")
    return frames, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--auto-seconds", type=float, default=None,
                        help="每个 episode 自动录制 N 秒后停止（无人值守/测试用，跳过回车交互）")
    args = parser.parse_args()

    config = load_config(args.config)
    cfg = config["collect"]
    task_dir = os.path.join(cfg["save_path"], cfg["task_name"])
    os.makedirs(task_dir, exist_ok=True)

    # 目录级元信息，供清洗/可视化/RLDS 转换读取
    config_json = os.path.join(task_dir, "config.json")
    if not os.path.exists(config_json):
        with open(config_json, "w", encoding="utf-8") as f:
            json.dump({"task_name": cfg["task_name"], "freq": cfg["freq"],
                       "robot": config["robot"]}, f, ensure_ascii=False, indent=2)

    robot = Robot(config)
    # 采集全程由主臂遥操作驱动，本进程只读。read_only=True 保证连接阶段一帧控制指令都不发：
    # 主从臂共用一条 CAN，采集端插一条 0x159 GripperCtrl 会把从臂夹爪从联动切成位置保持，
    # 之后夹爪就跟不上主臂的张开了。
    robot.connect(read_only=True)
    if robot.cameras:
        wait_cameras_ready(robot)

    # 建窗口要赶在第一次「按回车开始」之前：抢焦点只发生在新建窗口的那一刻，
    # 提前到这里之后，采集过程中终端焦点就不会再被抢走。
    preview = Preview(
        cfg.get("show_image", False),
        sorted(robot.cameras),
        image_size=cfg.get("preview_image_size"),
        preprocess=cfg.get("preview_preprocess"),
        screen_fraction=cfg.get("preview_screen_fraction", 0.5),
        every=cfg.get("preview_every", 2),
    )

    saved = []
    try:
        for ep in range(args.episodes):
            index = next_episode_index(task_dir)
            path = os.path.join(task_dir, f"{index}.hdf5")
            log("collect", f"episode {ep + 1}/{args.episodes} -> {path}", "INFO")

            if args.auto_seconds is None:
                wait_for_enter("按回车开始录制...", robot, preview)
            recorder = EpisodeRecorder(path, cfg["task_name"], cfg["freq"], config)
            if args.auto_seconds is None:
                log("collect", "录制中，按回车停止", "INFO")

            frames, abort_reason = record_episode(
                robot, recorder, cfg, args.auto_seconds, preview)

            if abort_reason:
                recorder.close(discard=True)
                log("collect", f"episode aborted ({abort_reason}), data discarded", "ERROR")
            elif frames < cfg.get("min_frames", 30):
                recorder.close(discard=True)
                log("collect", f"episode too short ({frames} frames), discarded", "WARNING")
            else:
                final_path = recorder.close()
                saved.append(final_path)
                log("collect", f"saved {frames} frames -> {final_path}", "INFO")
    except KeyboardInterrupt:
        log("collect", "interrupted by user", "WARNING")
    finally:
        preview.close()
        robot.disconnect()

    log("collect", f"done, {len(saved)} episodes saved in {task_dir}", "INFO")


if __name__ == "__main__":
    main()
