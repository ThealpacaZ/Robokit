"""部署时把相机流录成视频（诊断用）。

每台支持 ``wait_for_frame`` 的相机一个独立录制线程，以相机自身帧率逐帧消费
缓存，与控制循环没有任何同步点。录制跟不上时 ``wait_for_frame(last_seq)``
返回的永远是最新帧，天然跳帧——不排队、不涨内存、不反压任何人。

每写一帧同时向 sidecar JSONL 落 ``{video_frame, seq, capture_ts}``；trace 里
inference 事件带同源 capture_ts，事后可把「模型看到的那帧」对齐到视频帧号。

录像是诊断功能：任何错误（编码器打不开、磁盘满、相机中途死掉）只 WARNING
并停录，绝不打断执行。
"""
import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from robokit.utils import log


class VideoRecorder:
    def __init__(self, cameras, out_dir, tag="rec"):
        self.out_dir = Path(out_dir)
        self.tag = str(tag)
        self._stop = threading.Event()
        self._threads = []
        self._jobs = []
        for name, cam in cameras.items():
            if getattr(cam, "supports_frame_clock", lambda: False)():
                self._jobs.append((name, cam))
            else:
                log(self.tag, f"{name} 不支持逐帧消费，跳过录像", "WARNING")

    def start(self):
        if not self._jobs:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for name, cam in self._jobs:
            thread = threading.Thread(
                target=self._record, args=(name, cam), daemon=True
            )
            thread.start()
            self._threads.append(thread)
        log(self.tag, f"录像 → {self.out_dir}（{len(self._threads)} 路）", "INFO")

    def stop(self):
        self._stop.set()
        for thread in self._threads:
            thread.join(timeout=3.0)
        self._threads.clear()

    def _record(self, name, cam):
        import cv2

        writer, sidecar = None, None
        seq, written, dropped = -1, 0, 0
        try:
            sidecar = open(self.out_dir / f"{name}.frames.jsonl", "w")
            while not self._stop.is_set():
                frame = cam.wait_for_frame(seq, timeout=0.5)
                if frame is None:
                    if getattr(cam, "error", None) is not None:
                        log(self.tag, f"{name} 相机已停止，录像收尾", "WARNING")
                        break
                    # 相机断开（非 error）时 wait 会立即返回 None，避免空转。
                    time.sleep(0.02)
                    continue
                image = frame["image"]
                if writer is None:
                    height, width = image.shape[:2]
                    writer = cv2.VideoWriter(
                        str(self.out_dir / f"{name}.mp4"),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        float(getattr(cam, "fps", 30)),
                        (width, height),
                    )
                    if not writer.isOpened():
                        log(self.tag, f"{name} 编码器打不开，停录", "WARNING")
                        return
                if seq >= 0:
                    dropped += frame["seq"] - seq - 1
                seq = frame["seq"]
                writer.write(cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                sidecar.write(json.dumps({
                    "video_frame": written,
                    "seq": seq,
                    "capture_ts": frame["capture_ts"],
                }) + "\n")
                written += 1
        except Exception as exc:
            log(self.tag, f"{name} 录像失败，停录：{type(exc).__name__}: {exc}",
                "WARNING")
        finally:
            if writer is not None:
                writer.release()   # 不 release 会缺 moov atom，文件打不开
            if sidecar is not None:
                sidecar.close()
            if written:
                log(self.tag, f"{name}: 写 {written} 帧，掉 {dropped} 帧 → "
                              f"{name}.mp4", "INFO")
                _to_h264(self.out_dir / f"{name}.mp4", self.tag)


def _to_h264(path, tag):
    """把 OpenCV 写出的 mp4v（MPEG-4 Part 2）转成 H.264。

    cv2.VideoWriter 只能稳定给 mp4v，而浏览器、聊天窗口、VS Code 预览统统不认它，
    只有 H.264 能直接播。cv2 自带的 avc1 依赖 openh264 且常常打不开，所以录完用
    ffmpeg 转一遍：体积也从十几 MB 缩到 1–2 MB。ffmpeg 不在或转码失败就原样保留，
    录像是诊断功能，绝不因为这一步丢文件。
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return
    tmp = path.with_suffix(".h264.tmp.mp4")
    cmd = [ffmpeg, "-v", "error", "-y", "-i", str(path), "-c:v", "libx264",
           "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
           "-movflags", "+faststart", str(tmp)]
    try:
        subprocess.run(cmd, check=True, timeout=600, capture_output=True)
        os.replace(tmp, path)
    except Exception as exc:
        log(tag, f"{path.name} 转 H.264 失败，保留 mp4v：{type(exc).__name__}: {exc}", "WARNING")
        try:
            tmp.unlink()
        except OSError:
            pass

