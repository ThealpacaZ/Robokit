#!/usr/bin/env python3
"""GPU 服务端唯一入口：按登记表加载一个模型，用 sync 或 RTC 方式对外提供推理。

    python scripts/serve_policy.py --model pi05-joint --mode rtc
    python scripts/serve_policy.py --model pi05-eef  --mode sync --port 8080

模型名字在 configs/models.yaml 里，`--list` 可以看全部可选值。端口默认取登记表里
那个模型的 port，因此两个模型可以同时起在各自端口上，真机端只换 --model 就切换。

回包里带 server_mode 与 action_space，真机端逐条校验：sync/RTC 端口接错、
EEF/joint 服务接错都会在任何动作下发之前 fail closed。

`--save-obs DIR` 会把「模型端拿到的图像」落成 PNG —— 收到的原始帧和预处理之后真正
进模型的那张各一份，用来排除「模型看到的画面本身就不对」。
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import os
from pathlib import Path
import socket
import sys
import time
import traceback

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robokit.comm import BiSocket
from robokit.deploy.obsview import ObsView
from robokit.deploy.registry import available, resolve_model
from robokit.policies.lerobot_dit import LeRobotDiTPolicy
from robokit.utils import log

REQUIRED_FILES = ("config.json", "policy_preprocessor.json", "policy_postprocessor.json")


def check_checkpoint(path: str) -> None:
    """本地 checkpoint 先做文件级体检；HF repo id 交给 LeRobot 自己解析。"""
    directory = Path(path).expanduser()
    if not directory.is_dir():
        log("serve", f"{path} 不是本地目录，按 HF repo id 处理", "INFO")
        return
    for name in REQUIRED_FILES:
        if not (directory / name).is_file() or (directory / name).stat().st_size == 0:
            raise SystemExit(f"checkpoint 不完整，缺 {name}: {directory}")
    if (directory / "adapter_config.json").is_file():
        if not (directory / "adapter_model.safetensors").is_file():
            raise SystemExit(f"adapter 权重缺失: {directory}/adapter_model.safetensors")
    elif not (directory / "model.safetensors").is_file():
        raise SystemExit(f"全量权重缺失: {directory}/model.safetensors")


class SyncService:
    """一次推理返回一块动作；机械臂在推理期间停着等。"""

    mode = "sync"

    def __init__(self, policy: LeRobotDiTPolicy):
        self.policy = policy

    def reset(self) -> None:
        self.policy.reset()

    def handle(self, message):
        start = time.monotonic()
        try:
            chunk = self.policy.infer(message)
            latency = time.monotonic() - start
            log("serve", f"sync 推理 {latency:.3f}s, chunk {chunk.shape}", "INFO")
            return {
                "action_chunk": np.asarray(chunk, dtype=np.float32),
                "latency_s": latency,
                "server_mode": "sync",
                "action_space": self.policy.action_space,
            }
        except Exception as exc:
            log("serve", f"请求失败: {type(exc).__name__}: {exc}", "ERROR")
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }


class RTCService:
    """一条 TCP 连接的归一化 chunk 缓存 + ΠGDM 引导推理。

    协议：
      * 第一个请求不带 rtc.previous_chunk_id，生成 A_init。
      * 之后的请求指明服务端缓存的某个**归一化** chunk，加上发起推理时已经执行掉的
        动作数，服务端据此做前缀引导。
      * 回包是物理单位的 robokit 动作加一个不透明 chunk id。
    """

    mode = "rtc"

    def __init__(self, policy: LeRobotDiTPolicy, cache_size: int = 4):
        self.policy = policy
        self.cache_size = int(cache_size)
        self.chunks: OrderedDict[str, object] = OrderedDict()
        self.sequence = 0

    def reset(self) -> None:
        self.policy.reset()
        self.chunks.clear()
        self.sequence = 0

    def handle(self, message):
        start = time.monotonic()
        try:
            if not isinstance(message, dict):
                raise TypeError(f"request must be a dict, got {type(message).__name__}")
            rtc = message.get("rtc") or {}
            previous_id = rtc.get("previous_chunk_id")
            executed = int(rtc.get("executed_at_start", 0))
            delay = int(rtc.get("inference_delay", 0))
            previous = None
            if previous_id is not None:
                try:
                    previous = self.chunks[str(previous_id)]
                except KeyError as exc:
                    raise KeyError(
                        f"unknown/expired previous_chunk_id={previous_id!r}; "
                        f"available={list(self.chunks)}"
                    ) from exc

            chunk, normalized = self.policy.infer_rtc(
                message,
                previous_chunk=previous,
                executed_at_start=executed,
                inference_delay=delay,
            )
            chunk_id = f"chunk-{self.sequence:08d}"
            self.sequence += 1
            self.chunks[chunk_id] = normalized
            while len(self.chunks) > self.cache_size:
                self.chunks.popitem(last=False)

            latency = time.monotonic() - start
            log(
                "serve",
                f"rtc 推理 {latency:.3f}s, chunk={chunk_id}, previous={previous_id}, "
                f"s={executed}, d_est={delay}",
                "INFO",
            )
            return {
                "action_chunk": np.asarray(chunk, dtype=np.float32),
                "chunk_id": chunk_id,
                "prediction_horizon": int(len(chunk)),
                "latency_s": latency,
                "rtc_guided": previous is not None,
                "server_mode": "rtc",
                "action_space": self.policy.action_space,
            }
        except Exception as exc:
            log("serve", f"请求失败: {type(exc).__name__}: {exc}", "ERROR")
            return {
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=8),
            }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="列出登记表里的模型后退出")
    parser.add_argument("--model", help="configs/models.yaml 里的模型名")
    parser.add_argument("--registry", default=None, help="换一份模型登记表")
    parser.add_argument("--mode", choices=("sync", "rtc"), default="sync",
                        help="sync=一次一块（默认）；rtc=ΠGDM 引导的异步分块")
    parser.add_argument("--checkpoint", default=None, help="覆盖登记表里的 checkpoint 路径")
    parser.add_argument("--bind", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=None, help="缺省取登记表里该模型的 port")
    parser.add_argument("--horizon", type=int, default=None,
                        help="只截断回包长度；缺省返回模型预测的完整 H，由真机端 "
                             "--horizon 决定执行几步。RTC 必须返回完整 H，此项无效")
    parser.add_argument("--camera", default=None)
    parser.add_argument("--arm", default=None)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--action-space", choices=("eef_delta", "joint"), default=None,
                        help="覆盖登记表；改了它就等于声明这个 checkpoint 训的是另一种动作")
    parser.add_argument("--device", default=None)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False,
                        help="torch.compile 首次推理会多花几分钟，默认关")
    # RTC 论文参数
    parser.add_argument("--num-inference-steps", type=int, default=None, help="论文 n")
    parser.add_argument("--max-guidance-weight", type=float, default=None, help="论文 beta")
    parser.add_argument("--schedule", choices=("EXP", "LINEAR", "ZEROS", "ONES"), default=None)
    parser.add_argument("--chunk-cache", type=int, default=4)
    # 看模型端拿到的图像
    parser.add_argument("--save-obs", default=None,
                        help="把收到的帧和预处理后真正进模型的那张落成 PNG 到该目录")
    parser.add_argument("--save-obs-every", type=int, default=10,
                        help="每几次推理落一张（默认 10）")
    args = parser.parse_args()

    if args.list:
        for name in available(args.registry):
            print(name)
        return
    if not args.model:
        parser.error("--model 必填（或用 --list 看可选值）")
    if args.chunk_cache < 2:
        parser.error("--chunk-cache 至少为 2")

    spec = resolve_model(args.model, args.registry)
    rtc_defaults = spec.rtc
    checkpoint = args.checkpoint or spec.checkpoint
    port = args.port if args.port is not None else spec.port
    action_space = args.action_space or spec.action_space
    check_checkpoint(checkpoint)

    if spec.notes:
        log("serve", f"{spec.name} 备注：\n{spec.notes}", "WARNING")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    view = ObsView(
        save_dir=args.save_obs, every=args.save_obs_every, tag="serve-obs"
    )
    policy = LeRobotDiTPolicy(
        checkpoint=checkpoint,
        family=spec.family,
        camera=args.camera or spec.camera,
        arm=args.arm if args.arm is not None else spec.arm,
        instruction=args.instruction if args.instruction is not None else (spec.instruction or None),
        # RTC 必须返回完整 H：控制器按 H 对齐时间步，截断会让 prefix 引导错位。
        horizon=None if args.mode == "rtc" else args.horizon,
        action_space=action_space,
        expected_chunk_size=spec.chunk_size,
        device=args.device or spec.device,
        compile_model=bool(args.compile),
        rtc=args.mode == "rtc",
        rtc_num_inference_steps=int(
            args.num_inference_steps
            if args.num_inference_steps is not None
            else rtc_defaults.get("num_inference_steps", 5)
        ),
        rtc_max_guidance_weight=float(
            args.max_guidance_weight
            if args.max_guidance_weight is not None
            else rtc_defaults.get("max_guidance_weight", 5.0)
        ),
        rtc_schedule=str(
            args.schedule or rtc_defaults.get("prefix_attention_schedule", "EXP")
        ),
        view=view,
    )
    log("serve", f"loaded {spec.name}: {policy.describe()}", "INFO")
    log("serve", f"checkpoint={checkpoint}", "INFO")

    service = (
        RTCService(policy, cache_size=args.chunk_cache)
        if args.mode == "rtc"
        else SyncService(policy)
    )

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.bind, port))
    server.listen(1)
    log("serve", f"mode={service.mode} listening on {args.bind}:{port}", "INFO")
    try:
        while True:
            conn, address = server.accept()
            log("serve", f"client connected: {address}", "INFO")
            # 新连接 = 新 episode：清 policy 记忆和 RTC chunk 缓存。
            service.reset()
            channel = BiSocket(conn, service.handle, send_back=True)
            while channel.running.is_set():
                time.sleep(0.5)
    except KeyboardInterrupt:
        log("serve", "shutting down", "WARNING")
    finally:
        view.close()
        server.close()


if __name__ == "__main__":
    main()
