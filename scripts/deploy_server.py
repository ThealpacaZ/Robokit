"""推理服务端：接收观测，调用 policy，返回 action chunk。

用法:
    python scripts/deploy_server.py --port 8080 --policy dummy
    python scripts/deploy_server.py --port 8080 --policy memvla_lora \\
        --policy-arg checkpoint=/path/to/RUN/checkpoints/xxx.pt \\
        --policy-arg camera=cam_high
    python scripts/deploy_server.py --policy 模块路径:类名 --policy-arg key=value ...

已注册 policy 与自定义 policy 的接口约定见 robokit/policies/__init__.py。
--policy-arg 的值按 YAML 解析（数字/布尔自动转换），透传给 policy 构造函数。

obs 格式（客户端 scripts/run_policy.py 发送）:
    {"images": {相机名: (H,W,3) uint8 RGB},
     "state":  {臂名: {"joint": (dof,), "eef_pose": (6,)|None, "gripper": float}},
     "instruction": str}

action chunk 布局与 RLDS 转换一致：按臂名排序，每臂
    joint 模式: [目标关节(dof), 夹爪]      eef_delta 模式: [局部delta位姿(6), 夹爪]
"""
import argparse
import os
import socket
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from robokit.comm import BiSocket
from robokit.policies import create_policy
from robokit.utils import log


def parse_policy_args(pairs):
    kwargs = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--policy-arg expects key=value, got '{pair}'")
        key, value = pair.split("=", 1)
        kwargs[key] = yaml.safe_load(value)  # "1.5"→float, "true"→bool, 其余保持字符串
    return kwargs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--policy", default="dummy", help='注册名（dummy/memvla_lora）或 "模块路径:类名"')
    parser.add_argument(
        "--action-space",
        choices=["joint", "eef_delta"],
        default="eef_delta",
        help="声明回包动作语义；客户端会据此阻止 EEF/joint 服务误连",
    )
    parser.add_argument("--policy-arg", action="append", default=[], metavar="KEY=VALUE",
                        help="透传给 policy 构造函数，可多次指定")
    args = parser.parse_args()

    policy = create_policy(args.policy, **parse_policy_args(args.policy_arg))
    log("server", f"policy '{args.policy}' loaded", "INFO")

    def handle(message):
        t0 = time.time()
        action_chunk = policy.infer(message)
        log("server", f"inference {time.time() - t0:.3f}s, chunk {np.shape(action_chunk)}", "INFO")
        return {
            "action_chunk": np.asarray(action_chunk),
            "server_mode": "sync",
            "action_space": args.action_space,
        }

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.host, args.port))
    server.listen(1)
    log("server", f"listening on {args.host}:{args.port}", "INFO")

    try:
        while True:
            conn, addr = server.accept()
            log("server", f"client connected: {addr}", "INFO")
            policy.reset()
            bisocket = BiSocket(conn, handle, send_back=True)
            while bisocket.running.is_set():
                time.sleep(0.5)
            log("server", "client disconnected, waiting for next", "WARNING")
    except KeyboardInterrupt:
        log("server", "shutting down", "WARNING")
    finally:
        server.close()


if __name__ == "__main__":
    main()
