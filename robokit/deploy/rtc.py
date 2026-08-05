"""RTC（Real-Time Chunking, arXiv:2506.07339）的真机侧机制。

GPU 侧的 ΠGDM 引导在 LeRobot 里；本模块只管必须留在机器人电脑上的三件事：

1. ``RTCController`` —— 论文 Algorithm 1 的控制器状态机：执行满 s_min 步后启动
   下一次推理、推理期间继续消费当前 chunk、用滚动最大值保守预测延迟、新 chunk
   到达后跳过已经过去的时间步再切入。
2. ``AsyncRequestClient`` —— 同一时刻只允许一个在飞请求的异步连接。
3. ``RTCActionExecutor`` —— 逐行进入生产 ChunkExecutor 的动作路径，保持 EEF
   delta 累加 / IK / 到位等待 / 安全闸 / 夹爪模式 / trace 完全不变。

RTC 只改变 chunk 的生成与调度，不改变一行动作怎么落到机械臂上。
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
import socket
from threading import Lock
from typing import Any

import numpy as np

from robokit.comm import BiSocket
from robokit.executor import ChunkExecutor
from robokit.utils import log


class RTCDeadlineError(RuntimeError):
    """The current action chunk ran out before background inference completed."""


@dataclass(frozen=True)
class RTCRequest:
    previous_chunk_id: str
    executed_at_start: int
    inference_delay: int


class RTCController:
    """State machine corresponding to ``GetAction`` + ``InferenceLoop``."""

    def __init__(
        self,
        *,
        prediction_horizon: int,
        min_execution_horizon: int,
        initial_delay: int,
        delay_buffer_size: int = 10,
    ):
        self.horizon = int(prediction_horizon)
        self.s_min = int(min_execution_horizon)
        self.delay_buffer_size = int(delay_buffer_size)
        if self.horizon <= 1:
            raise ValueError("prediction_horizon must be > 1")
        if not 1 <= self.s_min < self.horizon:
            raise ValueError("min_execution_horizon must satisfy 1 <= s_min < H")
        if self.delay_buffer_size <= 0:
            raise ValueError("delay_buffer_size must be positive")
        if not 0 <= int(initial_delay) <= min(self.s_min, self.horizon - self.s_min):
            raise ValueError("initial_delay must satisfy d <= s_min <= H-d")

        self.delays = deque([int(initial_delay)], maxlen=self.delay_buffer_size)
        self.chunk: np.ndarray | None = None
        self.chunk_id: str | None = None
        self.t = 0
        self._inference_start: int | None = None

    @property
    def inference_pending(self) -> bool:
        return self._inference_start is not None

    @property
    def predicted_delay(self) -> int:
        return max(self.delays)

    def initialize(self, chunk: Any, chunk_id: str) -> None:
        if self.chunk is not None:
            raise RuntimeError("RTC controller is already initialized")
        self.chunk = self._validate_chunk(chunk)
        self.chunk_id = str(chunk_id)
        self.t = 0

    def should_start_inference(self) -> bool:
        return self.chunk is not None and not self.inference_pending and self.t >= self.s_min

    def start_inference(self) -> RTCRequest:
        if not self.should_start_inference():
            raise RuntimeError(
                "inference may start only after s_min actions and with no request pending"
            )
        delay = self.predicted_delay
        overlap = self.horizon - self.t
        if delay > self.t or delay > overlap:
            raise RTCDeadlineError(
                f"RTC constraint violated: d={delay}, s={self.t}, H-s={overlap}; "
                "lower control frequency or inference latency"
            )
        self._inference_start = self.t
        return RTCRequest(
            previous_chunk_id=str(self.chunk_id),
            executed_at_start=self.t,
            inference_delay=delay,
        )

    def accept_inference(self, chunk: Any, chunk_id: str) -> int:
        if self._inference_start is None:
            raise RuntimeError("received an RTC chunk without a pending inference")
        observed_delay = self.t - self._inference_start
        if observed_delay < 0:
            raise RuntimeError("controller cursor moved backwards during inference")
        new_chunk = self._validate_chunk(chunk)
        if observed_delay >= self.horizon:
            raise RTCDeadlineError(
                f"inference consumed {observed_delay} steps, exhausting H={self.horizon}"
            )

        self.chunk = new_chunk
        self.chunk_id = str(chunk_id)
        # Algorithm 1 line 22: t <- t - s. These leading actions correspond
        # to timesteps already consumed from the previous chunk.
        self.t = observed_delay
        self.delays.append(observed_delay)
        self._inference_start = None
        return observed_delay

    def next_action(self) -> tuple[np.ndarray, int, str]:
        if self.chunk is None or self.chunk_id is None:
            raise RuntimeError("initialize the RTC controller first")
        if self.t >= self.horizon:
            raise RTCDeadlineError(
                f"action chunk {self.chunk_id!r} exhausted at H={self.horizon}"
            )
        index = self.t
        action = self.chunk[index].copy()
        self.t += 1
        return action, index, self.chunk_id

    def _validate_chunk(self, chunk: Any) -> np.ndarray:
        array = np.asarray(chunk, dtype=np.float32)
        if array.ndim != 2 or array.shape[0] != self.horizon:
            raise ValueError(
                f"action chunk must have shape (H, A) with H={self.horizon}, got {array.shape}"
            )
        if array.shape[1] <= 0 or not np.all(np.isfinite(array)):
            raise ValueError(
                "action chunk must have a positive action dimension and only finite values"
            )
        return array


class AsyncRequestClient:
    """One-in-flight asynchronous request client over robokit's BiSocket wire format."""

    def __init__(self, host: str, port: int):
        conn = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        conn.connect((host, int(port)))
        self._lock = Lock()
        self._pending: Future | None = None
        self._closed = False
        self.bisocket = BiSocket(conn, self._on_message)
        log("rtc-client", f"connected to {host}:{port}", "INFO")

    def _on_message(self, message) -> None:
        with self._lock:
            future = self._pending
            self._pending = None
        if future is None or future.done():
            log("rtc-client", "received an unexpected response", "WARNING")
            return
        if isinstance(message, dict) and message.get("error"):
            future.set_exception(RuntimeError(message["error"]))
        else:
            future.set_result(message)

    def request_async(self, payload: dict) -> Future:
        with self._lock:
            if self._closed:
                raise RuntimeError("RTC connection is closed")
            if self._pending is not None and not self._pending.done():
                raise RuntimeError("RTC permits exactly one inference request in flight")
            future = Future()
            self._pending = future
        self.bisocket.send(payload)
        return future

    def close(self) -> None:
        with self._lock:
            self._closed = True
            future = self._pending
            self._pending = None
        if future is not None and not future.done():
            future.set_exception(ConnectionError("RTC connection closed"))
        self.bisocket.close()


class RTCActionExecutor(ChunkExecutor):
    """Consume one RTC action per tick through the production action path."""

    def __init__(self, *args, action_space="eef_delta", **kwargs):
        super().__init__(*args, action_space=action_space, horizon=1, **kwargs)
        self._rtc_row = 0
        self._rtc_boundary_obs = None
        self._rtc_started = False
        self._next_deadline = None

    def begin_chunk(self, obs: dict) -> None:
        if self._rtc_started:
            self.chunk_index += 1
        self._rtc_started = True
        self._rtc_row = 0
        self._rtc_boundary_obs = obs
        # This is the same re-anchor performed by ChunkExecutor.execute at a
        # normal chunk boundary. It is deliberately absent in continuous mode.
        if self.action_space == "eef_delta" and self.chunk_base == "recursive":
            self._base.clear()

    def execute_action(self, robot, action) -> tuple[str, dict]:
        action = np.asarray(action, dtype=np.float64)
        if action.ndim != 1 or not np.all(np.isfinite(action)):
            return self._abort(
                "action_shape", f"expected one finite action row, got {action.shape}"
            )
        expected = sum(
            (arm.dof + 1) if self.action_space == "joint" else 7
            for arm in robot.arms.values()
        )
        if len(action) != expected:
            return self._abort(
                "action_shape",
                f"expected {expected} {self.action_space} values, got {len(action)}",
            )

        offset = 0
        for name in sorted(robot.arms):
            arm = robot.arms[name]
            if self.action_space == "joint":
                width = arm.dof + 1
                status = self._step_joint(
                    arm, name, action[offset:offset + width], self._rtc_row
                )
                offset += width
                if status is not None:
                    return status
                continue
            obs_pose = None
            if (
                self._rtc_row == 0
                and self._rtc_boundary_obs is not None
                and name in self._rtc_boundary_obs.get("arms", {})
            ):
                obs_pose = self._rtc_boundary_obs["arms"][name]["eef_pose"]
            status = self._step_eef(
                arm, name, action[offset:offset + 7], obs_pose, self._rtc_row
            )
            if status is not None:
                return status
            offset += 7

        self.step_index += 1
        self._rtc_row += 1
        period = 1.0 / self.control_freq
        if self.fixed_control_rate:
            now = self.clock()
            if self._next_deadline is None:
                self._next_deadline = now + period
            remaining = self._next_deadline - now
            if remaining > 0:
                self.sleep(remaining)
                self._next_deadline += period
            else:
                self._next_deadline = now + period
        else:
            self.sleep(period)
        if self.interrupt():
            return "interrupted", {"step": self.step_index}
        return "ok", {"steps": 1}


class RTCEEFDeltaExecutor(RTCActionExecutor):
    """Compatibility alias for existing tests and viewer integration."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, action_space="eef_delta", **kwargs)
