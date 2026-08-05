"""两条部署循环：同步（sync）与实时分块（RTC）。

两者共用 robokit/deploy/runtime.py 的连接、相机闸、退出路径，也共用
robokit/executor.py 的动作数学、IK、到位等待、安全闸、夹爪模式和 trace。区别只有
chunk 怎么产生与调度：

    sync   取观测 → 等推理返回 → 执行前 horizon 步 → 再取观测。推理期间机械臂停着。
    rtc    执行满 s_min 步后就异步发起下一次推理，当前 chunk 继续消费；新 chunk 由
           服务端用旧 chunk 的前缀做 ΠGDM 引导生成，到达后跳过已过去的时间步切入。

``max_steps`` 在两条循环里的含义一致：**总共执行多少个动作步**，0 或负数 = 不限制，
一直跑到回车 / Ctrl-C / 安全中止。它和 ``horizon`` 不是一回事 —— horizon 是每个
chunk 执行几步，max_steps 是整场执行几步。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
import time

import numpy as np

from robokit.deploy.rtc import (
    AsyncRequestClient,
    RTCActionExecutor,
    RTCController,
    RTCDeadlineError,
)
from robokit.deploy.runtime import build_payload, validate_inference_response
from robokit.executor import ChunkExecutor
from robokit.utils import is_enter_pressed, log


@dataclass
class LoopConfig:
    """一次部署循环的全部运行参数（已解析完配置与登记表，不再有 None 语义）。"""

    host: str
    port: int
    instruction: str
    action_space: str
    horizon: int                    # sync: 每块执行步数；rtc: 论文 s_min
    chunk_size: int                 # 模型预测的完整 H
    control_freq: float
    fixed_control_rate: bool = False
    chunk_base: str = "recursive"
    gripper_mode: str = "raw"
    gripper_rate: float | None = None
    guard: object | None = None
    guard_tracking: bool = True
    resync: float | None = None
    wait_arrival: tuple[float, float] | None = None
    max_steps: int = 0               # 总动作步数上限，<=0 不限
    # RTC 专属
    delay_buffer_size: int = 10
    initial_delay_steps: int | None = None
    warmup_rounds: int = 2
    request_timeout: float = 30.0
    rtc_params: dict = field(default_factory=dict)


def _remaining_steps(config: LoopConfig, executed: int) -> float:
    if config.max_steps <= 0:
        return math.inf
    return config.max_steps - executed


def _report(tag, config, status, detail, executed):
    if status == "interrupted":
        log(tag, f"用户中断，已执行 {executed} 步", "WARNING")
    elif status == "aborted":
        log(tag, f"ABORT: {detail['kind']} — {detail['message']}", "ERROR")


def run_sync(session, config: LoopConfig, view=None) -> str:
    """同步循环：一轮推理执行一块的前 ``horizon`` 步。"""
    from robokit.comm import RequestClient

    tag = "client"
    robot, trace = session.robot, session.trace
    executor = ChunkExecutor(
        action_space=config.action_space,
        control_freq=config.control_freq,
        fixed_control_rate=config.fixed_control_rate,
        horizon=config.horizon,
        chunk_base=config.chunk_base,
        gripper_mode=config.gripper_mode,
        gripper_rate=config.gripper_rate,
        guard=config.guard,
        trace=trace,
        resync_tracking_err=config.resync,
        interrupt=is_enter_pressed,
        wait_arrival=config.wait_arrival,
        guard_tracking=config.guard_tracking,
    )

    session.wait_for_start()
    client = RequestClient(config.host, config.port)
    status, detail = "ok", {}
    try:
        for round_index in itertools.count():
            remaining = _remaining_steps(config, executor.step_index)
            if remaining <= 0:
                log(tag, f"到达 --max-steps={config.max_steps}，停止", "INFO")
                break
            # 最后一块可能只允许执行剩下的几步，不要为了凑满 horizon 越过上限。
            executor.horizon = int(min(config.horizon, remaining))

            obs = robot.get_obs()
            cameras = session.cameras.validate(obs)
            if view is not None:
                view.images(
                    {name: frame["image"] for name, frame in obs["cams"].items()},
                    round_index,
                    kind="sent",
                )
            t0 = time.time()
            response = validate_inference_response(
                client.request(build_payload(obs, config.instruction)),
                "sync",
                config.action_space,
            )
            infer_s = time.time() - t0
            if trace is not None:
                trace.write({
                    "t": t0, "event": "inference", "round": round_index,
                    "latency_s": infer_s,
                    "chunk_shape": list(np.shape(response["action_chunk"])),
                    "horizon": executor.horizon,
                    "obs_pose": {
                        a: None if s["eef_pose"] is None
                        else np.asarray(s["eef_pose"]).tolist()
                        for a, s in obs["arms"].items()
                    },
                    "cameras": cameras,
                })
            status, detail = executor.execute(robot, response["action_chunk"], obs)
            if status != "ok":
                break
            if round_index % 10 == 0:
                log(tag, f"round {round_index}: {infer_s * 1000:.0f}ms, "
                         f"已执行 {executor.step_index} 步", "INFO")
    finally:
        client.close()
    _report(tag, config, status, detail, executor.step_index)
    return status


def _rtc_headroom_advice(chunk_size, s_min, delay, control_freq) -> str:
    """RTC deadline 相关报错/告警统一的下一步建议。

    ``--horizon`` 在 RTC 下是论文的 s_min，方向与 sync 相反：**越大越危险**。推理必须
    在重叠窗口 ``H - s_min`` 步之内回来，s_min 越大窗口越小。留 2 倍余量的上限就是
    ``H - 2d``。
    """
    delay = max(int(delay), 1)
    suggested = int(chunk_size) - 2 * delay
    parts = [
        f"RTC 下 --horizon 就是论文的 s_min，**越大越危险**：推理必须在重叠窗口 "
        f"H-s_min={int(chunk_size) - int(s_min)} 步内回来。"
    ]
    if suggested >= 1:
        parts.append(
            f"按 2 倍余量建议 --horizon <= {suggested}（当前 {int(s_min)}）。"
            f"或降低 --control-freq（现在 {float(control_freq):g}Hz）—— 同样的推理秒数"
            f"换算成更少的步数，窗口就变宽。"
        )
    else:
        parts.append(
            f"当前 H={int(chunk_size)} 放不下 {delay} 步的推理延迟，无论 s_min 取多少都不"
            f"成立：只能降低 --control-freq（现在 {float(control_freq):g}Hz）或改用 "
            f"--mode sync。"
        )
    return " ".join(parts)


def _rtc_payload(obs, instruction, rtc_request=None) -> dict:
    payload = build_payload(obs, instruction)
    if rtc_request is not None:
        payload["rtc"] = {
            "previous_chunk_id": rtc_request.previous_chunk_id,
            "executed_at_start": rtc_request.executed_at_start,
            "inference_delay": rtc_request.inference_delay,
        }
    return payload


def _rtc_response(response, action_space):
    return validate_inference_response(
        response, "rtc", action_space,
        required_keys=("action_chunk", "chunk_id", "prediction_horizon"),
    )


def run_rtc(session, config: LoopConfig, view=None) -> str:
    """RTC 循环：推理与执行并发，见 robokit/deploy/rtc.py。"""
    tag = "rtc-client"
    robot, trace = session.robot, session.trace
    executor = RTCActionExecutor(
        action_space=config.action_space,
        control_freq=config.control_freq,
        fixed_control_rate=config.fixed_control_rate,
        chunk_base=config.chunk_base,
        gripper_mode=config.gripper_mode,
        gripper_rate=config.gripper_rate,
        guard=config.guard,
        trace=trace,
        resync_tracking_err=config.resync,
        interrupt=is_enter_pressed,
        wait_arrival=config.wait_arrival,
        guard_tracking=config.guard_tracking,
    )

    session.wait_for_start()
    client = AsyncRequestClient(config.host, config.port)
    status, detail = "ok", {}
    controller = None
    initial_delay = 0
    try:
        initial_obs = robot.get_obs()
        initial_cameras = session.cameras.validate(initial_obs)
        if view is not None:
            view.images(
                {name: frame["image"] for name, frame in initial_obs["cams"].items()},
                0, kind="sent",
            )
        response = None
        warm_latency = None
        for warmup in range(config.warmup_rounds):
            started = time.monotonic()
            response = _rtc_response(
                client.request_async(
                    _rtc_payload(initial_obs, config.instruction)
                ).result(timeout=config.request_timeout),
                config.action_space,
            )
            warm_latency = time.monotonic() - started
            log(tag, f"warmup {warmup + 1}/{config.warmup_rounds}: {warm_latency:.3f}s", "INFO")
        assert response is not None and warm_latency is not None

        horizon = int(response["prediction_horizon"])
        max_delay = min(config.horizon, horizon - config.horizon)
        initial_delay = config.initial_delay_steps
        if initial_delay is None:
            # One extra tick is a conservative allowance for sub-timestep
            # response arrival, as ignored by the paper's discrete-time model.
            initial_delay = int(math.ceil(warm_latency * config.control_freq)) + 1
        overlap = horizon - config.horizon
        if not 0 <= initial_delay <= max_delay:
            raise RuntimeError(
                f"d_init={initial_delay} violates d <= s_min <= H-d for H={horizon}, "
                f"s_min={config.horizon}; measured warm latency was "
                f"{warm_latency:.3f}s at {config.control_freq:g}Hz. "
                f"{_rtc_headroom_advice(horizon, config.horizon, initial_delay, config.control_freq)}"
            )
        # d <= s_min <= H-d 只保证「理论上放得下一次推理」。真机上每一次往返都要落在
        # 重叠窗口 H-s_min 里，一次抖动就会耗尽当前 chunk。这里在**任何动作下发之前**
        # 把余量算出来告诉人，而不是等跑了上百步才 abort。
        if overlap < 2 * max(initial_delay, 1):
            log(
                tag,
                f"RTC 余量偏紧：重叠窗口只有 {overlap} 步 "
                f"({overlap / config.control_freq * 1000:.0f}ms)，而实测推理往返约 "
                f"{initial_delay} 步。一次稍慢的往返就会耗尽当前 chunk 并中止。"
                f"{_rtc_headroom_advice(horizon, config.horizon, initial_delay, config.control_freq)}",
                "WARNING",
            )

        controller = RTCController(
            prediction_horizon=horizon,
            min_execution_horizon=config.horizon,
            initial_delay=initial_delay,
            delay_buffer_size=config.delay_buffer_size,
        )
        controller.initialize(response["action_chunk"], response["chunk_id"])
        executor.begin_chunk(initial_obs)
        if trace is not None:
            trace.write({
                "t": time.time(), "event": "rtc_init",
                "chunk_id": response["chunk_id"],
                "prediction_horizon": horizon,
                "s_min": config.horizon,
                "d_init": initial_delay,
                "delay_buffer_size": config.delay_buffer_size,
                "warm_latency_s": warm_latency,
                "overlap_steps": overlap,
                "action_space": config.action_space,
                "max_steps": config.max_steps,
                "cameras": initial_cameras,
            })

        pending = None
        pending_started = None
        for global_step in itertools.count():
            if _remaining_steps(config, executor.step_index) <= 0:
                log(tag, f"到达 --max-steps={config.max_steps}，停止", "INFO")
                break

            if pending is not None and pending.done():
                result = _rtc_response(pending.result(), config.action_space)
                observed_delay = controller.accept_inference(
                    result["action_chunk"], result["chunk_id"]
                )
                boundary_obs = robot.get_obs()
                executor.begin_chunk(boundary_obs)
                if trace is not None:
                    trace.write({
                        "t": time.time(), "event": "rtc_swap",
                        "chunk_id": result["chunk_id"],
                        "observed_delay_steps": observed_delay,
                        "request_latency_s": time.monotonic() - pending_started,
                        "server_latency_s": result.get("latency_s"),
                    })
                pending = None
                pending_started = None

            if (
                pending is not None
                and time.monotonic() - pending_started > config.request_timeout
            ):
                raise TimeoutError(f"RTC inference exceeded {config.request_timeout:.1f}s")

            if controller.should_start_inference():
                obs = robot.get_obs()
                cameras = session.cameras.validate(obs)
                if view is not None:
                    view.images(
                        {name: frame["image"] for name, frame in obs["cams"].items()},
                        global_step, kind="sent",
                    )
                rtc_request = controller.start_inference()
                pending_started = time.monotonic()
                pending = client.request_async(
                    _rtc_payload(obs, config.instruction, rtc_request)
                )
                if trace is not None:
                    trace.write({
                        "t": time.time(), "event": "rtc_inference_start",
                        "chunk_id": rtc_request.previous_chunk_id,
                        "executed_at_start": rtc_request.executed_at_start,
                        "predicted_delay_steps": rtc_request.inference_delay,
                        "cameras": cameras,
                    })

            action, row, chunk_id = controller.next_action()
            status, detail = executor.execute_action(robot, action)
            if status != "ok":
                break
            if global_step % 25 == 0:
                log(tag, f"step={global_step}, chunk={chunk_id}, row={row}, "
                         f"d_est={controller.predicted_delay}", "INFO")
    except (RTCDeadlineError, TimeoutError) as exc:
        # 光说「chunk 耗尽」没法让人知道下一步改什么。把实测延迟和建议的 s_min
        # 一起打出来 —— 这是 RTC 唯一一个必须靠现场实测才能定的参数。
        observed = controller.predicted_delay if controller is not None else initial_delay
        advice = _rtc_headroom_advice(
            controller.horizon if controller is not None else config.chunk_size,
            config.horizon, observed, config.control_freq,
        )
        log(tag, f"RTC DEADLINE ABORT: {exc}", "ERROR")
        log(tag, f"实测推理延迟 {observed} 步。{advice}", "ERROR")
        if trace is not None:
            trace.write({
                "t": time.time(), "event": "rtc_deadline_abort", "message": str(exc),
                "observed_delay_steps": observed, "s_min": config.horizon,
                "control_freq": config.control_freq, "advice": advice,
            })
        return "aborted"
    finally:
        client.close()
    _report(tag, config, status, detail, executor.step_index)
    return status
