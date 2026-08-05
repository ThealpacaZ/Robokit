"""Piper 真机写入会话的 controller-reset 生命周期。

本模块不导入 :mod:`piper_sdk`，因此可用 fake SDK 做纯软件单测。调用方仍然负责：

* 在第一条控制帧之前取得现场确认并调用 :meth:`ControllerResetGuard.arm`；
* 确保 controller reset 导致的瞬间失力不会让机械臂坠落；
* 在 guard 完成后调用 ``DisconnectPort()``。

``controller_reset_and_verify`` 只调用一次 ``ResetPiper``。成功的定义不是“函数没有
抛错”，而是 reset 后收到了新的状态和低速反馈，并在稳定窗口内持续看到六轴失能、
``arm_status == 0``、``err_code == 0``。
"""

from __future__ import annotations

import sys
import time
from typing import Any


class ControllerResetError(RuntimeError):
    """Controller reset 未能得到可验证的安全末态。"""

    def __init__(self, message: str, record: dict[str, Any]):
        super().__init__(message)
        self.record = record


def _plain_number(value: Any) -> int | float:
    """把 SDK 的枚举/标量转换成可写入 JSON trace 的普通数字。"""
    try:
        return int(value)
    except (TypeError, ValueError):
        return float(value)


def _snapshot(sdk: Any) -> dict[str, Any]:
    status_wrapper = sdk.GetArmStatus()
    status = status_wrapper.arm_status
    low_wrapper = sdk.GetArmLowSpdInfoMsgs()
    enabled = [
        bool(
            getattr(low_wrapper, f"motor_{index}")
            .foc_status.driver_enable_status
        )
        for index in range(1, 7)
    ]
    snapshot = {
        "status_timestamp": float(status_wrapper.time_stamp),
        "low_timestamp": float(low_wrapper.time_stamp),
        "arm_status": int(status.arm_status),
        "err_code": int(status.err_code),
        "enabled": enabled,
    }
    # 这些字段有助于诊断，但较旧 fake/SDK 不一定提供。
    for name in ("ctrl_mode", "mode_feed", "motion_status"):
        if hasattr(status, name):
            snapshot[name] = _plain_number(getattr(status, name))
    return snapshot


def _write_trace(trace: Any, record: dict[str, Any]) -> None:
    if trace is not None:
        trace.write(record)


def _finish_failure(
    record: dict[str, Any],
    trace: Any,
    failure: str,
    message: str,
    started_monotonic: float,
    cause: BaseException | None = None,
) -> ControllerResetError:
    record.update(
        {
            "verified": False,
            "failure": failure,
            "error_type": (
                type(cause).__name__
                if cause is not None
                else "ControllerResetError"
            ),
            "error": message,
            "duration_s": time.monotonic() - started_monotonic,
            "finished_at": time.time(),
        }
    )
    _write_trace(trace, record)
    error = ControllerResetError(message, record)
    if cause is not None:
        error.__cause__ = cause
    return error


def controller_reset_and_verify(
    sdk: Any,
    trace: Any = None,
    timeout: float = 8,
    stable_s: float = 0.5,
) -> dict[str, Any]:
    """Reset Piper once and verify a fresh, stable, fully-disabled state.

    Args:
        sdk: 已连接的 ``C_PiperInterface_V2`` 或兼容 fake。
        trace: 可选的、提供 ``write(dict)`` 的 trace。
        timeout: 等待验证成功的总秒数。
        stable_s: 安全状态必须连续保持的秒数。正值还要求稳定窗口内 status/low
            时间戳各至少再次前进一次，避免把冻结的 SDK 缓存误判为稳定反馈。

    Returns:
        可直接写入 JSON 的结构化记录。

    Raises:
        ValueError: 参数无效。
        ControllerResetError: reset 调用、反馈读取或验证失败。异常的 ``record`` 属性
            包含同样写入 trace 的失败记录。
    """
    timeout = float(timeout)
    stable_s = float(stable_s)
    if not timeout > 0:
        raise ValueError("timeout 必须为正数")
    if stable_s < 0:
        raise ValueError("stable_s 不能为负数")
    if stable_s >= timeout:
        raise ValueError("stable_s 必须小于 timeout")

    started_monotonic = time.monotonic()
    record: dict[str, Any] = {
        "event": "controller_reset",
        "attempted": False,
        "verified": False,
        "timeout_s": timeout,
        "stable_s": stable_s,
        "started_at": time.time(),
        "pre": None,
        "post": None,
    }

    pre = None
    try:
        pre = _snapshot(sdk)
    except BaseException as exc:
        # Cleanup must still attempt ResetPiper exactly once after control has
        # started, even when the receive thread/getters are already broken.
        # Without a pre-reset baseline, verification below establishes a
        # post-reset baseline and requires later timestamps to advance.
        record["precheck_error"] = {
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    record["pre"] = pre

    # attempted 在调用前置位：即使 SDK 在发送期间抛错，guard 也绝不会重复 reset。
    record["attempted"] = True
    try:
        sdk.ResetPiper()
    except BaseException as exc:
        error = _finish_failure(
            record,
            trace,
            "reset_error",
            f"ResetPiper 调用失败：{exc}",
            started_monotonic,
            exc,
        )
        raise error from exc

    deadline = started_monotonic + timeout
    stable_since: float | None = None
    stable_first_status_timestamp: float | None = None
    stable_first_low_timestamp: float | None = None
    last_snapshot = pre
    post_baseline_status_timestamp: float | None = None
    post_baseline_low_timestamp: float | None = None
    poll_s = min(0.02, max(0.001, timeout / 50.0))

    while True:
        now = time.monotonic()
        if now >= deadline:
            record["post"] = last_snapshot
            message = (
                f"{timeout:.3f}s 内未验证 controller reset 安全末态；"
                f"最后反馈={last_snapshot}"
            )
            raise _finish_failure(
                record,
                trace,
                "timeout",
                message,
                started_monotonic,
            )

        try:
            current = _snapshot(sdk)
        except BaseException as exc:
            record["post"] = last_snapshot
            error = _finish_failure(
                record,
                trace,
                "feedback_error",
                f"controller reset 后读取反馈失败：{exc}",
                started_monotonic,
                exc,
            )
            raise error from exc
        last_snapshot = current
        record["post"] = current

        if pre is not None:
            fresh = (
                current["status_timestamp"] > pre["status_timestamp"]
                and current["low_timestamp"] > pre["low_timestamp"]
            )
        elif post_baseline_status_timestamp is None:
            post_baseline_status_timestamp = current["status_timestamp"]
            post_baseline_low_timestamp = current["low_timestamp"]
            fresh = False
        else:
            fresh = (
                current["status_timestamp"]
                > post_baseline_status_timestamp
                and current["low_timestamp"]
                > post_baseline_low_timestamp
            )
        settled = (
            fresh
            and not any(current["enabled"])
            and current["arm_status"] == 0
            and current["err_code"] == 0
        )
        if settled:
            if stable_since is None:
                stable_since = now
                stable_first_status_timestamp = current[
                    "status_timestamp"
                ]
                stable_first_low_timestamp = current["low_timestamp"]

            elapsed_stable = now - stable_since
            feedback_advanced = (
                stable_s == 0
                or (
                    current["status_timestamp"]
                    > stable_first_status_timestamp
                    and current["low_timestamp"]
                    > stable_first_low_timestamp
                )
            )
            if elapsed_stable >= stable_s and feedback_advanced:
                record.update(
                    {
                        "verified": True,
                        "failure": None,
                        "duration_s": (
                            time.monotonic() - started_monotonic
                        ),
                        "stable_observed_s": elapsed_stable,
                        "finished_at": time.time(),
                    }
                )
                _write_trace(trace, record)
                return record
        else:
            stable_since = None
            stable_first_status_timestamp = None
            stable_first_low_timestamp = None

        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(poll_s, remaining))


class ControllerResetGuard:
    """在 context/finally 中提供 exactly-once controller reset。

    ``arm()`` 必须紧挨在第一条控制帧之前。未 arm 的预览、dry-run 或用户取消路径不会
    发送 reset。``close()`` 会自动识别当前正在传播的异常：reset 也失败时把说明附加到
    原异常而不覆盖它；没有原异常时则正常抛出 :class:`ControllerResetError`。

    本类刻意不调用 ``DisconnectPort``。推荐用法::

        guard = ControllerResetGuard(sdk, trace)
        try:
            with guard:
                guard.arm()
                sdk.MotionCtrl_2(...)
                ...
        finally:
            sdk.DisconnectPort()
    """

    def __init__(
        self,
        sdk: Any,
        trace: Any = None,
        timeout: float = 8,
        stable_s: float = 0.5,
    ):
        self.sdk = sdk
        self.trace = trace
        self.timeout = timeout
        self.stable_s = stable_s
        self._armed = False
        self._attempted = False
        self._record: dict[str, Any] | None = None
        self._error: BaseException | None = None
        self._noted_exception: BaseException | None = None

    @property
    def armed(self) -> bool:
        return self._armed

    @property
    def attempted(self) -> bool:
        return self._attempted

    @property
    def record(self) -> dict[str, Any] | None:
        return self._record

    @property
    def error(self) -> BaseException | None:
        return self._error

    def arm(self) -> "ControllerResetGuard":
        if self._attempted:
            raise RuntimeError(
                "controller reset 已尝试，不能重新 arm 同一个 guard"
            )
        self._armed = True
        return self

    def reset_once(self) -> dict[str, Any] | None:
        """执行至多一次 reset；重复调用复用首次结果或异常。"""
        if not self._armed:
            return None
        if self._attempted:
            if self._error is not None:
                raise self._error
            return self._record

        # 在进入函数前置位，覆盖参数错误、precheck 错误和 ResetPiper 抛错。
        self._attempted = True
        try:
            self._record = controller_reset_and_verify(
                self.sdk,
                trace=self.trace,
                timeout=self.timeout,
                stable_s=self.stable_s,
            )
            return self._record
        except BaseException as exc:
            self._error = exc
            if isinstance(exc, ControllerResetError):
                self._record = exc.record
            raise

    def close(self) -> dict[str, Any] | None:
        """适用于 ``finally`` 的关闭操作，不覆盖正在传播的主异常。"""
        active_exception = sys.exc_info()[1]
        try:
            return self.reset_once()
        except BaseException as reset_error:
            if (
                active_exception is not None
                and active_exception is not reset_error
            ):
                if self._noted_exception is not active_exception:
                    note = (
                        "controller reset cleanup 同时失败："
                        f"{type(reset_error).__name__}: {reset_error}"
                    )
                    add_note = getattr(active_exception, "add_note", None)
                    if add_note is not None:
                        add_note(note)
                    self._noted_exception = active_exception
                return self._record
            raise

    def __enter__(self) -> "ControllerResetGuard":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        # 在 __exit__ 内 sys.exc_info() 不保证跨 Python 实现都保留 exc，因此显式采用参数。
        try:
            self.reset_once()
        except BaseException as reset_error:
            if exc is None:
                raise
            if self._noted_exception is not exc:
                note = (
                    "controller reset cleanup 同时失败："
                    f"{type(reset_error).__name__}: {reset_error}"
                )
                add_note = getattr(exc, "add_note", None)
                if add_note is not None:
                    add_note(note)
                self._noted_exception = exc
        return False
