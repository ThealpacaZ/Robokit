"""把 ``--model NAME`` 解析成一份 checkpoint 描述。

登记表在 configs/models.yaml。这里只做「查表 + 校验 + 填默认值」，不加载权重、
不碰硬件，因此服务端和真机端可以共用，纯软件测试也能直接调。
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path

from robokit.utils import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_REGISTRY = REPO_ROOT / "configs" / "models.yaml"

FAMILIES = ("pi0", "pi05", "openvla_oft")
ACTION_SPACES = ("eef_delta", "joint")


@dataclass(frozen=True)
class ModelSpec:
    """一个可部署 checkpoint 的完整描述。"""

    name: str
    family: str
    checkpoint: str
    action_space: str
    chunk_size: int
    port: int
    horizon: int
    min_execution_horizon: int
    robot_config: str
    camera: str
    arm: str | None
    instruction: str
    device: str
    rtc: dict = field(default_factory=dict)
    notes: str = ""

    def execution_horizon(self, mode: str, override: int | None = None) -> int:
        """本模式下每个 chunk 承诺执行的步数。

        sync 下就是 ChunkExecutor 的 horizon；RTC 下是论文的 s_min —— 两者语义
        一致（「这一块我保证执行多少步」），因此共用同一个 ``--horizon`` 开关。
        """
        if override is not None:
            value = int(override)
        else:
            value = int(self.horizon if mode == "sync" else self.min_execution_horizon)
        if value <= 0:
            raise ValueError(f"horizon must be positive, got {value}")
        if value > self.chunk_size:
            raise ValueError(
                f"horizon={value} exceeds {self.name} chunk_size={self.chunk_size}；"
                "模型没有预测那么多步"
            )
        if mode == "rtc" and not 1 <= value < self.chunk_size:
            raise ValueError(
                f"RTC 的 s_min 必须满足 1 <= s_min < H={self.chunk_size}, got {value}"
            )
        return value

    def resolved_robot_config(self) -> str:
        path = Path(self.robot_config)
        if not path.is_absolute():
            path = REPO_ROOT / path
        return str(path)


def _registry_path(path: str | os.PathLike | None) -> Path:
    return Path(path) if path is not None else DEFAULT_REGISTRY


def load_registry(path: str | os.PathLike | None = None) -> dict[str, ModelSpec]:
    """读登记表并展开 defaults。任何一条记录不合法都在这里立刻报错。"""
    registry_file = _registry_path(path)
    document = load_config(registry_file) or {}
    defaults = dict(document.get("defaults") or {})
    default_rtc = dict(defaults.pop("rtc", None) or {})
    entries = document.get("models") or {}
    if not entries:
        raise ValueError(f"{registry_file}: 登记表里没有任何 models 条目")

    specs: dict[str, ModelSpec] = {}
    for name, raw in entries.items():
        entry = {**defaults, **(raw or {})}
        rtc = {**default_rtc, **(entry.pop("rtc", None) or {})}
        missing = [key for key in ("family", "checkpoint", "action_space") if not entry.get(key)]
        if missing:
            raise ValueError(f"{registry_file}: 模型 {name!r} 缺少必填字段 {missing}")
        family = str(entry["family"])
        if family not in FAMILIES:
            raise ValueError(
                f"{registry_file}: 模型 {name!r} 的 family={family!r} 不在 {FAMILIES}"
            )
        action_space = str(entry["action_space"])
        if action_space not in ACTION_SPACES:
            raise ValueError(
                f"{registry_file}: 模型 {name!r} 的 action_space={action_space!r} "
                f"不在 {ACTION_SPACES}"
            )
        chunk_size = int(entry.get("chunk_size", 50))
        if chunk_size <= 1:
            raise ValueError(f"{registry_file}: 模型 {name!r} 的 chunk_size 必须 > 1")
        horizon = int(entry.get("horizon", chunk_size))
        s_min = int(entry.get("min_execution_horizon", chunk_size // 2))
        specs[str(name)] = ModelSpec(
            name=str(name),
            family=family,
            checkpoint=str(entry["checkpoint"]),
            action_space=action_space,
            chunk_size=chunk_size,
            port=int(entry.get("port", 8080)),
            horizon=horizon,
            min_execution_horizon=s_min,
            robot_config=str(entry.get("robot_config", "configs/piper_single.yaml")),
            camera=str(entry.get("camera", "cam_high")),
            arm=None if entry.get("arm") in (None, "") else str(entry["arm"]),
            instruction=str(entry.get("instruction", "")),
            device=str(entry.get("device", "cuda")),
            rtc=rtc,
            notes=str(entry.get("notes", "") or "").strip(),
        )
    return specs


def available(path: str | os.PathLike | None = None) -> list[str]:
    return sorted(load_registry(path))


def resolve_model(name: str, path: str | os.PathLike | None = None) -> ModelSpec:
    registry = load_registry(path)
    try:
        return registry[name]
    except KeyError:
        raise SystemExit(
            f"未知模型 {name!r}；登记表 {_registry_path(path)} 里可选："
            f" {', '.join(sorted(registry))}"
        ) from None
