"""通用小工具：彩色日志、非阻塞回车检测、配置加载。"""
import os
import select
import sys

import yaml

_LEVELS = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40}
_COLORS = {"DEBUG": "\033[94m", "INFO": "\033[92m", "WARNING": "\033[93m", "ERROR": "\033[91m"}


def log(name, msg, level="INFO"):
    """打印带颜色的日志。环境变量 INFO_LEVEL 控制最低输出级别（默认 INFO）。"""
    env_level = _LEVELS.get(os.getenv("INFO_LEVEL", "INFO").upper(), 20)
    if _LEVELS.get(level.upper(), 20) < env_level:
        return
    color = _COLORS.get(level.upper(), "")
    print(f"{color}[{level}][{name}] {msg}\033[0m")


def is_enter_pressed():
    """非阻塞检测回车键。"""
    return bool(select.select([sys.stdin], [], [], 0)[0]) and sys.stdin.read(1) == "\n"


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
