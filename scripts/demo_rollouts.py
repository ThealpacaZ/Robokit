#!/usr/bin/env python3
"""挑 demo 片子、清冗余 rollout。

`run_policy.py --demo` 每跑一次就在 runs/deploy-*/ 下留一个 `*-video/` 目录：
里面是各相机的 mp4、帧号 sidecar 和 rollout.json（模型/指令/结果/时长）。
录 demo 要反复 rollout，满意的只有一两条，其余都是垃圾，却又不能无脑按时间删
（好的那条往往不是最后一条）。

    python scripts/demo_rollouts.py list                 # 看都录了什么
    python scripts/demo_rollouts.py keep 20260916-091530-pi05-joint-alltask-video
    python scripts/demo_rollouts.py keep --last          # 保留最近那次
    python scripts/demo_rollouts.py drop <名字>          # 取消保留
    python scripts/demo_rollouts.py prune                # 删掉没保留的（先列出来，--yes 才真删）
    python scripts/demo_rollouts.py prune --keep-last 3  # 再额外留最近 3 次

保留标记是目录里的一个空文件 KEEP，所以标记跟着录像走，移动目录不会丢。
只有含 rollout.json 的目录才会被这个工具管：--record-video 产生的诊断录像不受影响。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys

KEEP_MARKER = "KEEP"
META_NAME = "rollout.json"
DEFAULT_ROOT = Path("runs")


def find_rollouts(root: Path) -> list[Path]:
    """所有含 rollout.json 的录像目录，按录制时间从旧到新。"""
    found = [p.parent for p in root.glob("**/" + META_NAME)]
    return sorted(set(found), key=lambda p: (read_meta(p).get("recorded_at", ""), p.name))


def read_meta(directory: Path) -> dict:
    try:
        return json.loads((directory / META_NAME).read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - 坏的元数据不该让整个列表挂掉
        return {}


def dir_size_mb(directory: Path) -> float:
    return sum(p.stat().st_size for p in directory.rglob("*") if p.is_file()) / 1e6


def kept(directory: Path) -> bool:
    return (directory / KEEP_MARKER).exists()


def describe(directory: Path, root: Path) -> str:
    meta = read_meta(directory)
    videos = sorted(p.name for p in directory.glob("*.mp4"))
    instruction = str(meta.get("instruction", ""))
    # 指令里那截 " <control mode> ... <control mode>" 对挑片没用，占满整行
    instruction = instruction.split(" <control mode>")[0]
    return (
        f"{'[KEEP] ' if kept(directory) else '       '}"
        f"{directory.relative_to(root)}\n"
        f"         {meta.get('recorded_at', '?')}  {meta.get('model', '?')}"
        f"  {meta.get('mode', '?')}  H={meta.get('horizon', '?')}"
        f"  {meta.get('status', '?')}  {meta.get('seconds', '?')}s"
        f"  {dir_size_mb(directory):.0f}MB  {len(videos)} 个视频\n"
        f"         {instruction[:96]!r}"
    )


def resolve(root: Path, names: list[str]) -> list[Path]:
    """名字可以是目录名、相对路径，或者完整路径。"""
    rollouts = find_rollouts(root)
    by_name = {p.name: p for p in rollouts}
    by_rel = {str(p.relative_to(root)): p for p in rollouts}
    resolved, missing = [], []
    for name in names:
        candidate = by_name.get(name) or by_rel.get(name)
        if candidate is None:
            path = Path(name)
            if (path / META_NAME).is_file():
                candidate = path
        if candidate is None:
            missing.append(name)
        else:
            resolved.append(candidate)
    if missing:
        raise SystemExit(f"找不到这些 rollout: {missing}（先跑 list 看名字）")
    return resolved


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="搜索根目录，默认 runs/")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("list", help="列出全部 rollout")

    keep = sub.add_parser("keep", help="标记保留")
    keep.add_argument("names", nargs="*")
    keep.add_argument("--last", action="store_true", help="保留最近一次")

    drop = sub.add_parser("drop", help="取消保留")
    drop.add_argument("names", nargs="+")

    prune = sub.add_parser("prune", help="删掉没保留的")
    prune.add_argument("--keep-last", type=int, default=0, help="额外保留最近 N 次")
    prune.add_argument("--yes", action="store_true", help="真的删除（默认只列出要删什么）")

    args = parser.parse_args()
    root = args.root
    rollouts = find_rollouts(root)

    if args.command == "list":
        if not rollouts:
            print(f"{root} 下没有 demo rollout（跑 run_policy.py --demo 才会产生）")
            return
        for directory in rollouts:
            print(describe(directory, root))
        total = sum(dir_size_mb(d) for d in rollouts)
        marked = sum(1 for d in rollouts if kept(d))
        print(f"\n共 {len(rollouts)} 次，保留 {marked} 次，合计 {total:.0f} MB")
        return

    if args.command == "keep":
        targets = list(args.names)
        if args.last:
            if not rollouts:
                raise SystemExit("没有任何 rollout")
            targets.append(str(rollouts[-1]))
        if not targets:
            raise SystemExit("给个名字，或者用 --last")
        for directory in resolve(root, targets):
            (directory / KEEP_MARKER).touch()
            print(f"已保留 {directory.relative_to(root)}")
        return

    if args.command == "drop":
        for directory in resolve(root, args.names):
            (directory / KEEP_MARKER).unlink(missing_ok=True)
            print(f"已取消保留 {directory.relative_to(root)}")
        return

    # prune
    protected = {d for d in rollouts if kept(d)}
    if args.keep_last > 0:
        protected |= set(rollouts[-args.keep_last:])
    doomed = [d for d in rollouts if d not in protected]
    if not doomed:
        print(f"没有可删的（{len(rollouts)} 次全部受保护）")
        return
    freed = sum(dir_size_mb(d) for d in doomed)
    for directory in doomed:
        print(f"{'删除' if args.yes else '将删除'} {directory.relative_to(root)}"
              f"  {dir_size_mb(directory):.0f}MB  {read_meta(directory).get('status', '?')}")
        if args.yes:
            shutil.rmtree(directory)
            trace = read_meta(directory).get("trace")
            # trace 已经随目录一起没了参考价值；它在录像目录外面，顺手删掉
            if trace and Path(trace).is_file() and Path(trace).parent == directory.parent:
                Path(trace).unlink()
    print(f"\n{'已释放' if args.yes else '可释放'} {freed:.0f} MB"
          f"（保留 {len(protected)} 次）" + ("" if args.yes else "；加 --yes 真正删除"))


if __name__ == "__main__":
    sys.exit(main())
