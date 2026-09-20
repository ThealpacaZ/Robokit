#!/usr/bin/env python3
"""真机端唯一入口：选模型、选模式、选执行步数，跑真机推理。

    # 关节模型 + RTC，每块保证执行 15 步，总步数不限，护栏已解除
    python scripts/run_policy.py --model pi05-joint --mode rtc

    # EEF 模型 + 同步，每块执行 30 步
    python scripts/run_policy.py --model pi05-eef --mode sync --horizon 30

    # 第一次上真机：零运动链路检查
    python scripts/run_policy.py --model pi05-eef --dry-run

    # 换一条语言指令（-L = --L = --instruction），不必重启服务端
    python scripts/run_policy.py --model pi05-joint -L "Stack one cup on top of another cup"

四个开关是这个入口的全部要点：

    --model      要用哪个模型。名字在 configs/models.yaml，--list 可查。
                 它同时决定 action_space（eef_delta / joint）、默认端口、默认
                 horizon 和用哪份机器人配置，真机端不必再手写这些。
    --mode       sync = 一次推理执行一块（推理期间臂停着）；
                 rtc  = 推理与执行并发（arXiv:2506.07339）。
    --horizon    每个 chunk 承诺执行的步数。sync 下是 ChunkExecutor 的 horizon，
                 rtc 下是论文的 s_min。缺省取登记表里该模型的值。
    --max-steps  整场总共执行多少个动作步，**默认 0 = 不限制**，一直跑到回车 /
                 Ctrl-C / 安全中止。和 --horizon 不是一回事。

按回车中断执行后，**默认自动把机械臂复位到示教起点**（等价于手工跑
`scripts/reset_piper_to_demo_start.py --execute --skip-controller-reset`）：一轮跑完
就能直接跑下一轮，不必每次手工敲复位命令。数据集目录和 CAN 口默认从机器人配置推出
（collect.save_path/task_name、第一条 Piper 臂的 port），可用 --reset-dataset /
--reset-episode / --reset-port 覆盖，--no-reset-on-interrupt 关掉。复位在会话完全退出
（归位、controller reset、CAN 释放）之后才开始，Ctrl-C 与安全中止不触发它 —— 那两种
情况现场需要先被人看一眼。

安全护栏**默认解除**（--safety off）：ActionGuard、相机亮度/陈旧/冻结闸、IK 位姿
残差与伺服落后拒发、夹爪限幅全关，退出不做 controller reset 也不归位。要跑受保护
的版本用 --safety on。具体关掉了什么、以及哪些东西不属于「护栏」因而不会被关，
见 robokit/deploy/runtime.py 的 relax_safety()。

`--show-image` / `--save-obs DIR` 看这一端发出去的帧；模型预处理之后真正进模型的
那张要在服务端用 `scripts/serve_policy.py --save-obs`。

单位约定：位姿与动作全链路都是米 / 真弧度 / xyz 外旋欧拉序（见 robokit/pose.py）。
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from robokit.deploy.loops import LoopConfig, run_rtc, run_sync
from robokit.deploy.obsview import ObsView
from robokit.deploy.videorec import VideoRecorder
from robokit.deploy.registry import available, resolve_model
from robokit.deploy.runtime import (
    DeploySession,
    build_reset_command,
    override_piper_eef_backend,
    relax_safety,
    reset_to_demo_start,
    resolve_control_freq,
    resolve_reset_target,
)
from robokit.safety import ActionGuard
from robokit.utils import load_config, log


def write_rollout_meta(video_dir, *, spec, mode, horizon, instruction, status,
                       seconds, trace_path, tag) -> None:
    """一次 rollout 的名片，写在录像目录里。

    录 demo 要反复 rollout，光看时间戳分不出哪次成了、跑了哪个任务。挑片工具
    (scripts/demo_rollouts.py) 只读这个文件，所以非 demo 的录像目录不会被它管。
    写失败只告警：demo 元数据丢了是小事，不值得让已经跑完的 rollout 报错退出。
    """
    payload = {
        "recorded_at": datetime.now().isoformat(timespec="seconds"),
        "model": spec.name,
        "family": spec.family,
        "mode": mode,
        "horizon": horizon,
        "instruction": instruction,
        "status": status if status is not None else "unknown",
        "seconds": round(float(seconds), 1),
        "trace": str(trace_path),
    }
    try:
        directory = Path(video_dir)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "rollout.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        videos = sorted(p.name for p in directory.glob("*.mp4"))
        log(tag, f"demo rollout → {directory} "
                 f"(status={payload['status']}, {payload['seconds']}s, 视频 {videos or '无'})", "INFO")
        log(tag, f"挑片: python scripts/demo_rollouts.py list   保留: "
                 f"python scripts/demo_rollouts.py keep {directory.name}", "INFO")
    except Exception as exc:  # noqa: BLE001 - 元数据不该影响 rollout 结果
        log(tag, f"rollout.json 写入失败（不影响本次执行）: {exc}", "WARNING")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--list", action="store_true", help="列出登记表里的模型后退出")
    parser.add_argument("--model", help="configs/models.yaml 里的模型名")
    parser.add_argument("--registry", default=None, help="换一份模型登记表")
    parser.add_argument("--mode", choices=("sync", "rtc"), default="sync",
                        help="sync=一次推理执行一块（默认）；rtc=推理与执行并发")
    parser.add_argument("--horizon", type=int, default=None,
                        help="每个 chunk 执行的步数（rtc 下是论文 s_min）；"
                             "缺省取登记表")
    parser.add_argument("--max-steps", type=int, default=0,
                        help="整场总动作步数上限；默认 0 = 不限制")
    parser.add_argument("--safety", choices=("off", "on"), default="off",
                        help="off（默认）解除全部软件护栏；on 使用配置里的生产阈值")
    parser.add_argument("--config", default=None,
                        help="机器人配置 YAML；缺省取登记表里该模型的 robot_config")
    parser.add_argument("--host", default=None, help="推理服务器地址，缺省取配置 deploy.host")
    parser.add_argument("--port", type=int, default=None,
                        help="推理服务器端口，缺省取登记表里该模型的 port")
    parser.add_argument("--instruction", "--L", "-L", default=None,
                        help="语言指令（-L / --L 是同一个开关的短写），缺省取登记表/配置。"
                             "服务端未用 --instruction 钉死时，这里给什么就发什么")
    parser.add_argument("--action-space", choices=("joint", "eef_delta", "eef_delta_base"),
                        default=None,
                        help="覆盖登记表；服务端回报的动作空间必须与此一致，否则拒绝执行")
    parser.add_argument("--eef-backend", choices=("host_ik", "pinocchio_ik"), default=None,
                        help="仅覆盖 Piper EEF 后端；host_ik=SDK FK + SciPy，"
                             "pinocchio_ik=URDF + Pinocchio SE(3) 有界 IK")
    parser.add_argument("--chunk-base", default=None,
                        choices=("recursive", "feedback", "continuous"),
                        help="chunk 递推基准，见 robokit/executor.py。缺省 sync=recursive、"
                             "rtc=continuous（引导重叠已提供连续性，再重锚会每次交换都"
                             "累积 IK/FK 残差）。feedback 已确认是 bug（动作压缩到 0.34 倍）")
    parser.add_argument("--resync", type=float, default=None,
                        help="跟踪误差超过该米数时把基准拉回反馈位姿（continuous 的兜底，"
                             "例如 0.03）")
    parser.add_argument("--control-freq", type=float, default=None,
                        help="下发频率 Hz，覆盖配置 deploy.control_freq。频率过高会让每个"
                             "目标点还没走到就被下一个打断，落后量逐块放大直到撞"
                             "max_target_step 中止。降频是让整块动作照原路走完、只是走慢，"
                             "比减小 horizon（丢弃模型预测的后续步）更可取")
    parser.add_argument("--gripper-mode", default="raw",
                        choices=("raw", "binary", "hysteresis"))
    parser.add_argument("--gripper-squeeze", type=float, default=0.0,
                        help="闭合方向额外多压的满行程比例（0=关闭）。训练标签是夹爪的"
                             "实测开度，抓取时被物体挡住，回放该位置只贴不夹；0.05~0.10 "
                             "通常足以产生夹持力。张开方向不受影响")
    parser.add_argument("--gripper-rate", type=float, default=None,
                        help="夹爪单步变化上限（满行程比例）；0 或负数关闭。"
                             "示教数据单帧最大变化约 0.11")
    parser.add_argument("--wait-arrival", action=argparse.BooleanOptionalAction, default=None,
                        help="每步等机械臂真正走到目标再发下一条（闭环）。默认随 --safety："
                             "on 时开、off 时关。定频连发会让命令位姿跑在真实位姿前面，"
                             "而 delta 是局部坐标系量，基准姿态偏了方向就跟着转偏"
                             "（实测块内可放大到 104mm/17°）")
    parser.add_argument("--arrival-tol", type=float, default=0.0005,
                        help="等到位的位置容差 m（默认 0.5mm）。**必须远小于模型单步位移**"
                             "（实测均值约 4mm），否则臂还没走到就判定到位，每步欠一点、"
                             "累积成轨迹压缩。仿真实测路径比：3mm→0.45, 1mm→0.81, "
                             "0.5mm→0.91, 0.2mm→0.96；真机定位精度约 0.5mm")
    parser.add_argument("--arrival-timeout", type=float, default=2.5,
                        help="单步等待上限 s；超时立即中止，不再追加目标")
    parser.add_argument("--dry-run", action="store_true",
                        help="不下发运动指令，只跑链路与检查。第一次上真机必须先用它")
    parser.add_argument("--assume-safe", action=argparse.BooleanOptionalAction, default=None,
                        help="跳过开跑前的回车确认（现场安全由调用方保证）。"
                             "默认随 --safety：off 时跳过")
    parser.add_argument("--park-on-exit", action=argparse.BooleanOptionalAction, default=None,
                        help="退出前先回到开跑时的关节位姿，再做 controller reset。"
                             "默认随 --safety：on 时开、off 时关（保留终止现场）")
    parser.add_argument("--reset-on-interrupt", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="按回车中断后自动把机械臂复位到示教起点（默认开）。"
                             "只对回车中断生效：Ctrl-C 与安全中止不复位。--dry-run 下强制关闭")
    parser.add_argument("--reset-dataset", default=None,
                        help="复位读哪个数据集目录；缺省 collect.save_path/collect.task_name")
    parser.add_argument("--reset-episode", type=int, default=0,
                        help="复位到该 episode 的第一帧（默认 0）")
    parser.add_argument("--reset-port", default=None,
                        help="复位用的 CAN 口；缺省取配置里第一条 Piper 臂的 port")
    parser.add_argument("--trace", default=None, help="把每步执行写成 JSONL，供事后诊断")
    # 看图
    parser.add_argument("--show-image", action="store_true",
                        help="开窗实时显示发给模型的帧")
    parser.add_argument("--record-video", action="store_true",
                        help="把相机流全帧率录成 mp4 + 帧号 sidecar（独立线程，不碰控制循环）")
    parser.add_argument("--demo", action="store_true",
                        help="录 demo：开录像并写 rollout.json（模型/指令/时长/结果），"
                             "配 scripts/demo_rollouts.py 挑片和清冗余")
    parser.add_argument("--record-video-dir", default=None,
                        help="录像输出目录，缺省与 trace 同前缀（*-video/）")
    parser.add_argument("--save-obs", default=None, help="把发出去的帧落成 PNG 到该目录")
    parser.add_argument("--save-obs-every", type=int, default=10,
                        help="每几次推理看/落一张（默认 10）")
    # RTC 专属
    parser.add_argument("--delay-buffer-size", type=int, default=None, help="论文 b，默认 10")
    parser.add_argument("--initial-delay-steps", type=int, default=None,
                        help="论文 d_init；缺省由 warmup 实测延迟换算")
    parser.add_argument("--warmup-rounds", type=int, default=2)
    parser.add_argument("--request-timeout", type=float, default=30.0)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        for name in available(args.registry):
            print(name)
        return
    if not args.model:
        parser.error("--model 必填（或用 --list 看可选值）")
    if args.warmup_rounds < 1:
        parser.error("--warmup-rounds 至少为 1")
    if args.request_timeout <= 0:
        parser.error("--request-timeout 必须为正")

    spec = resolve_model(args.model, args.registry)
    horizon = spec.execution_horizon(args.mode, args.horizon)
    action_space = args.action_space or spec.action_space
    chunk_base = args.chunk_base or ("continuous" if args.mode == "rtc" else "recursive")
    safety_on = args.safety == "on"
    # 未显式给定的开关随 --safety 走。dry-run 例外见下。
    wait_arrival = safety_on if args.wait_arrival is None else args.wait_arrival
    park_on_exit = safety_on if args.park_on_exit is None else args.park_on_exit
    assume_safe = (not safety_on) if args.assume_safe is None else args.assume_safe

    # dry-run 把每条臂按只读连接，保证启动恢复和 policy 执行都不可能发出 CAN 运动帧。
    # 到位轮询会调用 arm.assert_healthy()，只读会话会被它拒绝；命令被故意拦截时等到位
    # 本身也没有意义。直接 --dry-run 必须自洽，不能依赖调用方记得加 --no-wait-arrival。
    if args.dry_run:
        wait_arrival = False
        park_on_exit = False

    config = load_config(args.config or spec.resolved_robot_config())
    overridden_arms = override_piper_eef_backend(config, args.eef_backend)
    relaxed = [] if safety_on else relax_safety(config)
    deploy_cfg = config["deploy"]
    host = args.host or deploy_cfg.get("host", "localhost")
    port = args.port if args.port is not None else spec.port
    instruction = (
        args.instruction
        or deploy_cfg.get("instruction")
        or spec.instruction
        or config.get("collect", {}).get("task_name", "")
    )
    if not instruction:
        parser.error("指令为空：用 -L/--instruction 给一个，或在登记表/配置里填")
    control_freq = resolve_control_freq(deploy_cfg, args.control_freq)
    guard = ActionGuard.from_config(deploy_cfg) if safety_on else None
    gripper_rate = (
        args.gripper_rate if args.gripper_rate is not None
        else deploy_cfg.get("gripper_rate", 0.12)
    )
    if not gripper_rate or gripper_rate <= 0:
        gripper_rate = None

    # dry-run 承诺零运动，复位是真运动，两者不能共存。
    reset_command = None
    if args.reset_on_interrupt and not args.dry_run:
        try:
            reset_dataset, reset_episode, reset_port = resolve_reset_target(
                config,
                dataset=args.reset_dataset,
                episode=args.reset_episode,
                port=args.reset_port,
            )
        except (FileNotFoundError, ValueError) as exc:
            parser.error(str(exc))
        reset_command = build_reset_command(reset_dataset, reset_episode, reset_port)

    trace_path = args.trace or os.path.join(
        "runs", f"deploy-{args.mode}",
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{spec.name}.jsonl",
    )

    tag = "rtc-client" if args.mode == "rtc" else "client"
    log(tag, f"model={spec.name} ({spec.family}), mode={args.mode}, "
             f"action_space={action_space}, server={host}:{port}", "INFO")
    log(tag, f"horizon={horizon}"
             f"{' (s_min)' if args.mode == 'rtc' else ''}/H={spec.chunk_size}, "
             f"max_steps={'不限' if args.max_steps <= 0 else args.max_steps}, "
             f"chunk_base={chunk_base}, control_freq={control_freq:g}Hz"
             f"{' (locked)' if deploy_cfg.get('control_freq_locked') else ''}", "INFO")
    log(tag, f"instruction={instruction!r}, config={args.config or spec.resolved_robot_config()}",
        "INFO")
    if spec.instruction and instruction != spec.instruction:
        log(tag, f"instruction 与登记表不一致（登记表：{spec.instruction!r}）。"
                 "VLA 动作完全由语言条件决定，字符串与训练时的 task 不逐字相同就是"
                 "另一个条件，行为会变；且服务端若用 --instruction 钉死过，这里给的"
                 "会被它覆盖", "WARNING")
    if overridden_arms:
        log(tag, f"EEF backend override: {args.eef_backend} for {overridden_arms}", "WARNING")
    if safety_on:
        log(tag, f"safety=on: {guard.describe() if guard else 'deploy.safety.enabled=false'}",
            "INFO")
    else:
        log(tag, "safety=off —— 以下软件护栏已解除：", "WARNING")
        for item in relaxed:
            log(tag, f"  · {item}", "WARNING")
        log(tag, "  机械臂可能撞台面、自撞或在奇异点做大角度分支跳变；"
                 "请有人守着急停", "WARNING")
    log(tag, f"wait_arrival={'on' if wait_arrival else 'off'}, "
             f"park_on_exit={'on' if park_on_exit else 'off'}, "
             f"gripper_rate={gripper_rate}, gripper_squeeze={args.gripper_squeeze}", "INFO")
    if reset_command is not None:
        where = (
            "固定示教起点（写死在 reset_piper_to_demo_start.py，不读数据集）"
            if reset_dataset is None
            else f"{reset_dataset!r} 的 episode {reset_episode} 第一帧"
        )
        log(tag, f"回车中断后自动复位到 {where}（CAN {reset_port}）；"
                 "--no-reset-on-interrupt 可关闭", "INFO")
    else:
        log(tag, "回车中断后不自动复位，机械臂停在中断位姿", "INFO")
    if args.mode == "rtc" and action_space.startswith("eef_delta") and spec.notes:
        log(tag, f"⚠ {spec.name} 在 RTC 下的已知问题：\n{spec.notes}", "WARNING")

    loop_config = LoopConfig(
        host=host,
        port=port,
        instruction=instruction,
        action_space=action_space,
        horizon=horizon,
        chunk_size=spec.chunk_size,
        control_freq=control_freq,
        fixed_control_rate=bool(deploy_cfg.get("control_freq_locked", False)),
        chunk_base=chunk_base,
        gripper_mode=args.gripper_mode,
        gripper_rate=gripper_rate,
        gripper_squeeze=args.gripper_squeeze,
        guard=guard,
        guard_tracking=not args.dry_run,
        resync=args.resync,
        wait_arrival=(
            (args.arrival_tol, args.arrival_timeout) if wait_arrival else None
        ),
        max_steps=args.max_steps,
        delay_buffer_size=int(
            args.delay_buffer_size
            if args.delay_buffer_size is not None
            else spec.rtc.get("delay_buffer_size", 10)
        ),
        initial_delay_steps=args.initial_delay_steps,
        warmup_rounds=args.warmup_rounds,
        request_timeout=args.request_timeout,
        rtc_params=spec.rtc,
    )

    view = ObsView(
        show=args.show_image,
        save_dir=args.save_obs,
        every=args.save_obs_every,
        tag=f"{tag}-obs",
    )
    run = run_rtc if args.mode == "rtc" else run_sync
    status = None
    try:
        with DeploySession(
            config,
            tag=tag,
            trace_path=trace_path,
            dry_run=args.dry_run,
            park_on_exit=park_on_exit,
            assume_safe=assume_safe,
        ) as session:
            recorder = None
            if args.record_video or args.demo:
                video_dir = args.record_video_dir or (
                    os.path.splitext(trace_path)[0] + "-video"
                )
                recorder = VideoRecorder(
                    session.robot.cameras, video_dir, tag=f"{tag}-rec"
                )
                recorder.start()
            started_at = time.time()
            try:
                status = run(session, loop_config, view=view)
            except KeyboardInterrupt:
                status = "interrupted"
                log(tag, "interrupted", "WARNING")
            finally:
                if recorder is not None:
                    recorder.stop()
                if args.demo:
                    # 一次 rollout 一份 rollout.json：挑片工具只认这个文件，没有它的
                    # 目录一律当成非 demo 产物，不会被 prune 碰到。
                    write_rollout_meta(
                        video_dir, spec=spec, mode=args.mode, horizon=horizon,
                        instruction=instruction, status=status,
                        seconds=time.time() - started_at, trace_path=trace_path, tag=tag,
                    )
    finally:
        view.close()

    # 复位脚本要独占从臂（连上后会监听总线，发现别的控制进程就拒绝执行），所以必须
    # 等 DeploySession 退出、归位与 controller reset 做完、CAN 让出来之后再跑。
    if status == "interrupted" and reset_command is not None:
        reset_to_demo_start(reset_command, tag=tag)


if __name__ == "__main__":
    main()
