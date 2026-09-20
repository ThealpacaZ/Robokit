"""真机部署的公共运行时：观测打包、相机闸、护栏开关、会话生命周期、中断复位。

sync 与 RTC 两条循环唯一的区别是「chunk 怎么产生、怎么调度」；连接顺序、相机闸、
退出归位、trace、信号处理必须完全一致，否则一条链路上修的 bug 不会出现在另一条。
这些共同部分全部集中在本模块，两条循环只写各自的调度。

单位约定：位姿与动作全链路都是米 / 真弧度 / xyz 外旋欧拉序（见 robokit/pose.py）。
"""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import numpy as np

from robokit.comm import encode_image_jpeg
from robokit.robot import Robot
from robokit.trace import TraceWriter
from robokit.utils import is_enter_pressed, log

# 护栏解除时写进配置的“不可能触发”阈值。用有限大数而不是 inf：安全闸和 IK 都要求
# 阈值是有限值，inf 会在校验处直接被拒。
UNBOUNDED = 1e9

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESET_SCRIPT = REPO_ROOT / "scripts" / "reset_piper_to_demo_start.py"


def build_payload(obs, instruction):
    """观测 → 推理请求。图像发 JPEG，预处理由服务端 policy 负责。

    远端推理（隧道/代理）时原始帧 900KB 是 RTC 跑不起来的唯一原因：2026-08-14 在
    bjb2 链路上实测 901KB 往返 1.2-3.7s（服务端自报推理只有 0.17s），折合 d≈40-110
    步，H=50 无论 s_min 取什么都不成立；换成 JPEG 后真实画面只有 31KB，往返 0.25s、
    d≈8 步。编码本身 1ms，服务端 decode_image_maybe_jpeg 同时收原始帧和 JPEG。
    """
    return {
        "images": {cam: encode_image_jpeg(frame["image"]) for cam, frame in obs["cams"].items()},
        "state": {arm: {"joint": s["joint"], "eef_pose": s["eef_pose"], "gripper": s["gripper"]}
                  for arm, s in obs["arms"].items()},
        "instruction": instruction,
    }


def validate_inference_response(
    response,
    required_server_mode=None,
    required_action_space=None,
    required_keys=(),
):
    """Fail closed before a chunk from the wrong service reaches the executor."""
    if not isinstance(response, dict):
        raise RuntimeError(
            f"inference server returned {type(response).__name__}, expected dict"
        )
    if response.get("error"):
        raise RuntimeError(f"inference server error: {response['error']}")
    if "action_chunk" not in response:
        raise RuntimeError("inference response has no action_chunk")
    if required_server_mode is not None:
        actual = response.get("server_mode")
        if actual != required_server_mode:
            raise RuntimeError(
                f"expected {required_server_mode!r} inference service, got "
                f"server_mode={actual!r}; do not execute across a sync/RTC port mismatch"
            )
    if required_action_space is not None:
        actual = response.get("action_space")
        if actual != required_action_space:
            raise RuntimeError(
                f"expected {required_action_space!r} action space, got "
                f"action_space={actual!r}; do not execute an EEF/joint service mismatch"
            )
    missing = sorted(set(required_keys) - response.keys())
    if missing:
        raise RuntimeError(f"inference response is missing {missing}")
    return response


def resolve_control_freq(deploy_cfg, override=None):
    """Resolve control Hz and enforce an optional config-level permanent lock."""
    configured = float(deploy_cfg.get("control_freq", 30))
    if not np.isfinite(configured) or configured <= 0:
        raise ValueError(
            f"deploy.control_freq must be positive and finite, got {configured}"
        )
    locked = bool(deploy_cfg.get("control_freq_locked", False))
    if locked:
        if (
            override is not None
            and not np.isclose(float(override), configured, rtol=0.0, atol=1e-12)
        ):
            raise ValueError(
                "control frequency is permanently locked by config: "
                f"expected {configured:g}Hz, got --control-freq {float(override):g}Hz"
            )
        return configured
    resolved = configured if override is None else float(override)
    if not np.isfinite(resolved) or resolved <= 0:
        raise ValueError(f"control frequency must be positive and finite, got {resolved}")
    return resolved


def override_piper_eef_backend(config, backend):
    """仅覆盖 Piper 的 EEF 路由，供同一推理命令切换 host/firmware IK。"""
    if backend is None:
        return []
    changed = []
    for name, arm_cfg in config.get("robot", {}).get("arms", {}).items():
        if arm_cfg.get("type") == "piper":
            arm_cfg["eef_backend"] = backend
            changed.append(name)
    if not changed:
        raise ValueError(
            f"--eef-backend={backend} requested but config contains no Piper arm"
        )
    return changed


def relax_safety(config):
    """在内存里解除全部软件护栏，返回一份人读的改动清单。

    这不是「把阈值调大一点」，是把所有**拒绝/中止层**关掉，只留「policy 输出 →
    EEF/关节目标 → 下发」这一条原始链路，用来看模型在物理世界的原始行为。等价于
    以前那份 configs/piper_single_noguard.yaml，但做成开关后任何配置都能用，不必
    为每个模型再复制一份 YAML。

    ⚠ 关掉之后机械臂可能撞台面、自撞，或在奇异点做大角度分支跳变。只在有人守着
    急停的情况下用。

    **没有关掉的东西**（它们不是护栏）：
      * IK 的 max_step_deg / continuity_weight —— 这是「同一个 EEF 位姿有多组关节
        解时选哪一支」的连续性机制。实测去掉后 IK 逐帧乱跳，单步关节变化到
        103.57°，臂走出的物理轨迹已经不是 policy 输出的 EEF 轨迹，反而更不忠实。
      * joint_limits_deg —— 这是主控 Flash 里的硬件行程，不是软件阈值。超出后固件
        会以 TARGET_POS_EXCEEDS_LIMIT 拒绝整条指令，臂反而不动。
      * 相机取帧失败、非有限动作、维度不符 —— 依然是硬错误。
    """
    changes = []
    deploy = config.setdefault("deploy", {})
    deploy["safety"] = {
        "enabled": False,
        "max_delta_xyz": UNBOUNDED,
        "max_delta_rpy": UNBOUNDED,
        "max_target_step": UNBOUNDED,
        "max_tracking_err": UNBOUNDED,
        "gripper_max_rate": UNBOUNDED,
        "workspace": {
            "x": [-UNBOUNDED, UNBOUNDED],
            "y": [-UNBOUNDED, UNBOUNDED],
            "z": [-UNBOUNDED, UNBOUNDED],
        },
    }
    changes.append("ActionGuard（单步位移/旋转、目标跳变、跟踪误差、工作区盒子）")
    deploy["gripper_rate"] = 0
    changes.append("夹爪单步限幅")
    deploy["camera_min_luminance"] = 0.0
    deploy["camera_max_luminance"] = 255.0
    deploy["camera_timeout_s"] = UNBOUNDED
    deploy["camera_duplicate_run"] = 10 ** 6
    changes.append("相机亮度/过曝/陈旧/冻结闸")

    for name, arm_cfg in config.get("robot", {}).get("arms", {}).items():
        # 退出时不做 controller reset：让臂带力停在 policy 的终止位姿上，保留现场
        # 供人工观察，否则失能后会直接下垂、看不到 policy 把臂带到了哪里。
        arm_cfg["reset_on_disconnect"] = False
        # 仅直接 joint 动作生效：有限 6-D 目标逐轴钳到最近机械限位而不是整条拒绝。
        # EEF/IK 路径不受影响，仍严格拒绝。
        arm_cfg["joint_limit_mode"] = "clip"
        for backend in ("pinocchio_ik", "host_ik"):
            ik = arm_cfg.get(backend)
            if not isinstance(ik, dict):
                continue
            ik["allow_best_effort"] = True
            ik["position_tolerance_mm"] = UNBOUNDED
            ik["rotation_tolerance_deg"] = UNBOUNDED
            ik["max_seed_tracking_deg"] = UNBOUNDED
            # 限位内缩保持 0：内缩越大，反馈关节贴近限位时越容易被判越界。
            ik["limit_margin_deg"] = 0.0
        changes.append(f"{name}: IK 位姿残差/伺服落后拒发、退出 reset、关节越界拒绝")
    return changes


def wait_cameras_ready(robot, timeout=10.0):
    """等待新帧、曝光稳定且亮度落在 policy 训练现场范围内。"""
    start = time.time()
    last_capture = {name: None for name in robot.cameras}
    stats = {name: [] for name in robot.cameras}
    ready = set()
    deploy_cfg = robot.config.get("deploy", {})
    min_luminance = float(deploy_cfg.get("camera_min_luminance", 60.0))
    max_luminance = float(deploy_cfg.get("camera_max_luminance", 220.0))
    while time.time() - start < timeout:
        for name, cam in robot.cameras.items():
            error = getattr(cam, "error", None)
            if error is not None:
                raise RuntimeError(f"{name} capture failed: {error}")
            frame = cam.read()
            if (
                frame is None
                or frame["capture_ts"] == last_capture[name]
                or time.time() - frame["receive_ts"] > 0.25
            ):
                continue
            last_capture[name] = frame["capture_ts"]
            rgb = frame["image"].mean(axis=(0, 1), dtype=np.float64)
            luminance = float(rgb @ np.array([0.2126, 0.7152, 0.0722]))
            stats[name].append((luminance, rgb))
            if time.time() - start < 1.0 or len(stats[name]) < 12:
                continue
            recent_luma = np.array([row[0] for row in stats[name][-12:]])
            recent_rgb = np.stack([row[1] for row in stats[name][-12:]])
            if np.ptp(recent_luma) <= 1.5 and np.ptp(recent_rgb, axis=0).max() <= 3.0:
                if not min_luminance <= luminance <= max_luminance:
                    raise RuntimeError(
                        f"{name} settled luminance={luminance:.1f} outside "
                        f"[{min_luminance:.1f}, {max_luminance:.1f}]"
                    )
                ready.add(name)
        if len(ready) == len(robot.cameras):
            return
        time.sleep(0.005)
    pending = sorted(set(robot.cameras) - ready)
    raise RuntimeError(f"cameras not ready within {timeout}s: {pending}")


class CameraRuntimeGuard:
    """Reject stale, stopped, frozen, dark or overexposed deployment frames."""

    def __init__(self, robot):
        cfg = robot.config.get("deploy", {})
        self.robot = robot
        self.timeout = float(cfg.get("camera_timeout_s", 0.25))
        self.min_luminance = float(cfg.get("camera_min_luminance", 60.0))
        self.max_luminance = float(cfg.get("camera_max_luminance", 220.0))
        self.max_duplicate_run = int(cfg.get("camera_duplicate_run", 3))
        self.last_capture = {}
        self.last_image = {}
        self.duplicate_run = {}

    def validate(self, obs):
        diagnostics = {}
        for name, cam in self.robot.cameras.items():
            error = getattr(cam, "error", None)
            if error is not None:
                raise RuntimeError(f"{name} capture failed: {error}")
            frame = obs["cams"].get(name)
            if frame is None:
                raise RuntimeError(f"{name} returned no frame")
            age = time.time() - frame["receive_ts"]
            if age > self.timeout or age < -0.1:
                raise RuntimeError(
                    f"{name} stale frame age={age:.3f}s > {self.timeout:.3f}s"
                )
            capture = float(frame["capture_ts"])
            previous_capture = self.last_capture.get(name)
            if previous_capture is not None and capture <= previous_capture:
                raise RuntimeError(
                    f"{name} capture timestamp did not advance: "
                    f"{capture} <= {previous_capture}"
                )
            image = np.asarray(frame["image"])
            previous_image = self.last_image.get(name)
            duplicate = (
                previous_image is not None and np.array_equal(image, previous_image)
            )
            run = self.duplicate_run.get(name, 0) + 1 if duplicate else 0
            if run >= self.max_duplicate_run:
                raise RuntimeError(f"{name} returned {run} exact duplicate new frames")
            rgb = image.mean(axis=(0, 1), dtype=np.float64)
            luminance = float(rgb @ np.array([0.2126, 0.7152, 0.0722]))
            if not self.min_luminance <= luminance <= self.max_luminance:
                raise RuntimeError(
                    f"{name} luminance={luminance:.1f} outside "
                    f"[{self.min_luminance:.1f},{self.max_luminance:.1f}]"
                )
            self.last_capture[name] = capture
            self.last_image[name] = image.copy()
            self.duplicate_run[name] = run
            diagnostics[name] = {
                "age_s": age,
                "capture_ts": capture,
                "duplicate_run": run,
                "luminance": luminance,
            }
        return diagnostics


def make_dry_run(robot):
    """只读模式：照常取观测、算目标、过安全闸、写 trace，但不真的下发运动指令。

    上真机的第一次跑必须用它 —— 能在零风险下确认 chunk 数学、限幅和时序。执行器会
    保留动作/工作空间/目标跳变等检查，只跳过不适用于静止反馈的到位与跟踪检查。
    """
    for arm in robot.arms.values():
        arm.move_eef = lambda pose, gripper=None: None
        preview_joint = getattr(arm, "preview_joint", None)
        if callable(preview_joint):
            # Use the production joint quantization/clipping path without
            # controller checks, motion mode changes, gripper writes or CAN.
            arm.move_joint = (
                lambda joint, gripper=None, preview=preview_joint: preview(joint)
            )
        else:
            arm.move_joint = lambda joint, gripper=None: None
    log("client", "DRY-RUN：不会下发任何运动指令", "WARNING")


def park_arms(robot, home_joint, trace=None, speed=15, tolerance_deg=1.0, timeout=15.0):
    """退出前把各臂送回开跑时的关节位姿，再交给 disconnect 做 controller reset。

    为什么必须有这一步：`reset_on_disconnect` 的 controller reset 会让六轴失能，
    机械臂随即失力下垂。如果 policy 刚好停在杯子上方/低位，那一下就是自由下落。
    先归位到已知安全的起始位姿，再让它在那里失力，下垂幅度只有 j2/j3 的两三度。

    本函数绝不抛异常：归位失败也必须继续走 reset，否则会留下一条使能着、
    没人管的机械臂会话。
    """
    for name, arm in robot.arms.items():
        target = home_joint.get(name)
        if target is None:
            continue
        record = {"event": "park_on_exit", "arm": name,
                  "target_deg": np.degrees(target).tolist()}
        saved_speed = getattr(arm, "speed", None)
        try:
            if saved_speed is not None:
                arm.speed = min(saved_speed, speed)
            arm.move_joint(target, None)
            deadline = time.time() + timeout
            err = float("inf")
            while time.time() < deadline:
                now = np.asarray(arm.get_state()["joint"], dtype=np.float64)
                err = float(np.degrees(np.abs(now - target)).max())
                if err <= tolerance_deg:
                    break
                time.sleep(0.05)
            record.update(reached=err <= tolerance_deg, error_deg=err)
            log(name, f"park: 归位误差 {err:.3f}°"
                      f"{'' if err <= tolerance_deg else '（未到位，仍继续 reset）'}",
                "INFO" if err <= tolerance_deg else "WARNING")
        except BaseException as exc:  # 归位失败绝不能挡住 reset
            record.update(reached=False, error=f"{type(exc).__name__}: {exc}")
            log(name, f"park 失败：{exc}；仍继续 controller reset", "WARNING")
        finally:
            if saved_speed is not None:
                arm.speed = saved_speed
            if trace is not None:
                trace.write(record)


def resolve_reset_target(config, dataset=None, episode=0, port=None):
    """解析「回车中断后复位到哪个示教起点」的三个参数，并当场验证目标文件存在。

    缺省全部从机器人配置推出来，真机端不必再手写一遍：数据集目录取
    ``collect.save_path/collect.task_name``（采集脚本落盘的就是这里），CAN 口取第一条
    Piper 臂的 ``port``。校验放在开跑前而不是中断时 —— 复位命令写错了要在机械臂还停
    在示教起点时就报出来，而不是等 policy 跑完、臂停在半空才发现复位跑不起来。

    返回 ``(dataset, episode, port)``。
    """
    # 不传 --reset-dataset 就用复位脚本里写死的固定起点：这个位姿不变，为它在本机
    # 常驻一份 HDF5 只会多出「数据不在就复位不了」这个失败模式。
    if port is None:
        ports = [
            str(arm_cfg["port"])
            for arm_cfg in config.get("robot", {}).get("arms", {}).values()
            if arm_cfg.get("type") == "piper" and arm_cfg.get("port")
        ]
        if not ports:
            raise ValueError(
                "配置里没有带 port 的 Piper 臂，推不出复位用的 CAN 口；用 --reset-port 指定"
            )
        port = ports[0]
    if dataset is None:
        return None, None, str(port)
    episode = int(episode)
    demo = os.path.join(dataset, f"{episode}.hdf5")
    if not os.path.isfile(demo):
        raise FileNotFoundError(
            f"中断复位要用的示教首帧不存在：{demo}。用 --reset-dataset/--reset-episode "
            f"指到真实数据集，去掉 --reset-dataset 走固定起点，"
            f"或用 --no-reset-on-interrupt 关掉自动复位"
        )
    return dataset, episode, str(port)


def build_reset_command(dataset, episode, port, python=None):
    """拼出复位命令。

    ``--skip-controller-reset`` 是这里唯一可用的选项：另一支 ``--controller-reset``
    会阻塞在 input("输入 RESET") 上，自动流程等不到人回答；而它要清的「高跟随/示教
    遗留状态」本来也不是 policy 跑完之后的情形。``--assume-safe`` 同理跳过 MOVE 确认
    —— 复位脚本自身的固件状态、驱动使能、分段步长、到位容差、超时检查全部照常执行。
    """
    command = [python or sys.executable, str(RESET_SCRIPT)]
    # dataset 为 None = 用复位脚本写死的固定起点，命令里就不出现 --dataset/--episode。
    if dataset is not None:
        command += ["--dataset", str(dataset), "--episode", str(int(episode))]
    command += [
        "--port", str(port),
        "--execute",
        "--skip-controller-reset",
        "--assume-safe",
    ]
    return command


def reset_to_demo_start(command, tag="client"):
    """跑复位脚本，把臂送回示教起点。返回 True 表示复位脚本以 0 退出。

    **必须在 DeploySession 退出之后调用**：复位脚本会自己开一条 CAN 连接，并在首次发
    送前监听 0.3s，一旦发现总线上还有别的控制进程就拒绝执行。本进程的臂要先
    disconnect（归位 + controller reset + DisconnectPort）把从臂让出来。

    本函数绝不抛异常：复位失败只报告并交还非零结果，让调用方照常收尾，不能因为复位
    脚本起不来就把主流程也带崩。
    """
    printable = " ".join(shlex.quote(part) for part in command)
    log(tag, f"回车中断 → 自动复位到示教起点：{printable}", "INFO")
    try:
        completed = subprocess.run(command)
    except BaseException as exc:
        log(tag, f"复位脚本无法启动：{type(exc).__name__}: {exc}", "ERROR")
        log(tag, f"机械臂停在中断位姿，请手工执行：{printable}", "ERROR")
        return False
    if completed.returncode != 0:
        log(tag, f"复位失败（exit={completed.returncode}）：机械臂停在中断位姿，"
                 f"请检查上面的输出后手工执行：{printable}", "ERROR")
        return False
    log(tag, "已复位到示教起点", "INFO")
    return True


class DeploySession:
    """一次部署会话：连接 → 相机闸 → 记归位点 → 执行 → 归位 → reset → 收尾。

    sync 与 RTC 共用同一个上下文管理器，保证两条链路的连接顺序、退出路径、信号
    处理逐字一致。用法::

        with DeploySession(config, tag="client", dry_run=..., ...) as session:
            session.wait_for_start()
            ...  # 自己的调度循环，用 session.robot / session.trace / session.cameras
    """

    def __init__(self, config, *, tag, trace_path, dry_run=False,
                 park_on_exit=True, assume_safe=False):
        self.config = config
        self.tag = tag
        self.dry_run = bool(dry_run)
        self.park_on_exit = bool(park_on_exit)
        self.assume_safe = bool(assume_safe)
        self.trace_path = trace_path
        self.trace = None
        self.robot = None
        self.cameras = None
        self.home_joint = {}
        self._old_sigterm = None

    def __enter__(self):
        if self.dry_run:
            # dry-run 承诺零运动：连接阶段的关节越界自动回挪也必须禁用（只报告不动作）
            for arm_cfg in self.config["robot"].get("arms", {}).values():
                arm_cfg["auto_recover"] = False
        self.trace = TraceWriter(self.trace_path)
        self.robot = Robot(self.config)
        self._old_sigterm = signal.signal(
            signal.SIGTERM,
            lambda _signum, _frame: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
        try:
            # Validate cameras before EnableArm.  A dark/frozen camera must not
            # create an enabled arm session merely to discover it is unusable.
            self.robot.connect_cameras()
            if self.robot.cameras:
                wait_cameras_ready(self.robot)
            self.robot.connect_arms(read_only=self.dry_run)
            log(
                "robot",
                f"connected: arms={list(self.robot.arms)}"
                f"{' (read-only)' if self.dry_run else ''}, "
                f"cameras={list(self.robot.cameras)}",
                "INFO",
            )
            # 开跑前的关节位姿 = 已知安全的归位目标（调用方已 reset 到示教起点）。
            # 退出时先回到这里再让 controller reset 失能，避免在杯子上方直接失力下垂。
            if not self.dry_run and self.park_on_exit:
                self.home_joint = {
                    name: np.asarray(arm.get_state()["joint"], dtype=np.float64)
                    for name, arm in self.robot.arms.items()
                }
            if self.dry_run:
                make_dry_run(self.robot)
            self.cameras = CameraRuntimeGuard(self.robot)
        except BaseException:
            self._teardown()
            raise
        return self

    def wait_for_start(self):
        """开跑前那道回车确认问的是现场安全，由调用方承担。

        --assume-safe 只跳过这一句人机确认；相机闸、安全闸、IK 拒绝、到位超时中止
        等程序化保护是否生效由各自的开关决定，与这里无关。
        """
        if self.assume_safe:
            log(self.tag, "--assume-safe：跳过回车确认，直接开始执行", "INFO")
        else:
            log(self.tag, "按回车开始执行...", "INFO")
            while not is_enter_pressed():
                time.sleep(0.05)
        # 临时论文 demo 相机在正式执行这一刻才开始录像；普通相机没有该方法，
        # 因此生产部署行为不变。
        for camera in self.robot.cameras.values():
            start_recording = getattr(camera, "start_recording", None)
            if start_recording is not None:
                start_recording()

    def __exit__(self, exc_type, exc, tb):
        self._teardown()
        return False

    def _teardown(self):
        robot = self.robot
        if robot is not None:
            # 录像只覆盖 policy 执行，不把退出归位动作混进论文 demo。
            for camera in robot.cameras.values():
                stop_recording = getattr(camera, "stop_recording", None)
                if stop_recording is not None:
                    try:
                        stop_recording()
                    except BaseException as exc:
                        log(camera.name, f"停止 demo 录像失败：{exc}", "ERROR")
        saved_signals = {
            sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)
        }
        for sig in saved_signals:
            signal.signal(sig, signal.SIG_IGN)
        try:
            if robot is not None:
                if self.home_joint:
                    park_arms(robot, self.home_joint, trace=self.trace)
                robot.disconnect(trace=self.trace)
        finally:
            for sig, handler in saved_signals.items():
                signal.signal(sig, handler)
            if self.trace is not None:
                self.trace.close()
                log(self.tag, f"trace → {self.trace_path}", "INFO")
            if self._old_sigterm is not None:
                signal.signal(signal.SIGTERM, self._old_sigterm)
                self._old_sigterm = None
