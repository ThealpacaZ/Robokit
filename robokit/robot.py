"""Robot：由 YAML 配置装配任意数量机械臂 + 相机，提供统一观测/控制接口。"""
import sys

from robokit.arms import create_arm
from robokit.cameras import create_camera
from robokit.utils import log


class Robot:
    def __init__(self, config: dict):
        self.config = config
        robot_cfg = config["robot"]
        self.arms = {name: create_arm(name, cfg) for name, cfg in robot_cfg.get("arms", {}).items()}
        self.cameras = {name: create_camera(name, cfg) for name, cfg in robot_cfg.get("cameras", {}).items()}
        if not self.arms:
            raise ValueError("config must define at least one arm")

    def connect(self, read_only=False):
        """read_only=True 时机械臂只读不下发（遥操作采集），相机不受影响。"""
        try:
            self.connect_cameras()
            self.connect_arms(read_only=read_only)
        except BaseException:
            # 连接是事务性的：相机先成功、机械臂后失败时也必须停掉相机采集线程和
            # 已创建的 SDK 接收线程，否则解释器退出时 librealsense 会直接 abort。
            self.disconnect()
            raise
        log("robot", f"connected: arms={list(self.arms)}{' (read-only)' if read_only else ''}, "
                     f"cameras={list(self.cameras)}", "INFO")

    def connect_cameras(self):
        for cam in self.cameras.values():
            cam.connect()

    def connect_arms(self, read_only=False):
        for arm in self.arms.values():
            arm.connect(read_only=read_only)

    def get_obs(self) -> dict:
        """一次性快照所有臂状态与相机最新帧（单进程内读取，天然同刻）。"""
        return {
            "arms": {name: arm.get_state() for name, arm in self.arms.items()},
            "cams": {name: cam.read() for name, cam in self.cameras.items()},
        }

    def stale_cameras(self, timeout: float) -> list:
        """返回超过 timeout 秒没有新帧的相机名列表。"""
        return [name for name, cam in self.cameras.items() if not cam.alive(timeout)]

    def move_joint(self, actions: dict):
        """actions: {arm_name: {"joint": (dof,), "gripper": float 或 None}}"""
        for name, action in actions.items():
            self.arms[name].move_joint(action["joint"], action.get("gripper"))

    def disconnect(self, trace=None):
        active_exception = sys.exc_info()[1]
        errors = []
        # Writable Piper arms reset here.  Do them before camera teardown, and
        # never let one cleanup failure skip the remaining devices.
        for arm in self.arms.values():
            try:
                if hasattr(arm, "set_cleanup_trace"):
                    arm.set_cleanup_trace(trace)
                arm.disconnect()
            except BaseException as exc:
                errors.append(exc)
        for cam in self.cameras.values():
            try:
                cam.disconnect()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            if active_exception is not None:
                add_note = getattr(active_exception, "add_note", None)
                if add_note is not None:
                    for error in errors:
                        add_note(
                            "robot cleanup 同时失败："
                            f"{type(error).__name__}: {error}"
                        )
                return
            raise errors[0]
