"""EEF 位姿数学。位姿统一为 [x, y, z, roll, pitch, yaw]，米 / 真弧度，xyz 固定轴外旋欧拉序。

欧拉约定取 **xyz 外旋**（scipy 小写记法，≡ 内旋 ZYX），因为这就是 Piper 固件
`GetArmEndPoseMsgs` 的 RX/RY/RZ 语义：按 xyz 解读时，记录位姿与官方 URDF 正运动学在
实测 5 段 1129 帧上最大只差 1.03°；按内旋 XYZ 解读则处处差到 ~180°，并会在手腕接近
pitch 90° 时凭空造出上百度的假旋转。

单位一律是真弧度。旧 Test_piper 管线的「度/1000」伪单位适配已整体移除 —— 那套数值
空间不是合法的旋转流形（R(pseudo)≈I，位姿合成退化成欧拉角数值相加）。

RLDS 转换与部署端复用同一实现，保证训练/推理动作空间对称。
"""
import numpy as np
from scipy.spatial.transform import Rotation

# 全链路唯一的欧拉序定义。scipy 记法：小写 = 固定轴外旋，大写 = 随动轴内旋。
# 改这个常量等于改动作空间的定义，训练数据必须同步重转。
EULER_SEQ = "xyz"


def local_delta_pose(base_pose, target_pose):
    """计算 base_pose 局部坐标系下到 target_pose 的位姿增量 (6,)。"""
    base = np.asarray(base_pose, dtype=np.float64)
    target = np.asarray(target_pose, dtype=np.float64)

    base_rot = Rotation.from_euler(EULER_SEQ, base[3:])
    target_rot = Rotation.from_euler(EULER_SEQ, target[3:])
    delta_rpy = (base_rot.inv() * target_rot).as_euler(EULER_SEQ)
    delta_xyz = base_rot.inv().apply(target[:3] - base[:3])
    return np.concatenate([delta_xyz, delta_rpy])


def wrap_euler(pose):
    """把位姿的欧拉角分量逐个折回 (-π, π]，姿态在物理上完全不变。

    xyz 外旋下 R = Rz(y)·Ry(p)·Rx(r)，每个初等旋转对自己那一个角都是 2π 周期的，
    因此逐分量取模是**严格保持姿态**的，也保持增量沿欧拉角累加的语义（角度本来就是
    模 2π 的）—— 与 Rotation.as_euler() 重新分解不同，后者可能返回另一组等价但三个
    分量全变的三元组，会让相邻命令看起来突跳。

    为什么需要：增量沿欧拉角连续累加时，命令角会越过 ±180° 继续增大（实测 policy 的
    yaw 跑到 -202°）。这样的值送进 scipy/IK/数字孪生会被重新归一化，显示上就是 360°
    跳变；送进机械臂固件则超出 SDK 文档的角度范围，固件若按范围截断会静默产生误差。

    默认不启用（ChunkExecutor(wrap_target=False)）：真弧度下折回本身无害，但两种表示
    对固件的实际影响还没在真机上验证；需比较物理姿态并检查固件是否报错或静默截断。
    """
    pose = np.asarray(pose, dtype=np.float64).copy()
    pose[3:] = (pose[3:] + np.pi) % (2 * np.pi) - np.pi
    return pose


def apply_local_delta_pose(base_pose, delta_pose):
    """把局部增量 delta_pose 应用到 base_pose，返回全局目标位姿 (6,)。"""
    base = np.asarray(base_pose, dtype=np.float64)
    delta = np.asarray(delta_pose, dtype=np.float64)

    base_rot = Rotation.from_euler(EULER_SEQ, base[3:])
    target_rpy = (base_rot * Rotation.from_euler(EULER_SEQ, delta[3:])).as_euler(EULER_SEQ)
    target_xyz = base[:3] + base_rot.apply(delta[:3])
    return np.concatenate([target_xyz, target_rpy])
