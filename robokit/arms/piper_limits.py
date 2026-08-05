"""Piper 关节限位——全仓唯一权威表。

单独成一个模块而不是放在 piper_ik.py 里，是因为 piper_ik 依赖 scipy
（`scipy.spatial.transform`，实测 import 要 0.43s）。只需要这张常量表的命令行
脚本（复位、越界恢复）不该为此付启动开销：reset_piper_to_demo_start.py 曾因为
从 piper_ik 取这张表，每次复位凭空多花 0.33s。

本模块只依赖 numpy。piper_ik 从这里 re-export，历史的
`from robokit.arms.piper_ik import DEFAULT_JOINT_LIMITS_DEG` 仍然可用。
"""
import numpy as np

# 取两个来源的并集（逐轴取宽者）：
#   来源 A 松灵官方 piper_description.urdf（弧度换算成度）
#   来源 B 本机主控 Flash 实读（SearchAllMotorMaxAngleSpd，2026-07-28）
#
#   轴   官方URDF          Flash            取宽
#   j1   ±150.00          ±150.0           ±150.0    一致
#   j2   [0, 179.91]      [0, 180.0]       [0, 180.0]
#   j3   [-170, 0]        [-170, 0]        [-170, 0] 一致
#   j4   ±99.98           ±100.0           ±100.0
#   j5   ±69.90           ±70.0            ±70.0
#   j6   ±120.00          ±180.0           ±180.0    差 60°
#
# Flash 在每一轴上都等于或宽于官方 URDF，所以并集就是 Flash 那一套。换机械臂或
# 改固件参数后必须重查 Flash，官方 URDF 那一列不随机器变。
#
# 这张表必须是全仓唯一的一份：j6 曾经在 piper_ik / piper.py / 两个脚本 / YAML
# 里分别写成 ±120、±170、±180，同一个 j6 目标会在一处放行、另一处被夹住。
DEFAULT_JOINT_LIMITS_DEG = np.array(
    [
        [-150.0, 150.0],
        [0.0, 180.0],
        [-170.0, 0.0],
        [-100.0, 100.0],
        [-70.0, 70.0],
        [-180.0, 180.0],
    ],
    dtype=np.float64,
)
