# RLDS 转换

把清洗后的 robokit HDF5 转成 RLDS (TFDS) 格式，供 X-embodiment / openpi 等训练框架使用。

## 环境

需要 tensorflow + tensorflow_datasets + tensorflow_hub（机器人端 robokit 环境**不含**这些）。
本机已有的 `pi0_demo` 环境满足要求，或新建：

```bash
conda create -n robokit_rlds python=3.10 -y
conda activate robokit_rlds
pip install tensorflow tensorflow_datasets tensorflow_hub h5py scipy numpy
```

## 使用

```bash
conda activate pi0_demo
export ROBOKIT_DATA_DIR=/home/ysh/robokit/datasets/你的任务名   # 指向含 N.hdf5 的目录
cd robokit_dataset
tfds build --overwrite
```

输出到 `~/tensorflow_datasets/robokit_dataset/`。

## 选项（环境变量）

| 变量 | 默认 | 说明 |
|------|------|------|
| `ROBOKIT_DATA_DIR` | `./data` | 任务目录（含 N.hdf5） |
| `ROBOKIT_ACTION_MODE` | `eef_delta` | `eef_delta`: state/action = 每臂 [EEF位姿6, 夹爪1]，action 为局部 delta pose + 下一帧夹爪；`joint`: state = [关节, 夹爪]，action = 下一帧 state |
| `NO_USE_EMBED` | 未设 | 设为 1 时跳过 USE 下载，language_embedding 用零向量（离线测试用） |

## 格式说明

- 1 个 HDF5 = 1 个 RLDS episode（T 帧 → T-1 个 step，无需旧版的 chunk 合并 / episodes.json）
- 语言指令取自 HDF5 attrs 里的 `task_name`
- 相机命名映射：`cam_high → image`，`cam_wrist → wrist_image`，其余保留原名
- 多臂按臂名排序拼接 state/action
