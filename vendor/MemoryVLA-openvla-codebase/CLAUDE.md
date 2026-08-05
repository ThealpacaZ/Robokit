# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在此代码库中工作时提供指导。

**重要：后续所有问题请用中文回答。**

## 项目概述

MemoryVLA 是一个面向机器人操作的认知-记忆-动作框架（ICLR 2026）。它在 OpenVLA 基础上扩展了受海马体启发的记忆系统，用于长时域、时序感知的动作生成。基准测试包括 Bridge、LIBERO、Fractal-VM/VA 和 ManiSkill2。

## 环境配置

```bash
conda create --name memvla python=3.10
conda activate memvla
pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 --index-url https://download.pytorch.org/whl/cu121
conda install -c nvidia cuda-nvcc=12.1 cuda-toolkit=12.1 -y
pip install flash-attn==2.5.5  # 训练必需
pip install -e .
```

## 常用命令

**训练：**
```bash
bash script/train/bridge/train_bridge.sh
bash script/train/libero/train_libero_spatial.sh   # 或 object/goal/100
bash script/train/fractal/train_fractal.sh
bash script/train/real_world/train_real.sh
```

**评估：**
```bash
bash script/eval/bridge/eval_bridge.sh
bash script/eval/libero/eval_libero.sh
bash script/eval/fractal/eval_fractal.sh
bash script/eval/real_world/deploy.sh
```

**代码质量**（配置见 `pyproject.toml`）：
```bash
black --line-length 121 .
ruff check .   # 规则：A, B, E, F, I, RUF, W
```

## 架构

### 数据流
```
图像 + 语言 → 视觉骨干网络 → 记忆编码器 →
  记忆交叉注意力 → 大语言模型 → 动作扩散 →
  动作 Token → 机器人动作
```

### 核心模块

**`vla/memory_vla.py`** — 核心模型，继承 `PrismaticVLM`，新增：
- `TimestepEmbedder`：扩散时间步的正弦位置编码
- `CrossTransformerBlock`：记忆上下文与当前观测之间的交叉注意力
- 记忆编码器/解码器，用于跨时间步的时序上下文

**`prismatic/`** — 视觉-语言骨干网络（来自 OpenVLA）：
- `prismatic/models/vlms/` — `PrismaticVLM` 基类
- `prismatic/models/backbones/vision/` — DINO、SigLIP 等视觉编码器
- `prismatic/models/backbones/llm/` — 基于 Llama 的语言模型

**`action_model/`** — 基于扩散的动作生成：
- `models.py` — DiT（扩散 Transformer）架构
- `gaussian_diffusion.py` — 前向/反向扩散过程
- `diffusion_utils.py` — DDIM 采样（快速推理）

**`vla/datasets/rlds/`** — 数据集流水线：
- 加载 RLDS/Open X-Embodiment 格式数据
- `obs_transforms.py` / `traj_transforms.py` — 图像预处理与动作分块
- `oxe/` — Open X-Embodiment 各数据集配置

**`training/strategies/fsdp.py`** — FSDP 分布式训练（针对 8× A100 设计）。

**`deploy.py`** — 用于真实世界部署的 Flask 服务器，支持自适应集成与动作分块，以 BFloat16 运行。

### 配置管理

配置以数据类形式注册在 `conf/vla.py` 中，通过 `draccus` 管理。训练入口为 `train.py`，模型加载（HuggingFace Hub 或本地）在 `vla/load.py`。
