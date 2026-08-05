# lememoryvla

把 robokit 已发布到 HF 的数据集变成 MemoryVLA 训练能直接吃的格式，**训练代码和超参数一个字都不改**——全部照上游 MemoryVLA 的 README/脚本原样使用。

> 位置：`vendor/MemoryVLA-openvla-codebase/lememoryvla/`——本文件夹就在被打补丁的
> 那份代码库**里面**，不是 robokit 仓库根目录下的独立目录。下文所有相对路径和命令都
> 以此为准；本文档里的 `..` 指的是 `vendor/MemoryVLA-openvla-codebase/` 本身。

## lememoryvla 和 MemoryVLA 的区别

**MemoryVLA** 是上游 ICLR'26 的模型+代码库，本文件夹的上一级目录（[`..`](..)）就是
它原封不动 vendor 进来的那份代码库。它自己的 README 训练的是
Bridge/LIBERO/Fractal——公开 benchmark 的 RLDS 数据，跟我们无关。

**lememoryvla**（这个文件夹）不是另一个模型，也不是训练代码的分支。它是 robokit 让
同一套代码库吃**我们自己 Piper 真机数据**的配方：单第三人称相机、无腕部相机、连续
（不二值化）夹爪、局部 EEF-delta 动作。这个配方就是给代码库的 OXE 数据集注册表
（`vla/datasets/rlds/oxe/{configs,transforms,mixtures}.py`）打的一个很小的、只增不改
的补丁——`robokit_dataset` 的 config + transform，加上每个数据批次一条注册，让 `train.py`
能找到它们。这些补丁 `prepare_data.py` 会自己装（已经装过就跳过），所以对着一份**干净的
上游 checkout** 跑也行，不必先有 robokit vendor 的那份。"lememory" 就是"用我们的数据训练
memory[VLA]"，模型结构和训练循环本身还是原版 MemoryVLA。

## 数据来源

两个任务都发布在 HF **dataset** 仓库 `shaohuan1/Memoryvla`（公开仓库，读取不需要
token），每个采集批次一个子目录：

```
<task>_b<N>/
  hdf5/*.hdf5                          原始 HDF5（这里用不到）
  robokit_dataset/<version>/*.tfrecord RLDS —— 本脚本要拉的就是这个
  source_meta/                         采集配置 + 清洗报告
```

| 任务 | HF 前缀 | 现状（2026-08-03 实查） |
|---|---|---|
| Stack one cup on top of another cup（把一个杯子摞到另一个杯子上） | `Stack_one_cup_on_top_of_another_cup_b*` | 已发布——`_b0` 一个批次，201 段 |
| Cover the building block with a cup, then lift up the cup covering the block.（用杯子盖住积木，再把盖住积木的杯子提起来） | `Cover_the_building_block_with_a_cup,_then_lift_up_the_cup_covering_the_block._b*` | 已发布——`_b0` 157 段 + `_b1` 144 段，共 301 段 |

两个任务合计 **502 段**（`lememory_all` 混合的规模就是这个数）。

两个任务都是 640×480、30 Hz、单 `cam_high`、`right_arm`，RLDS 的
`language_instruction` 就是 `source_meta/config.json` 里的 `task_name` 原文。上面的段数
只是写文档时的快照——真正的现状以 `prepare_data.py --list` 当场查到的为准，新批次发出来
不用改代码，重跑一遍就能吃到。

⚠️ **同仓库里还有一个 `stack_cups/`（74 段、320×240），以及另一个仓库
`shaohuan1/lememory`（"stack cups" 5 个批次共 401 段、320×240）——都是早期的旧数据，
和上面两个任务不是一回事，别混。** `stack_cups/` 没有 `_b<N>` 后缀，所以脚本不会把它
当批次（会打一行 `[warn] ... ignored`）；`shaohuan1/lememory` 则要显式
`--source-repo` 才会碰。

发布是一个文件一个文件上传的，所以「HF 上出现了 `<task>_b<N>/` 目录」不等于「这批 RLDS
传完了」。脚本对每个批次先查一次 Hub：`dataset_info.json` + `features.json` 齐、且
分片数和 `dataset_info.json` 里写的一致，才算这批可用；正在传的批次报一行 warn 就跳过
（不下载、不注册），等传完重跑即可。

## 前置条件

只需要 `huggingface_hub`——不需要 TensorFlow、不需要 torch、不需要 MemoryVLA 的环境。
RLDS 分片是原样搬运（软链接，不重新编码），这一步完全不需要读取它们的内容。

```bash
pip install huggingface_hub   # 或者：conda run -n robokit python lememoryvla/prepare_data.py ...
```

真正训练时仍然需要[代码库自己 README 里的环境](../README.md#install)。

## 用法

以下命令都在代码库根目录（即本文件夹的上一级，`vendor/MemoryVLA-openvla-codebase/`）
下执行：

```bash
cd vendor/MemoryVLA-openvla-codebase
```

只查看当前有什么（不下 RLDS 分片，只取每批几 KB 的 `dataset_info.json` 来核对完整性和段数）：

```bash
python lememoryvla/prepare_data.py --list
```

输出长这样——每批的段数、分片数、版本都是当场从 HF 查的：

```
[warn] 'stack_cups': not a `<task>_b<N>` batch dir -- ignored

Found in shaohuan1/Memoryvla:
  'Cover_the_building_block_with_a_cup,_then_lift_up_the_cup_covering_the_block.' -> 2 batch(es):
      ..._b0: 157 episodes, 128 shards, v1.1.0
      ..._b1: 144 episodes, 128 shards, v1.1.0
  'Stack_one_cup_on_top_of_another_cup' -> 1 batch(es):
      Stack_one_cup_on_top_of_another_cup_b0: 201 episodes, 128 shards, v1.1.0
```

把已发布的数据全部拉下来，并注册进当前这份代码库：

```bash
python lememoryvla/prepare_data.py
```

这一条命令做的事：
1. 对 `shaohuan1/Memoryvla` 里查到的每个**已传完**的 `<task>_b<N>` 批次，只下载它的
   RLDS 文件夹（不下 HDF5）。
2. 把每个批次软链接到 `~/tensorflow_datasets/lememoryvla/<task>_b<N>/<version>/`——
   一个 MemoryVLA 的 TFDS 加载器能直接打开的扁平目录。
3. 把每个 `<task>_b<N>` 注册成代码库自己 OXE 注册表里的一个数据集名，复用
   `robokit_dataset` config/transform（单相机、连续夹爪，见下；不在就先装上），并写入/更新
   每个任务各自的 `lememory_<task>` 混合，以及一个合并两个任务的 `lememory_all`。
4. 给 `vla/datasets/rlds/dataset.py` 装上 RLDS 加载器补丁（见下一节；已经有了就跳过）。

第 3、4 步会改代码库里的文件，只增不改逻辑，重复运行幂等。只想要数据、不想动代码库就加
`--skip-register`。

## RLDS 加载器补丁

上游 `vla/datasets/rlds/dataset.py` 用 `tfds.builder(name, data_dir=...)` 解析数据集名，
它**只认注册过 Python builder 类的名字，而且带 `,` 或 `.` 的名字直接拒绝解析**。我们的批次
目录名是拿任务提示词拼的，所以在干净的上游 checkout 上两个任务都读不出来（tfds 4.9.10 实测：
`Stack_one_cup_..._b0` 抛 `DatasetNotFoundError`，cover-block 那个带逗号的抛 `ValueError`）。

`prepare_data.py` 会把那一行调用换成「先照原样试，失败了退回
`tfds.builder_from_directory(<data_root>/<name>/<version>/)`」——后者两个限制都没有，实测
两个任务分别读出 201 / 157 段。**这一步不是可选的**：不打这个补丁，注册全都成功、训练一
开始读数据就崩。

如果你的 checkout 里那一行长得不一样（改过、或上游版本不同），脚本不会硬改，只打一条
`[WARN]` 叫你手动加，照上面这个逻辑改 `make_dataset_from_rlds` 里的 `builder = ...` 即可。

常用参数：

```bash
--only Stack_one_cup         # 只处理名字里含这个子串的任务（可多次传）
--source-repo <repo>         # 换数据源仓库（默认 shaohuan1/Memoryvla）
--out <dir>                  # 数据落到哪个 data_root_dir（默认 ~/tensorflow_datasets/lememoryvla）
--codebase <path>            # 要注册进哪份 MemoryVLA-openvla-codebase checkout（默认就是本文件夹所在的这份）
--skip-register              # 只落数据，不动代码库
```

可以随时重复运行：已经落地的批次会跳过（不动已有的软链接），新发布的批次会补上，
混合定义每次都按当前实际落地的数据重新生成——所以后面再采、再发新批次，直接重跑
这条命令就能吃到新数据。

注意混合名里的 `lememory_` 前缀是这套配方（`lememoryvla` 文件夹）的名字，跟数据在哪个
HF 仓库无关——数据源换成 `shaohuan1/Memoryvla` 之后混合名照旧，不用改训练命令行里已有的
写法之外的东西。

## 只把这个文件夹交给别人

对方**不需要 robokit 仓库**，需要的是：

1. 一份 MemoryVLA 的 openvla-codebase checkout：
   `git clone -b openvla-codebase https://github.com/shihao1895/MemoryVLA`（干净的上游就行）。
2. 把 `lememoryvla/` 整个放进那份 checkout 的**根目录**下，然后在根目录跑
   `python lememoryvla/prepare_data.py`。数据源是公开仓库，读取不需要 HF token。
3. 训练环境按代码库自己 README 的 Install 装（`prepare_data.py` 本身只要
   `huggingface_hub`）。

脚本会自己把 OXE config/transform/mixture 和上一节那个加载器补丁装进他们的 checkout，
不用手工改文件；跑完屏幕上会打出 `--data_root_dir` 和可用的 `--vla.data_mix`。

磁盘：RLDS 合计 **28.2 GiB**（201 段 8.7 + 157 段 10.6 + 144 段 9.0），HDF5 不下。
`snapshot_download` 落在 HF 缓存里，`~/tensorflow_datasets/lememoryvla/` 下面是**软链接**
——所以别单独打包/搬运那个目录，也别在训练期间清 HF 缓存，链接会断。

**这个文件夹里没有的东西**：robokit 给 vendor 那份代码库还打过一处与训练无关的本地改动
——`vla/memory_vla.py` 里的 `MEMVLA_BINARIZE_GRIPPER` 开关（推理时不把第 7 维夹爪硬二值化）。
只训练用不到；如果对方还要接 robokit 的部署端做真机评测，得把那处一起带上。

## 训练

下面这条命令就是原版 `train.py`——超参数（batch size、学习率、步数、LoRA/动作模型配置、
checkpoint 路径、wandb、hf_token）全部自己定，跟
[代码库自己 README 的 Training 一节](../README.md#training)
以及
[`script/train/real_world/train_real.sh`](../script/train/real_world/train_real.sh)
里演示的一样。本脚本只决定下面这两个参数：

```bash
cd vendor/MemoryVLA-openvla-codebase
python train.py \
  --data_root_dir ~/tensorflow_datasets/lememoryvla \
  --vla.data_mix lememory_stack_one_cup \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --pretrained_checkpoint /path/to/CogACT-Large.pt \
  ... # 其余全部自己定，完整示例见 train_real.sh
```

跑完 `prepare_data.py` 之后，可用的 `--vla.data_mix`（脚本结尾也会打印）：

- `lememory_stack_one_cup` —— 已发布的全部 stack-one-cup 批次（当前 `_b0`，201 段）
- `lememory_cover_block` —— 已发布的全部 cover-block 批次（当前 `_b0`+`_b1`，301 段）
- `lememory_all` —— 两个任务合并

混合里每个批次的权重默认都是 `1.0`（均匀）。如果想按段数加权、或者压低某个噪声较大的
批次，去改代码库里的 [`vla/datasets/rlds/oxe/mixtures.py`](../vla/datasets/rlds/oxe/mixtures.py)
里 `lememoryvla:auto-generated` 两条标记之间的那一段——但要知道，`prepare_data.py`
再跑一次会把这段重新生成回均匀权重。

## 为什么是按批次注册，不合并成一个数据集

TFDS 不支持在已经构建好的数据集上追加内容，robokit 自己的发布流水线正是因为这个原因把
每个采集批次当成独立的 RLDS 数据集（见 robokit 主 [README](../../../README.md) 的
"分批发布到 HuggingFace"一节）。与其把所有 HDF5 重新整体转换成一个数据集——慢，而且每来一个新批次
都要重来一遍——不如把每个已发布批次注册成自己的 OXE 数据集名，交给
`OXE_NAMED_MIXTURES` 去做混合，这本来就是这个机制存在的意义。如果确实想要一个物理上
统一的数据集而不是混合，等某个任务采集彻底完成后，用 robokit 的
[`rlds/build.py`](../../../rlds/build.py) 对整个任务目录重新转换一次
（见 robokit 主 README 的数据流水线一节）。
