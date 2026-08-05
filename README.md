# robokit

> **SSH / TUN 约定：本机长期保持 Mihomo TUN 开启。** 当前代理节点连接 SeetaCloud
> 会在 SSH banner 前关闭或超时；普通进程走物理网卡直连又会被本地网络过滤。已验证可用路径是：
> 临时把 Mihomo `GLOBAL` 选择器切到其内部 `DIRECT`，再经本地 SOCKS 端口建立 SSH；
> SSH 退出后恢复原选择器。可在当前 shell 定义以下包装函数：
>
> ```bash
> seeta_via_tun() (
>   set -e
>   local mihomo_socket=/tmp/verge/verge-mihomo.sock
>   local previous_global restore_payload
>   previous_global=$(curl -fsS --unix-socket "$mihomo_socket" \
>     http://localhost/proxies/GLOBAL | \
>     python3 -c 'import json,sys; print(json.load(sys.stdin)["now"])')
>   restore_global() {
>     restore_payload=$(MIHOMO_NODE="$previous_global" python3 -c \
>       'import json,os; print(json.dumps({"name": os.environ["MIHOMO_NODE"]}, ensure_ascii=False))')
>     curl -fsS --unix-socket "$mihomo_socket" -X PUT \
>       -H 'Content-Type: application/json' --data "$restore_payload" \
>       http://localhost/proxies/GLOBAL >/dev/null
>   }
>   trap restore_global EXIT INT TERM
>   curl -fsS --unix-socket "$mihomo_socket" -X PUT \
>     -H 'Content-Type: application/json' --data '{"name":"DIRECT"}' \
>     http://localhost/proxies/GLOBAL >/dev/null
>   "$@"
> )
>
> seeta_via_tun ssh \
>   -o 'ProxyCommand=nc -X 5 -x 127.0.0.1:7897 %h %p' \
>   -p "$SSH_PORT" root@connect.weste.seetacloud.com
> ```
>
> 自动化时把最后的 `ssh ...` 改为 `sshpass -p "$SSH_PASSWORD" ssh ...`。端口和当日密码以
> [PROJECT_MEMORY.md](PROJECT_MEMORY.md) 顶部为准。`seeta_via_tun` 在子 shell 中运行，正常退出、
> 报错或中断都会恢复原节点；无需关闭 TUN，也不要添加永久系统路由。若仍在 banner 前失败，
> 再检查实例状态和 SSH 映射端口，此时密码尚未参与认证。

Piper 机械臂的 VLA 真机工具包。两件事：

1. **采集数据** — 遥操作录制 → 清洗 → RLDS 转换，产出可训练的数据集
2. **执行推理** — 加载 VLA 权重，在真机上闭环执行 policy 输出的动作

配置驱动（YAML），相机与机械臂的数量、自由度任意。

---

## 接手本项目的规则

**每次接手本项目，都必须把这一轮得到的处理经验补进 [TASK_STATE.md](TASK_STATE.md)，
并覆盖其中已经过时的旧经验。**

不是追加流水账，是维护一份始终为真的现状：

- **新结论覆盖旧结论。** 同一个问题有了更准的答案，直接改写原文，不要并列保留两种说法。
  旧说法如果曾经误导过人，用一句话写清它错在哪，然后删掉它本身。
- **只留还成立的。** 已经修好的 bug、已经作废的路径、已经完成的验证步骤，从「待解决」
  移走或删掉，不要留在那里让人重做。
- **写清判据和数字。** 「跑通了」没有信息量；「573 步零拒绝、残差 0.002mm」才能让下一个人
  判断你的结论还成不成立。
- **排除性结论同样要写。** 查过并否掉的假因（「不是模型的问题，因为…」）价值不亚于正面结论，
  它直接省掉下一轮的重复排查。

文档只有三份，不要新建 `docs/`：README 写项目是什么和怎么用，TASK_STATE 写当前状态和经验，
PROJECT_MEMORY 写长期不变的事实（服务器、权重）与待用户定夺的开放项。

---

## 环境

```bash
/home/ysh/miniconda3/envs/robokit/bin/python        # conda env robokit, py3.12, piper-sdk 0.6.1
python -m unittest discover -s scripts -p "test_*.py"   # pytest 未安装，用 unittest
```

- `type: ros2` 相机需要 `source /opt/ros/jazzy/setup.bash`；默认的 RealSense 直连不需要。
- RLDS 转换用 `pi0_demo` 环境（tensorflow/tfds）。
- 推理服务端用模型自己的环境，只需能 import `robokit/comm.py`（纯标准库）+ numpy。

无硬件自测：

```bash
python scripts/collect.py --config configs/mock.yaml --episodes 2 --auto-seconds 3
python scripts/clean.py --data "datasets/mock task"
```

---

## 功能一：采集数据

```bash
# can0 已经 UP 时，Linux 不允许直接修改 bitrate，会报
# "RTNETLINK answers: Device or resource busy"。停止真机控制进程后再重配：
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
ip -details link show can0      # 应显示 state UP、bitrate 1000000

# 1) 录制：回车开始 → 回车停止落盘。1 episode = 1 个 HDF5
python scripts/collect.py --config configs/piper_single.yaml --episodes 25

# 采集分辨率 2026-08-03 由 320x240 提到 640x480（仍是 4:3，模型几何不变，0.92MB/帧）。
# preview_preprocess: lememory 把采集窗口显示成 leMemory train_aligned 的模型视图：
# 完整 480x640 拉伸成方图，再做 0.9 中心裁剪；HDF5 仍保存原始 480x640。
# 模型吃的是 224x224，预览按 preview_image_size 画大（480x480 = 不插值的上限，
# 竖向与传感器 1:1，横向 640→480 正是模型那步压扁）；几何一致，只是像素更多。
# 预览窗口从屏幕左上角平铺，每个等比放大到屏幕面积的 1/4（preview_screen_fraction 调）；
# 预览预处理 6.9ms/帧，会跟采集节拍抢时间，所以默认 preview_every: 2 隔帧画；
# 回车在终端里按或在预览窗口里按都算，不用为了敲回车先用鼠标点回终端。

# 2) 清洗：生成 clean_report.json；--quarantine 把坏数据移入 _quarantine/
python scripts/clean.py --data "datasets/任务名" [--quarantine]

# 3) 可视化：视频 + 曲线
python scripts/visualize.py --data "datasets/任务名" [--episode 3 7] [--no-video]

# 4) 转 RLDS（pi0_demo 环境）
python rlds/build.py --data "datasets/任务名" --action-mode eef_delta --overwrite
```

采集时机械臂是**只读**的：`connect(read_only=True)` 跳过 EnableArm/GripperCtrl/`piper_init`，
一帧 CAN 控制帧都不发。主臂从臂挂同一条总线，发一帧 `0x159 GripperCtrl` 就会把从臂夹爪
从联动切成位置保持，之后主臂张开也带不动。

清洗结论是**硬闸**：没跑过清洗、目录里还留着 bad episode、报告过期、跨 episode 形状不一致，
`rlds/build.py` 都会直接拒绝转换（对照实验用 `--allow-unclean`）。

### 分批发布到 HuggingFace（边采边发）

采完一批就发一批，发布期间可以接着采下一批。**本地全程只有一份数据**：每段 HDF5 要么在
任务目录、要么在批次目录（冻结是 `mv`，不是复制也不是硬链接），RLDS 是派生物，核对完远端就删。

```bash
hf auth login                                                  # 首次：必须是 role=write 的 hf_… token
./scripts/publish_batch.sh --repo lememory "stack cups"        # 发任务目录里现存的所有段
./scripts/publish_batch.sh --repo Memoryvla "stack cups" 0 65  # 或只发指定编号范围
./scripts/publish_batch.sh --audit --repo lememory             # 核对整个仓库（所有任务）
./scripts/publish_batch.sh --audit --repo lememory "stack cups"   # 或只核对一个任务
KEEP_RLDS=1 ./scripts/publish_batch.sh --repo lememory "stack cups"   # 本机还要用 RLDS 就别让它删
```

一条命令跑完 **冻结（mv）→ 清洗 → 转 RLDS → HDF5 与 RLDS 一起上传 → 逐文件比对远端大小 →
写 manifest → 删本地 RLDS → 刷新仓库首页索引**。

**先选仓库，任务在仓库下面**。`--repo` 给裸名字就自动补 `shaohuan1/`（`HF_OWNER=` 可改），
不给就用 `HF_REPO`、再不给用默认的 `Memoryvla`。现有两个仓库：`shaohuan1/Memoryvla`、
`shaohuan1/lememory`。一个仓库装多个任务，一个任务一个目录，任务下面一批一个目录：

```
<任务slug>/<批次>/
  hdf5/N.hdf5                       原始 HDF5（关节角、各路时间戳只在这里）
  robokit_dataset/1.1.0/*.tfrecord  RLDS
  source_meta/config.json, clean_report.json
```

本地按同一套结构镜像（`upload-large-folder` 没有 `--path-in-repo`，仓库内路径 = 相对上传根的
路径，所以「仓库名」那一层就是上传根）：

```
datasets/_batches/<仓库>/<任务slug>/<批次>/hdf5/N.hdf5   ← HDF5 上传根 = datasets/_batches/<仓库>
datasets/_batches/<仓库>/<任务slug>/<批次>.json          ← manifest 台账，和批次目录并排
~/tensorflow_datasets/<仓库>/<任务slug>/<批次>/robokit_dataset/...   ← RLDS 上传根
```

**磁盘上任何时刻只有一份**：

| 阶段 | 数据在哪 | 额外占用 |
|---|---|---|
| 采集中 | `datasets/<任务名>/N.hdf5` | — |
| 冻结后 | `datasets/_batches/<仓库>/<slug>/<批次>/hdf5/N.hdf5`（同分区 `mv`，瞬间完成） | 0，任务目录里那份没有了 |
| 转换中 | 多一份 RLDS（≈HDF5 的 0.3~0.6×） | 校验通过即删，`KEEP_RLDS=1` 可保留 |
| 校验通过后 | 只剩批次目录里的 HDF5 | 1 份，随时可 `rm -rf` |

台账是每批一个 `<批次>.json`（manifest，几十 KB）：这一批的编号范围 + 每个已上传文件的大小。
**批次数据目录随时可以删**——`--audit` 只读 manifest 和远端，不看本地数据，删完照样能逐文件
核对。一个任务采完发完之后：

```bash
./scripts/publish_batch.sh --audit --repo lememory   # 通过了再删
rm -rf datasets/_batches/lememory/*/*/               # 只删批次数据目录，manifest 是 .json 不受影响
```

连 `datasets/` 整个删掉也只丢两样东西：「下一批从哪接着发」和核对能力；已发布的数据、
训练、部署都不受影响。

仓库首页 `README.md` 每次发布后自动重刷，内容是**按远端实际文件列表现算**的每任务批次索引
（含编号范围、段数、RLDS 有没有、缺号标记）。所以同一个仓库里别的任务、别的机器发的、
以及老布局（`<slug>_<批次>/`，`_b` 拼在一起那种）都会照样出现在首页，不会被这次发布抹掉。

⚠️ 已发布的段**不在** `datasets/<任务名>/` 里了。visualize、整任务重转这类要看整个任务的操作
得指到批次目录；临时凑一个整任务视图就硬链接合并（同分区，不占额外磁盘）：

```bash
mkdir -p /tmp/whole_task && ln -f datasets/_batches/lememory/<slug>/b*/hdf5/*.hdf5 /tmp/whole_task/
```

编号不会重来：脚本在搬走文件**之前**把水位写进 `datasets/<任务名>/.next_index`，
`next_episode_index()` 会读它。没有这个水位，任务目录被搬空后下一段会从 `0.hdf5` 重新编号，
跟远端已发布的段同名不同内容。

**断点续跑**——断在哪一步重跑同一条命令即可：

| 步骤 | 续跑方式 |
|---|---|
| 冻结 | 批次目录已存在就复用，之后一律以它的**实际内容**为准，不用调用方给的意图范围；上次搬到一半就接着搬 |
| 清洗 | `clean_report.json` 在就跳过 |
| 转 RLDS | 只认「`dataset_info.json` 存在 **且** episode 数 == 这批段数」为完成；半截产物整体重转。TFDS 不支持转换中途续跑，所以按批切分——一批最多重来约 10 分钟 |
| 上传 | `upload-large-folder` 原生续传，状态在 `<上传根>/.cache/huggingface/` |
| 校验 | 通过了才写 manifest 的 `verified`；已 verified 的批次直接跳过，不重发 |

不带参数重跑会**先把没发完的那批发完**（批次目录在、manifest 没 verified），再动新采的段。
少了这一步，任务目录里剩下的段会被当成新的一批，断掉的那批就永远留在本地不再被碰。

**为什么必须先把这一批搬出任务目录**：builder 在构造时（`_probe`）和生成时（`_generate_examples`）
各 glob 一次数据目录，两次之间新落盘的 episode 会**绕过清洗硬闸**被静默转进训练集。搬走之后
这一批的内容不再变化，边发边采才是安全的；直接拿 `datasets/任务名/` 边采边转就没有这个保证。

实测开销（320×240 / 30 Hz / 66 段 / 3.1 GB）：

| 环节 | 实测 |
|---|---|
| 转 RLDS | 9.0 s/段，66 段共 9 分 47 秒，**单核**（99% CPU），峰值 RSS **1.97 GB** |
| 体积 | HDF5 3.07 GiB → RLDS 1.75 GiB，**0.57×**（PNG 编码）；校验通过后本地删掉 |
| 上传 RLDS | 1.89 GB / 约 100 秒 ≈ **19 MB/s** |
| 上传 HDF5 | 3.30 GB / 约 130 秒 ≈ **25 MB/s** |
| 一批合计 | 66 段从冻结到校验完约 **14 分钟**（转换 10 min 是大头，不是上传） |
| 采集侧内存 | recorder 队列 600 帧封顶 ≈ 138 MB，满了是反压不是涨内存 |

⚠️ **640×480 的转换耗时不是按像素等比放大的**：像素多 4 倍，实测 **75 s/段**（不是 36），
PNG 编码是大头。73 段要转约 **1.5 小时**，比上传（23 GB / 约 20 分钟）慢得多。脚本按第一段
HDF5 的实际分辨率查实测表给估计，没测过的分辨率才按像素比外推。

转换内存不会失控：tfds 的 shuffle buffer 到 1 GB 就落盘分桶（`shuffle.py MAX_MEM_BUFFER_SIZE`）。
脚本默认 `NO_USE_EMBED=1`——MemoryVLA 训练只读 `language_instruction` 原文，从不读
`language_embedding`，跳过 USE 省约 1 GB 下载 + 1.5 GB 常驻内存；换成会消费该字段的训练栈
必须去掉这个变量重转。

⚠️ 各批是**独立的 RLDS 数据集**（`<slug>/b0_65/`、`<slug>/b66_115/`…），TFDS 不支持增量追加。
训练时可多数据集混采；要一个统一数据集就在整个任务采完后，按上面的硬链接合并凑出整任务目录
再整体重转一次（服务器侧则是 `pi0.5/prepare_lememory.py` 从 HF 把各批下下来合并）。

### 整任务一次性上传并删本地

上面脚本内部做的就是这几步，手动排错或要删本地时用。数据都在 `shaohuan1/Memoryvla`
（`--repo-type dataset`），**一个任务一个子目录**。
`--out` 必须用任务专属目录，否则不同任务都写进同一个 `robokit_dataset/` 互相覆盖。

```bash
export TASK="stack cups"; export SLUG=stack_cups; export HF_REPO=shaohuan1/Memoryvla
python rlds/build.py --data "datasets/$TASK" --out ~/tensorflow_datasets/$SLUG --overwrite

ALL_PROXY= all_proxy= HF_HUB_DISABLE_XET=1 hf upload-large-folder "$HF_REPO" \
  ~/tensorflow_datasets --repo-type dataset \
  --include "$SLUG/robokit_dataset/**" --num-workers 8

# 来源凭据：删掉 HDF5 后这是唯一的采集配置与清洗结论
ALL_PROXY= all_proxy= HF_HUB_DISABLE_XET=1 hf upload "$HF_REPO" \
  "datasets/$TASK/config.json" "$SLUG/source_meta/config.json" --repo-type dataset
ALL_PROXY= all_proxy= HF_HUB_DISABLE_XET=1 hf upload "$HF_REPO" \
  "datasets/$TASK/clean_report.json" "$SLUG/source_meta/clean_report.json" --repo-type dataset

# 逐文件比对远端大小，全一致才删本地 RLDS（HDF5 不删，见下面的不可逆警告）
python3 -c "
import os, sys
from huggingface_hub import HfApi
root = os.path.expanduser('~/tensorflow_datasets/$SLUG/robokit_dataset')
prefix = '$SLUG/robokit_dataset'
remote = {e.path[len(prefix)+1:]: e.size for e in
          HfApi().list_repo_tree('$HF_REPO', repo_type='dataset', path_in_repo=prefix, recursive=True)
          if getattr(e, 'size', None) is not None}
bad = []
for dp, _, ns in os.walk(root):
    for n in ns:
        p = os.path.join(dp, n)
        rel = os.path.relpath(p, root)
        if remote.get(rel) != os.path.getsize(p):
            bad.append(rel)
if bad:
    print(f'校验失败，{len(bad)} 个文件不一致，不删：', bad[:10]); sys.exit(1)
print(f'校验通过：本地 RLDS 与远端 {len(remote)} 个文件全一致')
" && rm -rf ~/tensorflow_datasets/$SLUG/robokit_dataset && echo "==> 已删本地 RLDS：~/tensorflow_datasets/$SLUG/robokit_dataset"
```

三个本机特有的坑，上面命令都已处理：

- **`ALL_PROXY` 是 `socks://…`，`hf` CLI 底层的 httpx 不认**，报 `Unknown scheme for proxy URL`。每条 hf 命令前都要清掉。
- **必须 `HF_HUB_DISABLE_XET=1`**。huggingface_hub 1.x 默认走 Xet，实测在 "Finished hashing" 后卡死到 2 KB/s；关掉走普通 LFS 立刻回到 2.2 MB/s。
- 必须是 **role=write 的 `hf_…` token**。只读 token 会在建仓步骤报 `401 /api/repos/create`，哪怕仓库早就存在。S3 bucket 凭据是另一套对象存储，不能用。

用 `upload-large-folder` 而不是 `hf upload`：断了能续传。它没有 `path_in_repo`，**仓库内路径 = 相对上传根的路径**。

上行速率**不稳定，差一个数量级，别按某一次的实测做计划**：2026-07-26 实测 2~2.5 MB/s（当时判断是链路 ≈20 Mbit/s 封顶，加并发只快 15%）；2026-07-28 同样命令、同样 `--num-workers 8` 实测 **19~25 MB/s**（5.2 GB 约 4 分钟）。所以 2.5 MB/s 不是链路天花板。慢的时候先确认 `HF_HUB_DISABLE_XET=1` 生效。

删本地前**必须**逐文件比对远端大小（`HfApi().list_repo_tree`），全一致才算成功。

⚠️ **删 HDF5 不可逆，RLDS 不是 HDF5 的超集**：转换只留 EEF 位姿 + 夹爪 + 图像 + 语言，
**关节角、`timestamps/frame`、相机 capture/receive 时间戳都不进 RLDS**。还要做关节真值核对、
时间抖动分析或换动作空间重转就不能删。

### HDF5 格式（robokit-1.0）

```
observations/images/{cam}        (T,H,W,3) uint8
observations/{arm}/joint         (T,dof)   float32
observations/{arm}/eef_pose      (T,6)     float32
observations/{arm}/gripper       (T,1)     float32   [0,1]
timestamps/frame                 (T,)      float64   epoch 秒
timestamps/cams/{cam}/capture    (T,)                ROS header.stamp
timestamps/cams/{cam}/receive    (T,)                本机收到并解码完成
timestamps/arms/{arm}            (T,)
attrs: task_name, freq, config_json, version
```

动作不落盘，转 RLDS 时由状态算（局部 delta 位姿 + next gripper）。

**单位约定**：全链路 米 / 真弧度 / xyz 外旋欧拉序（`robokit/pose.py`）。旧 `Test_piper`
用「度/1000」伪单位且欧拉序是内旋 XYZ，**旋转维度数值不兼容，不要混用训练**。

π0 关节角对照使用独立 LeRobot 数据集，避免覆盖 PI0.5 EEF 标签与归一化统计：

```bash
python pi0.5/convert_hdf5_to_lerobot.py \
  --data "datasets/stack cups" \
  --repo-id YOUR_HF_USER/stackcups_joint_pi0_lerobot \
  --camera cam_high --arm right_arm \
  --instruction "stack cups" --action-space joint
```

每行 state 是当前 `[j1..j6, gripper]`，action 是下一帧绝对目标
`[j1..j6, gripper]`；六轴均为真弧度，不是关节 delta。π0 不消费这里的 RLDS。

---

## 功能二：执行推理

服务端（GPU）+ 客户端（接机械臂），TCP 通讯。远程 GPU 先开隧道
`ssh -N -L 8080:127.0.0.1:8080 -p <SSH端口> <用户>@<服务器>`（关节模型是 8081）。

一共只有两个入口，两端都用 `--model` 选权重：

| 入口 | 在哪跑 | 干什么 |
|---|---|---|
| `scripts/serve_policy.py` | GPU 服务器 | 加载一个模型，用 sync 或 RTC 方式提供推理 |
| `scripts/run_policy.py` | 机器人端 | 取观测、发推理、执行动作块 |

模型名在 `configs/models.yaml`，两端 `--list` 都能查：

```bash
python scripts/run_policy.py --list
# pi05-eef     PI0.5 全量微调，EEF delta 数据训练，端口 8080
# pi05-joint   PI0.5 全量微调，关节角数据训练，端口 8081
```

### 启动命令

```bash
# ── GPU 服务器 ─────────────────────────────────────────────────────────────
cd /root/robokit
/root/autodl-tmp/envs/pi05/bin/python scripts/serve_policy.py \
  --model pi05-joint --mode rtc          # 或 --mode sync

# ── 机器人端 1) 复位到示教起点 ─────────────────────────────────────────────
python scripts/reset_piper_to_demo_start.py \
  --dataset "datasets/stack cups" --episode 0 --port can0 \
  --execute --skip-controller-reset

# ── 机器人端 2) 第一次先做零运动检查 ───────────────────────────────────────
python scripts/run_policy.py --model pi05-joint --mode rtc --dry-run --max-steps 30

# ── 机器人端 3) 真机执行 ───────────────────────────────────────────────────
# 按回车中断后会自动跑一遍上面那条复位命令，直接接着跑下一轮
python scripts/run_policy.py --model pi05-joint --mode rtc
```

第 1 步只在**开机第一次**需要手工跑：之后按回车中断执行，`run_policy.py` 会在会话完全
退出（归位、controller reset、CAN 释放）之后自动执行同一条复位命令，把臂送回示教起点。
数据集目录和 CAN 口默认从机器人配置推出（`collect.save_path`/`task_name`、第一条 Piper
臂的 `port`），可用 `--reset-dataset` / `--reset-episode` / `--reset-port` 覆盖，
`--no-reset-on-interrupt` 关掉。只有回车中断会触发它：`Ctrl-C` 与安全中止不复位，那两种
情况现场需要先被人看一眼。

### 四个开关

| 开关 | 默认 | 含义 |
|---|---|---|
| `--model` | 必填 | 用哪个模型。同时决定动作空间（`eef_delta`/`joint`）、端口、默认 horizon、用哪份机器人配置 |
| `--mode` | `sync` | `sync`=一次推理执行一块，推理期间臂停着；`rtc`=推理与执行并发（arXiv:2506.07339） |
| `--horizon` | 取登记表 | **每个 chunk** 承诺执行的步数。sync 下是 `ChunkExecutor` 的 horizon，rtc 下是论文的 `s_min` |
| `--max-steps` | `0` = **不限** | **整场**总共执行多少个动作步。跟 `--horizon` 不是一回事，一直跑到回车 / `Ctrl-C` / 安全中止 |

`--horizon` 与 `--max-steps` 的关系：`--horizon 15 --max-steps 40` 会执行
`15 + 15 + 10`，最后一块只执行剩下的 10 步，不会为了凑满 horizon 越过上限。

**⚠ `--horizon` 在两种模式下的安全方向相反：**

* **sync**：越大越省心 —— 每块多执行几步，推理次数变少。上限就是模型的 `H`。
* **rtc**：`--horizon` 就是论文的 `s_min`，**越大越危险**。推理是在执行满 `s_min` 步
  之后才发起的，必须在剩下的 `H - s_min` 步内回来，否则当前 chunk 用尽 → deadline
  abort。s_min 越大，留给推理的窗口越小。

现场实测（30Hz、pi05-joint、SSH 隧道）：一次往返 `297ms ≈ 9 步`（服务端只占 117ms，
其余是网络与序列化）。`--horizon 40` 时窗口只剩 `50-40 = 10 步 = 334ms`，**余量 1.1
步**，第二次请求就超时中止了。按 2 倍余量的经验上限是 `s_min <= H - 2d`，即这条链路上
**`--horizon <= 32`**；登记表默认 15 是安全的。

客户端在 warmup 之后、**任何动作下发之前**就会把余量算出来：窗口小于 `2d` 时打
WARNING，真出现 deadline abort 时会连实测延迟和建议的 `--horizon` 一起打出来。
窗口不够时降 `--control-freq` 比降 `--horizon` 更可取——同样的推理秒数换算成更少的
控制步数，整块动作照原路走完，只是走慢。

### 安全护栏默认解除

`--safety` 默认 `off`，也就是只保留「policy 输出 → EEF/关节目标 → 下发」这一条原始
链路，用来看模型在物理世界的原始行为。解除的是：

* ActionGuard：单步位移/旋转上限、相邻目标跳变、跟踪误差、工作区盒子
* 相机亮度/过曝/陈旧/冻结闸
* Pinocchio IK 的位姿残差与伺服落后拒发（改为 best-effort）
* 夹爪单步限幅、到位等待与到位超时中止
* 退出不归位、不做 controller reset —— 臂带力停在 policy 的终止位姿上，保留现场

**不会**解除的（它们不是护栏）：IK 的 `max_step_deg=5°` 与 `continuity_weight=0.03`
（这是「同一 EEF 位姿的多组关节解里选哪一支」的连续性机制，去掉后 IK 逐帧乱跳，
实测单步关节变化到 103.57°，臂走出的轨迹反而不是 policy 输出的轨迹）；
主控 Flash 里的六轴机械行程；相机取帧失败、非有限动作、维度不符仍是硬错误。

⚠ 护栏解除后机械臂可能撞台面、自撞或在奇异点做大角度分支跳变。**必须有人守着急停。**
要跑受保护的版本加 `--safety on`。具体改了哪些字段见
`robokit/deploy/runtime.py` 的 `relax_safety()`，启动日志也会逐条打印。

### 看模型端拿到的图像

排查「动作诡异」时第一个要排除的是模型看到的画面本身不对（视角/亮度变了、裁剪把物体
切掉了、送错相机）。两端都能看：

```bash
# 机器人端：开窗实时看自己发出去的帧，同时落 PNG
python scripts/run_policy.py --model pi05-joint --show-image \
  --save-obs runs/obs/sent --save-obs-every 10

# 服务端（AutoDL 无显示器）：落 PNG，事后 scp 回来
python scripts/serve_policy.py --model pi05-joint --mode rtc \
  --save-obs /root/autodl-tmp/obs --save-obs-every 10
```

服务端会落两种图：`recv-*.png` 是网络上收到的原始帧（与机器人发出的逐字节相同），
`model-*.png` 是**交给 `predict_action_chunk` 的那个张量反解回来的图**（LeRobot
preprocessor 的归一化已生效）。PI0/PI0.5 的 resize 在模型 forward 内部，所以
`model-*.png` 的分辨率通常仍等于收到的帧 —— 它回答的是「送进模型的像素内容对不对」
（视角、亮度、送错相机、图被裁掉），不是「模型内部最后那层 224×224 长什么样」。

### 其他常用开关

```bash
--dry-run                 # 不下发运动帧，只跑链路与检查。上真机的第一次跑用它
--instruction "stack cups"  # 覆盖登记表里的语言指令
--host / --port           # 覆盖服务器地址与端口
--chunk-base recursive|continuous|feedback   # 递推基准，缺省 sync=recursive、rtc=continuous
--control-freq 20         # 降频；比减小 horizon（丢弃模型预测的后续步）更可取
--eef-backend host_ik|pinocchio_ik           # 显式 A/B 两个 IK 后端
--wait-arrival            # 在 --safety off 下重新打开闭环到位等待
--trace runs/deploy/xxx.jsonl                # 缺省自动写 runs/deploy-<mode>/
```

MemoryVLA 等非 DiT 权重仍走旧的通用服务端：

```bash
python scripts/deploy_server.py --port 8080 --policy memvla_lora \
    --policy-arg checkpoint=/path/to/step-010000.pt \
    --policy-arg base=/path/to/CogACT-Large.pt \
    --policy-arg codebase=/path/to/MemoryVLA-openvla-codebase
# 联调用 --policy dummy（回发当前状态，不需要模型）
```

OpenVLA-OFT 的 latest-only LoRA + continuous action head 也走这个通用同步服务端。当前冻结的
StackCups step 25,000 权重严格使用以下命令：

```bash
# GPU 服务器
cd /root/robokit
/root/autodl-tmp/envs/openvla_oft/bin/python scripts/deploy_server.py \
  --port 8080 --action-space eef_delta --policy openvla_oft \
  --policy-arg checkpoint=/root/autodl-tmp/openvla_oft_deploy/stackcups-step-025000 \
  --policy-arg base=/root/autodl-tmp/openvla-7b-oft-base \
  --policy-arg codebase=/root/autodl-tmp/openvla-oft \
  --policy-arg robot_platform=piper

# 机器人端：先建立 README 顶部所述 TUN SSH 隧道，再直接运行真机同步控制
python scripts/run_policy.py --model openvla-oft --mode sync
```

该 checkpoint 的动作契约是单 `cam_high`、`(30,7)` local EEF delta + continuous gripper；
只允许 `sync`，不要为它使用 `--mode rtc`。

### 数字孪生：`piper_data_reviewer`

本机独立工具 `/home/ysh/piper_data_reviewer` 可把 HDF5/RLDS 的相机、state 和 action
放在同一时间轴播放，并用**当前 robokit checkout 的** `ChunkExecutor`、`ActionGuard`
和 `SimArm` 显示 commanded target 与一阶跟随反馈。它只读源数据，不导入 Piper SDK，
不打开 CAN/ROS，也不会向真机发送命令。

以下命令都直接在 `/home/ysh/robokit` 终端执行；依赖已安装，不重复运行
`pip install -r requirements.txt`。

**按任务看（推荐）**：`--task` 把一个任务的本地 episode 全部拉到一条编号时间轴上，不管它们
现在躺在哪 —— 分批发布会把已发布的段从 `datasets/<任务名>/` **搬进**
`datasets/_batches/<仓库>/<slug>/<批次>/hdf5/`，只看任务目录就只剩还没发的那几段：

```bash
python /home/ysh/piper_data_reviewer/run.py \
  --task "Open the drawer, put the fruit and the cup from the table inside, and close the drawer." \
  --robokit-root /home/ysh/robokit \
  --robokit-config configs/piper_single.yaml \
  --port 0 \
  --idle-timeout 30 \
  --open-browser
```

启动时会打印这一条视图的来源构成，例如
`任务视图：79 段 = 已发布 73 段（批次 b0_74） + 未发布 4 段 + 清洗隔离 2 段`。
任务名和 slug（空格换下划线）都认，尾随空格会被忽略；清洗隔离的段也在列表里，路径带
`_quarantine` 一眼可辨 —— 「52 为什么被判 bad」正是要在孪生里看的。`--task` 省略 `--data`
时默认扫 `<robokit-root>/datasets`，实时跟随照常生效（每 2s 重扫，只认这个任务）。

不给 `--task` 就是老用法，直接指一个目录或单个文件：

```bash
python /home/ysh/piper_data_reviewer/run.py \
  --data "datasets/_batches/Memoryvla/Place_the_banana,_kiwi,_and_wax_apple_into_the_basket_in_that_order./b0_74/hdf5" \
  --robokit-root /home/ysh/robokit --port 0 --idle-timeout 30 --open-browser
```

`--port 0` 会自动选择空闲端口，实际地址以终端打印的 `Piper reviewer: http://...`
为准，避免已有 reviewer 占用默认 `8765` 时启动失败。关闭全部 reviewer 页面后，
服务会在 30 秒内自动退出并释放端口；终端 `Ctrl+C` 可立即退出。需要常驻时才用
`--idle-timeout 0`。

#### 边采集边让同学在他自己电脑上审查

同学看的是同一个正在生长的目录：`collect.py` 每停一条录制，`EpisodeRecorder.close()`
把 `N.hdf5.tmp` 重命名成 `N.hdf5`，新 episode 就在几秒内自己出现在他已打开的页面上，
不需要重启服务，也不需要他刷新。录制中的 `.tmp` 只显示成「● 正在录制」，不会被当成
可打开的 episode——HDF5 还没写完。

先自检网络（不读数据、不起服务，可以提前跑）：

```bash
python /home/ysh/piper_data_reviewer/run.py --check-network --port 8765
```

它会指出哪个地址能发给同学、哪些是本机代理 TUN（**本机常开 Mihomo TUN，
`198.18.0.1`/`2.0.0.1` 发过去必然连不上**）、ufw 是否在挡这个端口并给出放行命令。
本机 ufw 默认是开的，不放行时同学的连接会被内核直接丢弃，表现和校园网隔离一模一样。

再起服务：

```bash
python /home/ysh/piper_data_reviewer/run.py \
  --data /home/ysh/robokit/datasets \
  --robokit-root /home/ysh/robokit \
  --share --port 8765 --idle-timeout 0
```

把终端打印的**完整链接**（含随机 `token`）发给同学。页面右上角有「跟随最新」，
勾上就会随着你采集自动切到刚录完的那一条；他手动选了别的文件会自动取消勾选。
`--watch-interval` 控制重扫间隔（默认 2s，`0` 关闭）。

数据只读，服务不接触 CAN/ROS/真机；分享结束在终端按 `Ctrl+C`，链接立即失效。这是局域网
访问而非公网上传。校园网 AP 开了客户端隔离时无解，B 计划是手机开热点、两台电脑都连上去、
重跑命令（IP 会变）。

页面可切换 observation、原始 action、当前部署执行器生成的命令轨迹和 `SimArm` 反馈，
也可调整 `continuous / recursive / feedback`、horizon，并用 shadow guard 查看候选动作
是否会被安全闸拦截。只想看数据、不加载部署执行器时加 `--no-deploy-runtime`。

直接查看 RLDS 需要 TensorFlow/TFDS，使用 `pi0_demo` 环境：

```bash
conda run -n pi0_demo python /home/ysh/piper_data_reviewer/run.py \
  --data ~/tensorflow_datasets/stack_cups/robokit_dataset/1.1.0 \
  --robokit-root /home/ysh/robokit \
  --port 0 \
  --idle-timeout 30 \
  --fps 30
```

GPU policy 在环时，先按上文启动 `deploy_server.py`（远端服务用 SSH 转发到本机 8080），
再启动 reviewer：

```bash
python /home/ysh/piper_data_reviewer/run.py \
  --data "datasets/stack cups" \
  --robokit-root /home/ysh/robokit \
  --robokit-config configs/piper_single.yaml \
  --policy-host 127.0.0.1 --policy-port 8080 \
  --instruction "stack cups" \
  --port 0 \
  --idle-timeout 30 \
  --open-browser
```

推理只在页面点击「推理并执行」后发生；每次运行会新建 TCP 连接，因此
`deploy_server` 会把它当成新 episode 并重置 policy 记忆。trace 写到
`runs/policy-viewer/<run_id>/trace.jsonl`。

边界：policy 在环仍是**开环视觉**——图像按 demo 帧推进，不随孪生臂运动改变，
所以不能据此宣称任务成功。`SimArm` 默认 `tau=0.12s` 尚未用当前真机标定，
feedback 毫米误差只适合相对比较；commanded target 才是执行数学的直接结果。
RLDS 不含关节反馈时，3D 关节姿态是从 EEF pose 做 IK 重建的，不是真机编码器真值。

### EEF 执行链路

模型输出局部 delta 位姿，客户端累加成绝对目标，然后：

```
EEF 目标 → 主机连续有界 IK → MOVE_J / JointCtrl → 到位验收
```

当前生产配置显式使用真正的 Pinocchio 后端：
`eef_backend: pinocchio_ik`。旧 `host_ik`（SDK FK + SciPy）仍可显式选择作 A/B，
但两者是不同类和不同模型，不会把 Pinocchio 配置静默转给旧求解器。

不要把两套“官方 IK”混在一起：

- `piper_sdk.EndPoseCtrl` 只把末端目标发给主控，逆解实际发生在固件的
  `MOVE_P` 解析解中。松灵维护者已确认它在多解点可能发生关节角突变，
  且不会缓存上一条解析结果，并建议改用 `piper_ros` noetic 的 Pinocchio
  主机逆解（[piper_sdk #96](https://github.com/agilexrobotics/piper_sdk/issues/96)）。
- `piper_ros/piper_pinocchio.py` 是 Pinocchio + CasADi/IPOPT 的独立主机示例，
  不是 `piper_sdk` 内置 IK。原示例的上一解平滑项被注释、只正则到零位，
  大于 30° 的解仍会先返回再把下一次 seed 清零，也没有求解后 FK 残差硬验收；
  历史上还修过 RPY 轴约定错误
  （[piper_ros #30](https://github.com/agilexrobotics/piper_ros/pull/30)）。

本项目实现位于 `robokit/arms/piper_pinocchio_ik.py`，没有复制 demo 的控制节点：

- 生产模型是仓库内版本化的 `robokit/assets/piper_description.urdf`，活动关节顺序强制
  为 `joint1..joint6`，末端 frame 强制为 `link6`；运行时还要求 URDF 限位与当前
  `joint_limits_deg` 完全一致。
- 构造时用 SDK FK **只做模型对拍**，四组关节样本的最坏差为
  **0.127mm / 0.0037°**；真正的求解、Jacobian、realized FK 全部来自 Pinocchio。
- 位姿残差为末端 LOCAL frame 的
  `log6(T_current⁻¹ T_target)`；Jacobian 使用同一 LOCAL frame，并经 `Jlog6`
  求导。policy 的 local EEF delta 仍由原 `apply_local_delta_pose` 生成绝对目标，
  IK 不改动作空间。
- 每次 `move_eef` 都从最新物理关节反馈重新 seed。优化变量直接受六轴机械限位和
  `feedback seed ±5°` 交集约束，再加上一解连续性代价；不是先求任意解再裁单轴。
- `JointCtrl` 按 0.001° 量化后再次验限，并用 **Pinocchio FK** 生成
  `eef_target_realized`。到位等待只比较该实际可实现位姿，不比较理想 EEF 请求。

`pinocchio_ik` 关键参数：

| 参数 | 值 | 作用 |
|---|---|---|
| `max_step_deg` | 5.0 | 单步关节上限 |
| `continuity_weight` | 0.03 | 抑制腕奇异点附近 j4/j6 换支 |
| `seed_limit_tolerance_deg` | 10.0 | 物理反馈超限 `≤10°` 投影 seed，`>10°` abort |
| `allow_best_effort` | 正常配置 false；no-guard true | 是否把 pose residual / max iterations 仅记 trace |
| `max_nfev` | 200 | 有合法 best candidate 时，达到上限不等同于拒发 |

前两项**不是安全闸**，是「同一位姿有多组关节解时选哪一支」的机制。去掉它们反而更不忠实于
policy 的信号：实测单步关节跳变到 103°（j1 一步从 -71.8° 甩到 -142.0°），臂在关节空间乱甩，
走出的物理轨迹不是 policy 输出的那条。

无护栏配置的 `allow_best_effort: true` 只把理想目标残差、限位饱和和求解器
`converged` 状态写入 trace；只要 best candidate 有限、在机械限位内且单步不超过
5°就发送。NaN/Inf、反馈陈旧、反馈超限 `>10°`、没有任何合法候选、量化后越限以及
CAN/驱动/控制器异常仍然 abort。

Pinocchio 3.6.0 是机器人端环境的显式依赖；已有环境更新：

```bash
conda install -n robokit -c conda-forge pinocchio=3.6.0 "numpy<2"
```

### π0 关节角训练与执行（训练 50/30，当前运行 chunk=15、30Hz）

这套对照使用 LeRobot `lerobot/pi0_base`，不是 MemoryVLA/PI0.5。源仍为 202 段
`stack cups` HDF5，转换为独立的 `shaohuan1/stackcups_joint_pi0_lerobot`：

- `observation.state = 当前 [j1..j6, gripper]`
- `action = 下一帧绝对 [j1..j6, gripper]`
- 六轴单位为弧度；`use_relative_actions=false`
- checkpoint 训练配置保持 `chunk_size=50`、`n_action_steps=30`
- 当前运行时服务只回传预测块前 15 行，客户端也只执行 15 行；不需要重训
- LoRA r32/alpha32、batch 8、**30,000 step**、每 1000 step 保存/评估
- 固定跑满 30k，不做早停；最终发布最后一个 `030000/pretrained_model`，不按 eval 选 best

PI0 base 固定 revision `25c379b52ba2ff8788cab921758a3cc3fe3f77f2`；
`prepare_pi0_base.sh` 用多连接下载后核对 14,005,618,584-byte
`model.safetensors` 的 SHA-256
`8229fd9a7c3c2aafc1e223567b61b5fe3e25eef873bb4233928dbee4bd836303`。
旧误配 run（10k、执行 39）已归档，不得当作候选权重。当前正确 run：
[stackcups_joint_pi0_h30 / i8tmjy90](https://wandb.ai/yangshaohuan720-university/stackcups_joint_pi0_h30/runs/i8tmjy90)。
该 run 已固定跑满 30,000 step，完成标记为
`/root/autodl-tmp/outputs/stackcups_joint_pi0_h30/COMPLETED`；最终 checkpoint 配置回读为
π0、`chunk_size=50`、`n_action_steps=30`、`use_relative_actions=false`，发布的是最后
一个 `030000`，不是 held-out eval best。

服务器后台入口：

```bash
set -a
source /root/autodl-tmp/secrets/pi05.env
set +a
export PATH=/root/autodl-tmp/envs/pi05/bin:$PATH
export PYTHON_BIN=/root/autodl-tmp/envs/pi05/bin/python
bash pi0/run_joint_training_server.sh
```

训练完成后在独立端口 8081 启动关节服务，避免与当前旧 EEF 服务的 8080 冲突：

在 `configs/models.yaml` 里加一条 `family: pi0` 的记录（`action_space: joint`、
`port: 8081`、`horizon: 15`），之后两端都只用 `--model` 指名，不再有第二套脚本：

```bash
# GPU 服务器
python scripts/serve_policy.py --model pi0-joint --mode sync
```

关节配置 `control_freq_locked: true` 会拒绝任何不是 30Hz 的 `--control-freq` 覆盖：

```bash
# 先建立 ssh -L 8081:localhost:8081 ...，再做零运动检查
python scripts/run_policy.py --model pi0-joint --dry-run --max-steps 15
```

确认 dry-run 的 `arm_command.joint_target_deg` 后，真机只跑一块：

```bash
python scripts/run_policy.py --model pi0-joint --max-steps 15 \
  --trace runs/deploy/pi0-joint-real.jsonl
```

服务明确回报 `server_mode=sync`、`action_space=joint`；误连 EEF 服务会在执行器前失败。
执行路径直接调用 `JointCtrl`，不会做 EEF 累加或 IK。第一次真机执行仍应先审查 dry-run
的 `(15,7)` 输出，再以少轮数测试。

2026-07-29 当前实例为远端 PID 586280、8081、`--horizon 15`，本机 SSH 隧道监听
`127.0.0.1:8081`；旧 EEF 8080 同时保留。切换前 H=30 测试曾确认 π0 flow 采样有随机
关节越界，处理方式见下方限位钳制。

2026-07-29 用户决定这套 **joint 专用配置**关闭 ActionGuard，并把有限的 6-D 直接关节
目标逐轴钳到最近机械限位后执行：`deploy.safety.enabled=false`、
`joint_limit_mode=clip`。例如 j2=-1.2° / j3=0.9° 会成为 0° / 0°；
dry-run 也走同一套纯软件量化/钳制路径，在 trace 的 `arm_command` 中同时记录
`joint_requested_deg`、`joint_clipped_axes`、`joint_limit_clip_delta_deg` 和最终
`joint_target_deg`，但不发 CAN。无需再传 `--no-guard`。该行为只作用于
`configs/piper_single_joint.yaml` 的直接 joint 路径；EEF/IK 配置保持严格拒绝。
错误维度、NaN/Inf、反馈/CAN/驱动器/控制器异常仍会硬中止。

当前端到端证据 `runs/deploy/pi0-joint-h15-30hz-fixed-dry-run.jsonl`：服务返回
`(15,7)`，15/15 行、0 abort、0 CAN；15 行的 j2/j3 均按配置钳制，最终六轴全部合法。
绝对 deadline 调度实测平均间隔 33.395ms，即 29.944Hz；它会扣除每步软件处理耗时，
而不是处理完成后再额外 sleep 33.333ms。非锁定的旧 EEF 配置保持历史调度行为。

### 安全闸

`robokit/safety.py`，配置在 `deploy.safety`，下发前逐条检查，超限立即中止：
Δxyz / Δrpy 上限、目标跳变、跟踪落后、夹爪速率、工作区盒子。相机侧另有亮度 / 帧龄 /
冻结帧三闸，IK 和到位等待也有各自的拒绝条件。

这些层由 `scripts/run_policy.py --safety` **一起**开关，不再有「只关了一半」的中间态：
`--safety on` 全部启用（配置里的生产阈值），默认的 `--safety off` 全部解除。改了哪些
字段见 `robokit/deploy/runtime.py` 的 `relax_safety()`，**只保留上面两项 IK 分支约束
及不可绕过的硬件错误**。启动日志会逐条打印解除了什么。

---

## π0.5 单相机 LoRA（`pi0.5/`）

专用链路：

```text
robokit HDF5
  → 单相机 LeRobot v3（MP4 + Parquet）
  → Hugging Face Dataset
  → lerobot/pi05_base + PEFT LoRA
  → Hugging Face LoRA Adapter
  → robokit TCP 服务
  → scripts/run_policy.py 同步或 RTC 异步真机执行
```

### 已定方案

- **用 LeRobot，不用 RLDS。** `shaohuan1/lememory` 同时有 HDF5 和 RLDS；PI05 训练消费
  LeRobot。虽然现成 RLDS 较小，但 RLDS→LeRobot 还要装 TensorFlow、读 TFRecord 后再次
  转码。当前选择结构可直接核验且不增加 TensorFlow 的 HDF5 路径：首次在数据盘从 HF
  下载 202 个 HDF5，转换一次并把 MP4+Parquet 发布到
  `shaohuan1/stackcups_one_task_lerobot`；以后服务器可直接下载这个 LeRobot 派生仓库，
  不再重复下载或转换原始 HDF5。
- **单图输入可用。** 训练命令的 `--policy.input_features=null` 从数据集推断输入，本数据
  只有 `observation.images.cam_high`。PI05 至少需要一张有效图，但不强制腕部相机；沿用
  多相机槽位时，缺失视角也可用黑图加 `mask=false` 屏蔽。因此当前无需用户定夺，默认
  固定单路 `cam_high`。多视角可能减少遮挡、提高效果，但那是性能取舍，不是接口限制。
- **不启用 LeRobot 的 relative action。** 转换出的 action 已经是局部 EEF 增量，再设
  `--policy.use_relative_actions=true` 会错误地从增量上再减一次 state。

数据契约：

| 字段 | 7 维含义 |
|---|---|
| `observation.state` | 当前 `[x,y,z,roll,pitch,yaw,gripper]` |
| `action` | 下一帧的局部 `[dx,dy,dz,droll,dpitch,dyaw,next_gripper]` |

单位是米 / 真弧度，固定轴外旋 `xyz`，与 `robokit/pose.py` 和
`ChunkExecutor(action_space="eef_delta")` 相同；每段 T 帧 HDF5 生成 T-1 个 LeRobot 帧。

2026-07-28 全量 dry-run：**202 段、45,496 个原始时刻、45,294 个训练帧、10,505,312,600
bytes HDF5、320×240 RGB、30 Hz、单 `right_arm` / `cam_high`、任务均为 `stack cups`**。
清洗结果 36 ok + 166 warn + **0 bad**。warn 主要是欧拉数值 wrap；转换按 SO(3) 相对旋转
计算，不会把 wrap 误制成大动作。全量局部动作绝对最大值：
`[12.81,14.35,12.93] mm / [3.77°,4.60°,4.12°]`。

### 文件用途

| 文件 | 用途 |
|---|---|
| `prepare_lememory.py` | 只下载 HF 两批 stack-cups HDF5/元数据，核对 0..201 连续编号并合并清洗报告 |
| `convert_hdf5_to_lerobot.py` | 清洗硬闸、全量 HDF5 校验、单相机 LeRobot 转换和 HF 上传 |
| `validate_lerobot.py` | 检查单图、7 维 state/action、quantile stats 和抽样有限值 |
| `prefetch_hf.py` | 服务器预下载 Dataset、PI05 base 和 LoRA adapter 到 HF cache |
| `prepare_paligemma_tokenizer.py` | 校验 OpenPI 公开 tokenizer、验证 token IDs，并生成不依赖 gated repo 的本地 PI05 视图 |
| `train_lora.sh` | LeRobot 原生 PEFT LoRA，超参数由环境变量覆盖 |
| `monitor_convergence.py` | 读取 held-out eval loss；平台达到条件后等完整 checkpoint 落盘再终止 trainer |
| `publish_checkpoint.py` | 把确认完整的 LoRA `pretrained_model` 发布到权重仓库根并回读验证 |
| `run_training_server.sh` | HF 准备→转换/发布→训练→W&B→收敛发布→条件关机的服务端守护入口 |
| `test_contract.py` | 验证 EEF 局部增量、RTC 切块/跳步/deadline 和服务端缓存契约 |
| `requirements.txt` | 固定核对过的 LeRobot revision 和 PI05/PEFT 依赖 |

推理/执行本身已经不在本目录：PI0 与 PI0.5 的加载、sync/RTC 服务、RTC 控制器和两条
执行循环统一在下面这几个文件里，两端各只有一个入口脚本。

| 文件 | 用途 |
|---|---|
| `configs/models.yaml` | 模型登记表：`--model` 的名字、checkpoint、动作空间、端口、默认 horizon |
| `robokit/policies/lerobot_dit.py` | PI0/PI0.5 × 全量/adapter × EEF/joint × sync/RTC 的单一适配器 |
| `robokit/deploy/registry.py` | 登记表解析与 horizon 校验 |
| `robokit/deploy/runtime.py` | 观测打包、相机闸、护栏解除、会话生命周期（两条循环共用） |
| `robokit/deploy/rtc.py` | 论文 Algorithm 1 状态机、单请求在飞异步客户端、逐行执行器 |
| `robokit/deploy/loops.py` | sync 与 RTC 两条调度循环 |
| `robokit/deploy/obsview.py` | 看/落盘「模型端拿到的图像」 |
| `scripts/serve_policy.py` | GPU 服务端唯一入口 |
| `scripts/run_policy.py` | 机器人端唯一入口 |

### 从 `shaohuan1/lememory` 转换并上传数据

服务器只取两批 `stack_cups_b*/hdf5/*.hdf5` 与 `source_meta`，不会把现成 RLDS 再下载一份：

```bash
export HF_HOME=/root/autodl-tmp/hf
export HF_LEROBOT_HOME=/root/autodl-tmp/hf/lerobot
python pi0.5/prepare_lememory.py \
  --source-repo shaohuan1/lememory \
  --output /root/autodl-tmp/pi05/source/stackcups \
  --expected-episodes 202
```

脚本要求恰好 202 段且编号连续为 0..201；源文件在 HF cache 中只存一份，输出目录使用
symlink 平铺，不复制 10.5 GB 数据。

若本机已有原始数据，也可先过清洗闸，再做无需 LeRobot 的全量预检：

```bash
python scripts/clean.py --data "datasets/stack cups" --quarantine
python pi0.5/convert_hdf5_to_lerobot.py \
  --data "datasets/stack cups" \
  --repo-id YOUR_HF_USER/piper-stack-cups-pi05 \
  --dry-run
```

固定的 LeRobot 0.6.1 revision 要求 Python ≥3.12，建立独立 3.12 环境：

```bash
python3.12 -m venv .venv-pi05
source .venv-pi05/bin/activate
pip install -r pi0.5/requirements.txt
hf auth login
```

转换为 MP4+Parquet 并上传。输出已存在时默认拒绝覆盖；确认重建才加 `--overwrite`：

```bash
python pi0.5/convert_hdf5_to_lerobot.py \
  --data "datasets/stack cups" \
  --repo-id YOUR_HF_USER/piper-stack-cups-pi05 \
  --root "/path/to/converted/piper-stack-cups-pi05" \
  --camera cam_high \
  --push-to-hub --private --upload-large-folder

python pi0.5/validate_lerobot.py \
  --repo-id YOUR_HF_USER/piper-stack-cups-pi05
```

输出根的 `conversion_manifest.json` 记录源 episode 清单、相机、机械臂、FPS、动作契约和欧拉序；
`source_meta/` 同时保存采集 `config.json` 与这次全量 `clean_report.json`。

### 服务器一键训练、W&B 与收敛关机

缓存必须放数据盘，不能落到 30 GB 系统盘：

```bash
export HF_HOME=/root/autodl-tmp/hf
export HF_LEROBOT_HOME=/root/autodl-tmp/hf/lerobot
export HF_HUB_DISABLE_XET=1
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME"

cd /root/robokit
conda create -y -p /root/autodl-tmp/envs/pi05 python=3.12 pip
/root/autodl-tmp/envs/pi05/bin/pip install -r pi0.5/requirements.txt
```

若服务器不能访问 GitHub，把固定提交
`f37be3edbee60f3a09a5183788b91eb19f0c07d1` 的 LeRobot 源码传到数据盘后执行：

```bash
/root/autodl-tmp/envs/pi05/bin/pip install \
  "/root/autodl-tmp/src/lerobot-f37be3ed[pi,peft,training]" h5py scipy
```

`HF_TOKEN`、`WANDB_API_KEY` 只通过服务器权限 0600 的环境文件注入。当前任务的完整入口：

```bash
set -a
source /root/autodl-tmp/secrets/pi05.env
set +a
export PATH=/root/autodl-tmp/envs/pi05/bin:$PATH
export PYTHON_BIN=/root/autodl-tmp/envs/pi05/bin/python
bash /root/robokit/pi0.5/run_training_server.sh
```

PI05 的公开权重 processor 默认仍引用 gated 的 `google/paligemma-3b-pt-224` tokenizer；
若 HF 账号没在该模型页接受条款会报 403。守护入口不会借用第三方模型权重，而是调用
`prepare_paligemma_tokenizer.py`：下载 OpenPI 官方代码使用的匿名公开
`gs://big_vision/paligemma_tokenizer.model`，校验固定
SHA-256 `8986bb4f…a168fc6` 与 4,264,023 bytes；所需 Transformers metadata 也逐文件固定
大小/SHA，并以 stack-cups 的 PI05 完整 prompt 验证 token IDs 与官方 SentencePiece
完全一致。随后创建 `/root/autodl-tmp/models/pi05_base_local`，其中模型权重仍 symlink
到已校验的 `lerobot/pi05_base` blob，只把本地 processor 的 tokenizer 路径改到数据盘。

默认配置为数据 `shaohuan1/lememory`、派生数据
`shaohuan1/stackcups_one_task_lerobot`、权重 `shaohuan1/stackcups_one_task`；
rank=32、alpha=32、batch=8、最多 30k steps。LoRA 目标由当前 LeRobot PI05 实现提供：
action expert attention 的 q/v 投影与 state/action 投影层。W&B project/run 名均为
`stackcups_one_task`，每 50 step 记训练曲线，每 1000 step 在最后 10% episode 上记
`eval_loss` 并保存 checkpoint。

自动收敛判据不是训练 loss：至少 10k step、至少 6 次 held-out evaluation，并且连续 4 次
相对改善不到 0.5%。命中后守护器先等待平台命中 step 的 adapter/config/training state
全部落盘，再终止 trainer；随后按同一个 0.5% 有效改善阈值选择 **best checkpoint**
发布到 `shaohuan1/stackcups_one_task` 根目录并通过 HF 文件列表回读验证，而不是发布后面的
平台探测 checkpoint。**只有这些步骤全部成功才执行 `shutdown -h now`**。训练异常、到
30k 仍未满足判据、W&B/HF 发布失败都保留服务器，供人工检查。held-out loss 平台只代表
离线优化停止，不能替代后续真机任务成功率验收。

HF 推理权重由 `lerobot/pi05_base` 与任务 LoRA adapter 两部分组成；adapter 的
`adapter_config.json` 引用底座。显存不足先把 batch 降到 4/2。

本次训练的 W&B run：
[stackcups_one_task / fnbhi812](https://wandb.ai/yangshaohuan720-university/stackcups_one_task/runs/fnbhi812)。
2026-07-29 已完成收敛：W&B API 回读到 train step 24,400 和 24 个 held-out eval 点；
step 16,000 的 `eval_loss=0.14999647438526154` 是按 0.5% 有效改善规则选出的 best，
17k–24k 连续 8 次没有有效改善，平台判据成立。step 19,000 的绝对数值
`0.14998169243335724` 仅比 16k 低 `1.48e-5`（约 0.0099%），不足以重置平台计数。
最终 HF 提交 `47690b157cafc54abbf0ccab33b930f21f62e972`，标题为
`Publish converged PI0.5 LoRA from 016000`；根目录 adapter/config 的 blob/LFS SHA
与 `checkpoints/016000/pretrained_model` 完全一致。发布回读完成后服务器已按约定关机。

### 未收敛候选权重的同步评测

需要在最终收敛前先做真机早期评测时，不能简单选择最新 step。2026-07-29 已审查并
实际部署过的候选固定为 step 12000：held-out `eval_loss=0.1520`；它的
adapter/config/preprocessor/postprocessor 均完整，但仍是候选权重，不能当作最终成功率结论。
最终收敛权重是 step 16000，但尚未经过用户数字孪生审查；训练结束时服务器曾关机，
2026-07-29 已重新开机。要评测某个具体 step 而不是登记表里的默认权重，用
`--checkpoint` 显式覆盖，登记表本身不会被静默改写。

GPU 服务器启动：

```bash
cd /root/robokit

# PI0.5 预测 chunk=50；服务返回完整 50 行，执行几步由机器人端 --horizon 决定
/root/autodl-tmp/envs/pi05/bin/python scripts/serve_policy.py \
  --model pi05-eef --mode sync
# 换候选权重：--checkpoint /root/autodl-tmp/outputs/.../checkpoints/012000/pretrained_model
```

机器人端先做一次**零运动 dry-run**；客户端要求服务端明确回报 `server_mode=sync`，
接错 RTC 端口会在 action 进入执行器前中止：

```bash
cd /home/ysh/robokit
conda activate robokit
python scripts/run_policy.py --model pi05-eef --mode sync \
  --eef-backend pinocchio_ik --dry-run --max-steps 30

# SSH 隧道重建后的标准形式：
# ssh -N -L 8080:127.0.0.1:8080 -p <SSH端口> <用户>@<服务器>
```

真机必须先回到与杯子现场对应的示教起点。下例使用 episode 0；若现场按其他 episode
摆放，把 `--episode` 改成对应编号。`--controller-reset` 与
`--skip-controller-reset` 必须按从臂是否已独立重启二选一：

```bash
cd /home/ysh/robokit
conda activate robokit
python scripts/reset_piper_to_demo_start.py \
  --dataset "datasets/stack cups" --episode 0 --port can0 \
  --execute --skip-controller-reset

# 原机制真机执行：EEF delta、recursive、每块 30 步、护栏默认解除、总步数不限
python scripts/run_policy.py --model pi05-eef --mode sync \
  --horizon 30 --eef-backend pinocchio_ik
```

`--safety off`（默认）关闭 ActionGuard、相机亮度/帧龄/冻结检查、Pinocchio IK 位姿
residual 与伺服 lag 拒绝、夹爪限速和到位超时；退出不归位、不做 controller reset，
机械臂会带力留在最终位姿。EEF-delta、recursive、30Hz 原控制逻辑不变。
`--max-steps 0`（默认）表示总执行步数不限，一直运行到再次回车、`Ctrl+C`、通信/非有限
模型输出、没有合法有界 IK candidate、反馈超限 `>10°`、CAN/驱动器/控制器异常等硬故障。
`max_step_deg=5` 和
`continuity_weight=0.03` 仍保留，它们用于避免 IK 在等价关节解之间跳支，不是位姿拒绝阈值。
训练守护与候选 checkpoint 独立，真机评测不会改训练权重。

2026-07-29 首次真机运行在第 36 个 action 报
`arrival_timeout: xyz=0.53mm, rot=1.00°, joint=0.074°`。trace 证明机械臂已经到达
下发关节目标；失败原因是 host IK 合法接受了相对理想目标 `0.902°` 的残差，而旧到位逻辑
仍要求反馈相对那个未被实际下发的理想 EEF 目标严格小于 `1.000°`，两套相同上限没有硬件
测量余量。现已改为：IK 发送量Open the drawer, put the fruit and the cup from the table inside, and close the drawer.化关节目标时同时返回其 FK 可实现位姿；到位逻辑对该实际
命令位姿验收，理想目标与可实现位姿的 1mm/1° 残差仍由 IK 在 CAN 发送前独立检查。
这项修复没有改变 EEF-delta 递推、模型动作或到位等待。

同日第二次真机运行 `runs/deploy/20260729-022237.jsonl` 在 action 50 被
`host_ik pose_tolerance` 中止：位置残差 `0.911mm` 已通过，旋转残差 `1.215°`
略高于生产配置 `1.0°`，关节步长达到保留的分支约束 `5.0°`，因此没有发送 CAN。
这说明旧命令里的 `--no-guard` 只关闭 ActionGuard，并没有加载已经存在的完整解锁配置。
现在这些层由 `--safety off`（默认）统一解除，不会再因该 IK 位姿阈值或到位等待中止。

之后的旧 `host_ik` 长时程 policy trace 又出现 15 次 `host_ik_input`：上一条主机 IK 已把 `j5`
合法限制在 `70.000°`，但伺服贴着限位运行时反馈会略过 70°；旧 `reset_seed` 固定只容忍
0.3°，于是下一条 EEF action 尚未求解就 abort。现在 `seed_limit_tolerance_deg` 可配置：
无护栏 policy 配置为 **10°**。反馈逐轴超限 `≤10°` 时只把 IK seed 投影回最近的机械
限位继续求解；`>10°` 才按大幅越界拒绝。当前 `pinocchio_ik` 的正常与 no-guard 配置
都执行这条物理反馈规则，IK 变量和最终 `JointCtrl` 始终受原六轴硬边界约束。
trace 的 `seed_limit_overrun_deg` 可直接看到每轴投影量。

现有 HDF5 已按真机同步 policy 的控制语义做过全量验证，不是直接回放绝对关节角：

```bash
python scripts/validate_hdf5_policy_path.py \
  --config configs/piper_single.yaml --safety off \
  --data-dir "datasets/stack cups" \
  --horizon 30 \
  --report runs/validation/hdf5-policy-path-pinocchio.json
```

验证把 HDF5 相邻 EEF 位姿转成训练同款 local EEF delta，按 `recursive + H=30` 进入与
真机部署相同的
`ChunkExecutor → PiperArm.move_eef → _create_eef_ik → PiperPinocchioIK → JointCtrl`
路径；验证器没有复制 IK 数学，只用内存 SDK 接收最终 CAN 整数帧，因此不会启动真机或
数字孪生。

2026-07-29 全量结果：**202/202 episode、45,294/45,294 commands、0 abort、
0 JointCtrl 越限**；最大位置/旋转 residual **0.604mm/0.669°**，最大关节步长
**5.000°**，物理反馈最大超限投影 0.028°，449 条候选触及局部/机械边界，
0 条未收敛，最大 21 iterations。报告保存在
`runs/validation/hdf5-policy-path-pinocchio.json`。这证明现有数据轨迹不会再被
本机软件判据中止；CAN、驱动器、现场碰撞和真实伺服跟踪仍只能由人工在真机运行时确认。

验证器同时逐帧比较
`Pinocchio JointCtrl[action i] - HDF5 observations/joint[i+1]`（度，不做角度 wrap）：

| 统计 | 六轴合并 |
|---|---:|
| 平均绝对误差 / RMSE | **0.0315° / 0.1678°** |
| P50 / P95 / P99 绝对误差 | **0.0090° / 0.0980° / 0.4510°** |
| 最大绝对误差 | **9.6430°** |

逐轴 MAE 为 `j1..j6 = 0.0293°, 0.0143°, 0.0209°, 0.0530°, 0.0138°, 0.0579°`。
最大差异位于 `31.hdf5` action 37 的 j6；同一点 j4/j6 相对数据分别相差
8.952°/9.643°，但两组关节角的 FK 末端只差 **0.036mm / 0.025°**。该点 Jacobian
条件数约 `2.17e4`、最小奇异值约 `8.1e-5`，且差异在相邻动作中平滑产生并恢复：
这是腕部奇异点附近同一末端位姿的零空间放大，不是模型零位错误或随机换支。EEF policy
没有数据集参考关节角可用于指定示教轨迹的零空间，因此不能保证逐关节完全重现；生产
约束仍保证每步不超过 5°、最终命令不越机械限位。

**RTC 暂停真机：**step-12000 在 `92.hdf5` 的已有数字孪生 trace 中，RTC 第四个 guided
chunk 的局部位移均值增至 10.12mm、最大 12.76mm，并连续累积到
`x=-0.101m`，被 workspace guard 中止；相同 checkpoint 的原同步机制完成 180/180
动作且 0 违规。根因是本数据动作是“以上一帧 EEF 为基准的逐步局部 delta”，而当前
LeRobot `RTCProcessor` 直接把 action 向量当作可跨 chunk 对齐的轨迹坐标做 prefix
guidance。上游对 relative-action policy 会先把剩余动作恢复到绝对空间、按当前 state
重锚并重新归一化；本项目这种逐步局部 SE(3) delta 还需要自定义的可微累积/重锚变换，
不能直接套现成 processor。

因此 **RTC 请优先配 `--model pi05-joint`**：绝对关节目标正是 prefix guidance 假设的
「可跨 chunk 对齐的轨迹坐标」，语义成立。`--model pi05-eef --mode rtc` 仍然可以选，
但启动时会打印上面这条已知问题的警告 —— 在完成可微累积/重锚变换之前，它的 guided
chunk 会逐块放大位移。

### 普通同步推理

服务器：

```bash
python scripts/serve_policy.py --model pi05-eef --mode sync
```

机器人端第一次只走 dry-run：

```bash
python scripts/run_policy.py \
  --model pi05-eef --mode sync --horizon 30 \
  --dry-run --max-steps 60 \
  --trace runs/deploy/pi05-dry-run.jsonl
```

确认返回 `(N,7)`、相机亮度正常、安全闸没有持续拒绝、动作量级合理后才去掉
`--dry-run`。动作契约的纯软件验证为 `python pi0.5/test_contract.py`。

### RTC 实验性异步推理（arXiv:2506.07339，暂禁真机）

这条链按论文 **Real-Time Chunking (RTC)** 做，不是对输出事后均值平滑：

1. 当前 action chunk 在机器人端继续执行，下一次 PI05 推理同时在服务器运行。
2. 服务器保存上一个 chunk 的**归一化**版本；新 chunk 每个 flow denoising step 都通过
   LeRobot `RTCProcessor` 做 ΠGDM 向量-雅可比积引导。
3. 前 `d` 个必然在推理期间执行的 action 权重为 1；中间重叠区使用论文式指数衰减；
   最后 `s` 个新 action 权重为 0。新 chunk 到达后跳过已经过去的 `d_obs` 行再切入。
4. 最近 `b=10` 次实测 action-step 延迟取最大值，作为下一轮保守 `d`。如果不再满足
   `d <= s <= H-d`，或旧 chunk 在回包前耗尽，立即中止，绝不重复最后一条动作。

论文实机参数 `H=50, s_min=25, n=5, beta=5, b=10` 已作为默认值；soft mask 默认 `EXP`。
`H` 始终从 checkpoint 回包读取，若 LoRA checkpoint 不是 50 步也会按实际 H 校验。
RTC 是纯推理时方法，**不改变 LoRA 训练**。

服务器改用 RTC 入口：

```bash
python scripts/serve_policy.py --model pi05-joint --mode rtc \
  --num-inference-steps 5 --max-guidance-weight 5 --schedule EXP
```

机器人端第一次仍只做零运动检查（`--dry-run` 会自动关掉到位等待，因为命令被故意拦截、
机械臂不会真的到达目标）：

```bash
python scripts/run_policy.py \
  --model pi05-joint --mode rtc --horizon 15 \
  --dry-run --max-steps 60 \
  --trace runs/deploy-rtc/pi05-dry-run.jsonl
```

去掉 `--dry-run` 即真机执行。`--horizon` 在 RTC 下就是论文的 `s_min`。`d_init` 默认由
两轮静止 warmup 的最后一轮耗时换算并加一个控制 tick，也可用 `--initial-delay-steps N`
显式给定。EEF 模型跑 RTC 的已知问题见上面的“RTC 暂停真机”。

**EEF 控制逻辑没有改变：**服务端和客户端之间仍是 `(H,7)`（论文默认 `H=50`）的
`[local dx,dy,dz,droll,dpitch,dyaw,next_gripper]`；`robokit/deploy/rtc.py` 仍逐行进入
`ChunkExecutor._step_eef`，继续使用 `apply_local_delta_pose`、到位等待、host IK、安全闸、
夹爪模式和 trace。RTC 不会把它变成 joint action 或 absolute EEF pose。

上游依据：
[openpi](https://github.com/Physical-Intelligence/openpi)、
[LeRobot PI05](https://huggingface.co/docs/lerobot/main/pi05)、
[RTC 论文](https://arxiv.org/abs/2506.07339)、
[LeRobot RTC 实现（固定 revision）](https://github.com/huggingface/lerobot/tree/f37be3edbee60f3a09a5183788b91eb19f0c07d1/src/lerobot/policies/rtc)、
[PI05 HF 底座](https://huggingface.co/lerobot/pi05_base)、
[LeRobot PEFT](https://huggingface.co/docs/lerobot/peft_training)。
本目录固定 LeRobot revision `f37be3edbee60f3a09a5183788b91eb19f0c07d1`；升级时必须重新
跑转换验证、processor 加载和真机 dry-run。

---

## 关节限位

全仓唯一权威表：[`robokit/arms/piper_limits.py`](robokit/arms/piper_limits.py)。
取松灵官方 URDF 与本机主控 Flash 实读的**并集**（逐轴取宽）：

```
j1 ±150    j2 [0,180]    j3 [-170,0]    j4 ±100    j5 ±70    j6 ±180
```

其余模块和脚本一律 import 这一份。换机械臂或改固件参数后必须重查 Flash
（`SearchAllMotorMaxAngleSpd`）。

> j6 曾经在四处分别写成 ±120 / ±170 / ±180，同一个目标在一处放行、另一处被夹住。

反馈越界的判定容差是 0.3°（`_SEED_LIMIT_TOLERANCE_DEG`），不是 JointCtrl 的指令分辨率
0.001°：伺服长期停在限位上时编码器会读出略微越界的值（示教数据自己就有 14 帧 j5=70.01），
按 0.001° 判会把正常硬件行为当成越界而中止。0.3° 同时仍能拒绝真正需要复位的下垂场景
（实测越界 1.2~1.9°）。

---

## 目录

```
configs/          YAML：机器人组成 + 采集 + 部署参数
robokit/
  arms/           base(任意DOF) / piper / piper_ik(主机IK) / piper_limits(唯一限位表)
                  piper_lifecycle(reset guard) / mock
  cameras/        base / realsense(直连) / ros2_camera / mock
  policies/       policy 注册表：dummy(联调) / memvla_lora(MemoryVLA 适配器)
  robot.py        由配置装配 N 臂 + M 相机
  recorder.py     HDF5 增量录制（后台写线程）
  episode.py      HDF5 读取辅助
  pose.py         局部 delta 位姿数学（采集/转换/部署共用同一份）
  executor.py     action chunk 执行
  safety.py       下发前的动作安全闸
  comm.py         BiSocket TCP（纯标准库）
  trace.py        真机部署的结构化 JSONL trace
scripts/          两个功能的入口 + publish_batch.sh(分批发布) + 5 个单测，见上文
rlds/             RLDS 转换
pi0/              PI0 绝对关节角（预测 50、执行 H=30）训练 / TCP 推理服务
pi0.5/            PI0/PI05 共用 LeRobot 转换工具 + PI05 EEF 训练 / 推理服务
datasets/_batches/<仓库>/<任务slug>/<批次>/hdf5/
                    已发布批次的 HDF5（从任务目录 mv 过来，全局唯一一份）
                    同级 <批次>.json 是 manifest 台账（不进 git；数据目录核对后可删）
```

逐文件的作用与依赖关系见 [PROJECT_MEMORY.md](PROJECT_MEMORY.md) 的「代码地图」。
改代码前后跑一次回归（纯软件，约 0.2 秒）：

```bash
python -m unittest discover -s scripts -p "test_*.py"   # 期望 Ran 37 tests OK
```

**扩展**：新机械臂继承 `arms/base.py` 的 `Arm`（connect/get_state/move_joint），
在 `arms/__init__.py` 注册类型名即可。加减相机/机械臂只改 YAML，全链路自适应。

---

## 真机注意事项

**停止 policy**：`reset_on_disconnect: true` 时任何退出都会执行 controller reset，
而 reset 让六轴**瞬间失力**。`pkill` 强停会让臂下垂到 j2<0 / j3>0 越界。正确做法是跑完
`--max-steps`，或用 `--park-on-exit`（默认开，退出前先归位到开跑时的关节位姿）。
`--safety off`（默认）会把 `reset_on_disconnect` 改成 false，Ctrl-C 不失力，臂带力
留在 policy 的终止位姿上。

**越界下垂恢复**：`reset_piper_to_demo_start.py --execute --assume-safe --skip-controller-reset`。
移动按 `--step-deg`（默认 15°）分段并逐段查固件状态/驱动使能，任意跨度都不会变成一次
无监控的大扫；`--max-start-gap-deg 0` 解除跨度拒绝。

**断电搁置的 Piper 会瘫到软限位外**（j2<0、j3>0）。`PiperArm.connect()` 会自动低速挪回
限位内（越界 >15° 报错要求人工处理）。

**candleLight USB-CAN 适配器偶发接收挂死**（已见过 `error -71` 和
`failed to re-submit IN URB: -EPERM`）。若 `can0` 虽为 `UP / ERROR-ACTIVE / 1Mbps`，但
`ip -details -statistics link show can0` 的 RX 始终为 0，且 `candump can0` 完全无帧，先确认
机械臂已上电、急停已释放、CAN H/L 与终端接头可靠；再物理重插 USB-CAN，然后分三步重建接口：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000
sudo ip link set can0 up
timeout 3 candump can0
```

健康的 Piper 总线会持续输出反馈帧；如果重建后仍是 RX=0 且总线错误也为 0，说明机械臂侧
没有任何电气流量，继续查电源/急停/线缆而不是改 SDK。采集启动若在相机连接后遇到这类
CAN 失败，会事务性关闭 Piper SDK 与 RealSense 线程并以普通异常退出，不应再出现
`terminate called without an active exception` 或 core dump。
