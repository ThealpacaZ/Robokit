# 任务状态

> 最后更新 2026-08-03。只记当前有效的结论和下一步；已验证的事实进 README，
> 操作手册进 README。历史诊断过程不留在这里。

---

## π0.5 全量 JOINT：201 段 Stack-one-cup 训练中（2026-08-05 启动）

- **与 EEF 轮次同一批数据、同一套超参**，只换动作空间和 prompt 后缀：
  `--action-space joint`（state = 当前 [j1..j6, gripper]，action = **下一帧绝对关节角，弧度**），
  后缀改为 **` <control mode> joint <control mode>`**（同样带空格，与 eef 轮次格式一致）。
- **`run_stack_one_cup_201_server.sh` 已参数化**：`ACTION_SPACE=eef_delta|joint` 一处决定
  CONTROL_MODE、prompt 后缀、数据集/模型仓库名、输出目录、W&B 项目名，避免两套近乎重复的脚本
  各自漂移。不传参时行为与 eef 轮次逐字节相同。
- **STEPS=30000**（不是 60000）：LR schedule 的 `decay_steps` 本来就是 30000，所以前 30k 步与
  eef 轮次配置完全一致；而 eef 轮次证明 eval_loss 从 5k 起就单调上升，跑到 60k 只是多烧 24 小时。
- **先试快的（compile 开，1.95 s/step），崩了自动回退慢的（2.86 s/step）并从最近 checkpoint 续训**，
  不是从头再来。`EVAL_STEPS=SAVE_FREQ=2500`，所以 compile 万一在 eval 崩，最多损失 2500 步
  （约 81 分钟）而不是 eef 轮次那次的 5000 步全丢。
- **关于 compile 崩溃点的推断**：崩溃 kernel 带动态符号 `s19`。eval 集约 3,879 帧 ÷ batch 32
  = 121 整批 + **1 个 7 帧残批**，残批形状触发 inductor 重编译——这很可能就是触发点。
  旧 smoke 用 `MAX_EVAL_SAMPLES=64`（正好被 32 整除）**永远不产生残批，所以复现不了**。
  现在 smoke 固定用 `2*batch+5` 强制产生残批。这是推断不是定论，但成本极低。
- **smoke 现在区分 OOM 与 compile 崩**：只有 OOM 才该减半 batch；inductor 故障减 batch 毫无用处，
  只会白跑四轮 smoke 再在 batch=1 失败。命中 `illegal memory access|No valid triton configs|
  _inductor|cudagraph` 时直接以 rc=3 退出，交给上层回退。
- **`MIN_STEPS=999999` 是故意的**：收敛路径自带 `shutdown -h now`，会在两份权重都发布之前把机器关掉。
  让它必定以 NOT_CONVERGED 结束，把关机决定权交给 `publish_run_and_shutdown.sh`。
  监控器的「最优权重硬链接到 FINAL_DIR」与收敛判定无关，照常生效。
- **发布顺序按优先级**：先发 **step 30000**，再发**最优 eval_loss**——中途失败也能保住高优先级那份。
  两份都回读校验通过才关机；任一失败则保留实例。
- 关节契约已 dry-run 验证：单帧最大关节变化 **0.0742 rad（4.25°）**，与之前 π0 关节轮次同量级。
- 全链路 `/root/chain_joint.sh` 常驻服务器（下载 → 暂存 → 转换 → 训练 → 发布 → 关机），
  PPID=1、无控制终端，已实测断连后存活。
- **eef 两份权重在删本地副本前已再次回读校验**，仍在 HF 上：
  `shaohuan1/stack_one_cup_201_eef_pi05_full`（best）与 `..._step30000`。

---

## π0.5 全量 EEF：201 段 Stack-one-cup 已结束并发布（2026-08-05 收尾）

**结论：这批数据量撑不住 3.6 B 全参数微调，最优权重出现在 step 5000，之后一路过拟合。**

| step | train loss | eval_loss | 说明 |
|---|---|---|---|
| 5,000 | 0.055 | **0.2071** | **最优**，已发布 |
| 10,000 | 0.040 | 0.2782 | |
| 15,000 | 0.031 | 0.3649 | |
| 20,000 | 0.026 | 0.4301 | |
| 25,000 | 0.030 | 0.5135 | |
| 30,000 | 0.022 | 0.5508 | 训练在此手动停止，已发布存档 |

- eval_loss **连续 6 次单调上升**（0.2071→0.5508，2.7×）、train/eval 差距从 3.8× 拉到 25×，
  是确凿过拟合，不是噪声。
  到 step 30,000 已经是 **29 个 epoch**（只有 180 段训练集 / 33.5k 帧）。
- **loss 的刻度**（`u_t = noise − action`，`loss = MSE(u_t, v_t)`，ACTION 用 QUANTILES 归一化）：
  完美预测 = 0；**「什么都没学到」的常数预测 = 1 + Var(a_norm) = 1.238**（由 37,204 行 action
  实算）。所以 0.207 ≈ 解释掉 83% 目标方差，0.514 ≈ 58%——还有信号，但泛化在持续退化。
- **发布物（均已逐文件字节回读校验，private）**：
  - `shaohuan1/stack_one_cup_201_eef_pi05_full` = **step 5000 最优权重**（部署用这个）
  - `shaohuan1/stack_one_cup_201_eef_pi05_full_step30000` = step 30000 权重（过拟合，仅存档）
- W&B：`https://wandb.ai/yangshaohuan720-university/stack_one_cup_201_pi05_full/runs/mihn111p`
- 监控器的「最优权重保全」机制是有效的：janitor 按 max-retained=1 删掉了 5k/10k/15k/20k 的
  训练副本，但监控器在每次 eval 后已把当时最优的 `pretrained_model` 硬链接到 `models/` 下，
  所以 step 5000 没丢。**这条设计不要拆。**
- **下次的判据**：180 段的 eval_loss 拐点在 **5 个 epoch 以内**。1500 段时一个 epoch ≈ 8,700 步
  （batch 32），同样 5 epoch ≈ 4 万步，所以 1500 段做 3w–6w 步全量微调是合理的。
  但**看 eval_loss 拐点、不要看固定步数**，并且位姿/光照多样性比条数更关键
  （已知新场景位姿 97% 落在旧数据分布外）。

---

## π0.5 全量 EEF：201 段 Stack-one-cup 训练运行中（2026-08-04 01:55 启动）

- **数据源纠正**：不是 `shaohuan1/lememory`（那是 401 段 "stack cups"，5 个 batch，
  320×240）。正确的是 **`shaohuan1/Memoryvla` 的 `Stack_one_cup_on_top_of_another_cup_b0/hdf5`**，
  201 段（index 0..200 连续）、34.49 GB、**640×480**、30 Hz、`task_name` 就是
  `Stack one cup on top of another cup`。同仓库另有 `Cover_the_building_block...b0`（157 段）
  和 `stack_cups`（无 hdf5），别混。
- **契约**：单 `cam_high`、`right_arm`、7 维 local `eef_delta`（局部 xyz/RPY delta +
  下一帧 gripper）。转换保留原 task prompt，仅追加 **` <control mode> eef <control mode>`**
  （**前导空格 + 标签内外都带空格**，与 200 段那轮一致；用户 2026-08-04 明确确认沿用此格式）。
  转换后从 `meta/tasks.parquet` 逐条硬校验，唯一 prompt 为
  `'Stack one cup on top of another cup <control mode> eef <control mode>'`。
- **数据集**：`shaohuan1/stack_one_cup_201_eef_pi05_lerobot`，**只在本地**
  `/root/autodl-tmp/hf/lerobot/` 下，未推 Hub（上行只有 2.5 MB/s，推一次要半小时，
  不值得堵住开训）。已实测 LeRobot 能以纯本地方式加载：201 段 / 37,204 帧 / 30 fps，
  切分 180 train / 21 eval。原始 34.5 GB HDF5 已在校验后删除。
- **⚠ torch.compile 在这块 sm_120 上会在第一次 eval 崩掉 —— 必须
  `compile_model=false`。** 第一轮（01:55 起）干净跑完 5,000 步（loss 0.465→0.059），
  走到 step 5000 的首次 eval 时 inductor 为 eval 形状重编译（日志 `[3/1]`），Triton 自动调优
  超出本卡共享内存上限（`Required: 196608 Hardware limit: 101376`），回退后生成的融合 kernel
  直接 `CUDA driver error: an illegal memory access was encountered`，进程退出。
  **不是显存不足**（当时只用 33/85 GB）。因为崩在首次 save 之前，`checkpoints/` 是空的，
  2 小时 48 分全部丢失。
- **smoke 必须跑到 eval + save，只跑训练步是拦不住的。** 原来的 3 步 smoke（`EVAL_STEPS=103`）
  永远走不到 eval，所以放过了上面那个必崩项。现改为 `STEPS=4 / EVAL_STEPS=2 / SAVE_FREQ=2 /
  SAVE_CHECKPOINT=true`，并顺带量出 checkpoint 真实体积。
- **训练**：单卡 RTX 6000D 全参数微调 `lerobot/pi05_base`（3.617 B 参数，fp32 底座）。
  BF16 + gradient checkpointing，**compile 关闭**，vision encoder 不冻结，
  `train_expert_only=false`。**batch=32 smoke 通过**（含 eval 与 checkpoint 落盘）。
  稳态 **2.86 s/step**、`mem_gb` 38.7、GPU 100%、42.5 GB、345 W。
  关 compile 的代价：1.95 → 2.86 s/step（慢 46%），30k 步约 **23.8 h**，60k 约 47.7 h。
  `data_s` 稳态 0.006（step 50 的 0.614 只是 dataloader 预热，不是瓶颈，不用加 num_workers）。
  1047 步/epoch。
- **checkpoint 实测 23 GB/份**（bf16 模型 7.2 + 优化器状态 14.5）。janitor 只留 1 份，
  保存瞬间两份并存 46 GB，73 GB 空闲够用，不需要清理 OpenVLA 旧产物。
- **进度基线**：step 100 loss 0.370（50→0.465），与第一轮逐值一致（同 seed）。
- **W&B（online 实时）**：
  `https://wandb.ai/yangshaohuan720-university/stack_one_cup_201_pi05_full/runs/mihn111p`
  （第一轮崩掉的是 `runs/ccwdfdiq`，别看错）
- **收敛与关机**：`eval_steps` 与 `save_freq` **都是 5000 且必须相等**——监控器按「同一 step 的
  eval_loss ↔ checkpoint」配对挑最优权重，不等就选不出来。至少 30,000 步 + 至少 6 次 eval 后
  才允许判收敛；连续 3 次 eval 无 ≥0.5% 改善才终止。收敛 → 回读校验 → 发布
  `shaohuan1/stack_one_cup_201_eef_pi05_full` → `shutdown -h now`。
  OOM / 异常 / 60k 未收敛都保留实例。checkpoint janitor 只留 1 份（单份约 22–43 GB，
  73 GB 空闲，保存瞬间会有两份并存）。
- **脱离本机**：整条链路 `setsid` 起，根进程 PPID=1、TTY 全为 `?`。已实测杀掉 SSH 主连接后
  supervisor / trainer / janitor / monitor 全部存活——**本机断电不影响服务器训练**。
- 修掉了 `pi0.5/train_full.sh` 的真 bug：`--multi_gpu` 是写死的，而 accelerate 拒绝
  `num_processes=1`，单卡路径必然启动失败（之前那轮是双卡所以没暴露）。现改为仅在
  `NUM_PROCESSES > 1` 时才传。

---

## π0.5 全量 EEF：200 段 LeMemory 训练已启动（2026-08-03）

- 新 AutoDL 实例 `autodl-container-b9j8ek84q3-6f705e7b` 有一张空闲 RTX 6000D（85.7 GiB）。训练环境已按锁定的官方 LeRobot `f37be3edbee60f3a09a5183788b91eb19f0c07d1` 安装并自检为 `lerobot=0.6.1`、`torch=2.11.0+cu128`、CUDA 可用。
- 最初误把 b0–b4 合并为 401 段；用户纠正为 200 条后，401 段 supervisor 已停止，且没有启动 GPU trainer。数据集只 stage 连续 episode `0..199`（精确 200 段）。因父 supervisor 被停止，现由低频恢复 wrapper PID `183249` 每 300 秒检查一次基座下载进程；日志为 `/root/autodl-tmp/outputs/stackcups_200_eef_pi05_full_ctrl_eef.resume.log`。
- 动作契约固定为 7-D local `eef_delta`（局部 xyz/RPY delta + 下一帧 gripper）。转换保留每条既有 task prompt，并仅追加 ` <control mode> eef <control mode>`；训练前读取 `meta/tasks.parquet` 验证每条 prompt 均以该后缀结尾。
- 全量训练目标为 `shaohuan1/stackcups_200_eef_pi05_full_ctrl_eef`，输出目录 `/root/autodl-tmp/outputs/stackcups_200_eef_pi05_full_ctrl_eef`；单卡全参数配置为 BF16、gradient checkpointing、compile、vision encoder 不冻结、`train_expert_only=false`，batch=16，先做 3-step full smoke，再训练至最多 60,000 步。
- 收敛监控只会在至少 30,000 steps、至少 6 次 eval 后生效：连续 3 次 eval 无 >=0.5% 改善，且最近 3 次无有效下降才终止。收敛时保留/发布最佳完整权重、回读校验后执行 `shutdown -h now`；OOM、训练异常或 60k 未收敛都会保留实例。
- 200 段编码、私有数据集上传及 prompt 校验已完成：200 episodes、44,885 frames，唯一 task prompt 的后缀验证通过。GPU 暂时 0% 的唯一剩余原因是 `lerobot/pi05_base` 的 14,467,165,872-byte 权重下载；旧 5,609,881,600-byte incomplete 文件已废弃，当前 prefetch PID `182654` 正在新 incomplete 文件上续传（最近观测为 2,327,838,720 bytes）。下载结束后 wrapper 会重新运行正确的 200 段 supervisor；本地数据集存在时它跳过原始 HDF5 重下载，避免恢复时退回 401 段。为避免 30GB 容器根盘因保留 19GB 原始 HDF5 Hub 缓存而无法容纳基座，supervisor 仅在新 LeRobot 数据集已上传并验证后删除该可从 `shaohuan1/lememory` 重下的本地源缓存；输出/checkpoint 数据盘不删除。

---

## OpenVLA-OFT LoRA：StackCups step 25,000 真机推理服务运行中（2026-08-02）

- 原训练子进程在 **step 16,626** 附近收到外部 `SIGKILL`；W&B 最后成功记录为
  **step 16,610**、train loss `0.1044921875`、LR `5e-4`。主机/容器 OOM 计数均为 0，
  且远未达到 80k 的最早收敛判据，因此不是收敛监控器主动停止。
- step 15,000 的 latest-only checkpoint 已回读验证：LoRA adapter 879 个张量、action head
  16 个张量，恢复前 SHA-256 清单保存在 run 目录的 `resume_from_15000.sha256`。
  上游 resume 原本既不识别 `--latest_checkpoint.pt`，也不加载已有 LoRA；已在远端
  `vla-scripts/finetune.py` 补齐，并保持 LR 在全局 step 75,000 衰减。
- 训练从全局 **step 15,000** 恢复后跑到 **step 25,000**；确认 checkpoint 保存函数返回并
  继续到 25,008 后，按用户要求冻结部署副本并停止 trainer/monitor。该权重没有达到 80k
  收敛判据，属于早期真机候选，不得标记为 converged。
- 冻结权重位于
  `/root/autodl-tmp/openvla_oft_deploy/stackcups-step-025000`，LoRA adapter 与 action head
  均已回读。推理服务 PID `37102`，远端监听 `8080`，约占 15.4 GiB 显存；本机 SSH 隧道
  监听 `127.0.0.1:8080`，PID 文件为
  `runs/deploy-sync/openvla-oft-step-025000.tunnel.pid`。
- 真机契约已从服务启动日志核对为 **PIPER constants**：单 `cam_high`、chunk=30、action=7 维
  local EEF delta + continuous gripper、`bounds_q99`。第一次启动曾命中上游默认 LIBERO
  chunk=8，未发送任何推理请求即停止；现通过 README 命令中的
  `--policy-arg robot_platform=piper` 固定为正确契约。
- SSH 连接问题来自 Mihomo 路径：普通进程直连被本地网络过滤，代理节点不能完成 SSH banner；
  将 Mihomo `GLOBAL` 选择器切到其内部 `DIRECT` 后，SSH 可正常连接。

- 数据固定为 HF dataset `shaohuan1/lememory` 提交
  `666ef5521cb3daf42d2f074bf4160cdaf92781bf`，使用 `stack_cups_b0`–`b4`
  的全部 **401 episodes**；5 个 RLDS 批次已合并并回读验证为 401 段。
- 训练栈为官方 `moojink/openvla-oft` 提交
  `e4287e94541f459edc4feabc4e181f537cd569a8`，底座 `openvla/openvla-7b`
  三个分片均按 HF 模型 API 的真实 LFS SHA-256 校验通过。
- 契约：单 `cam_high` 图像、7 维 local EEF delta + continuous gripper、action chunk=30、
  L1 action head、LoRA r32、image augmentation、无 proprio/FiLM。
- 服务器只有 1 张 RTX 6000D 85 GB。batch=24 实测 CUDA OOM；batch=16 通过并作为
  最大稳定批量，训练时约 76.8/85.7 GB、GPU 100%。
- W&B：
  `https://wandb.ai/yangshaohuan720-university/stackcups_openvla_oft_lora/runs/stackcups401oft20260801`。
  训练 PID 文件、日志和 checkpoint 根目录均在
  `/root/autodl-tmp/openvla_oft_runs/stackcups_401_openvla_oft_lora*`。
- 每 5000 step 覆盖保存一次完整 LoRA adapter + action head；上限 150,000 step，75,000
  step 衰减学习率。因为用户要求全部数据参与训练，没有 held-out split；收敛定义为 80k
  之后最近 2000 step 的训练 L1 median ≤0.01，且连续 3 个 checkpoint 没有 ≥0.5% 的
  有效改善。守护每 1800 秒查一次 W&B。
- 唯一自动关机成功路径：收敛证据成立 → 最新完整 checkpoint 稳定落盘 → 发布私有 HF
  模型 `shaohuan1/stackcups_openvla_oft_lora` → 回读确认 adapter/action head/config/证据齐全
  → `shutdown -h now`。训练异常、监控/W&B/HF 失败或 150k 仍未收敛都留机排查。

---

## π0 关节角：训练 50/30，当前运行 chunk=15、30Hz（2026-07-29）

用户确认模型是 LeRobot **π0**，不是 MemoryVLA/PI0.5；训练预算与前一套 LeRobot EEF
配置相同，固定 **30,000 step**。π0 的预测长度与执行长度必须分开：
`chunk_size=50`、`n_action_steps=30`。其他训练配置为
PI0 base + LoRA r32/alpha32、batch=8、每 1000 step 保存/评估。
以上 50/30 是已经完成的 checkpoint 训练契约；当前运行时后来改为只返回/执行前 15 步，
不修改 checkpoint、不重训。

动作契约已逐值核验：state=`当前 [j1..j6, gripper]`，action=
`下一帧绝对 [j1..j6, gripper]`，六轴为真弧度，不做 relative-action 二次变换。独立
LeRobot 数据仓库不会覆盖原 PI0.5 EEF 数据或统计。202 段 dry-run 得到 45,294 个训练帧，
最大单帧关节变化 0.09821 rad（5.63°）。

后台流水线已完成 LeRobot MP4+Parquet 转换/发布、`lerobot/pi0_base` 下载和训练：

| 项 | 路径 / 值 |
|---|---|
| completion marker | `/root/autodl-tmp/outputs/stackcups_joint_pi0_h30/COMPLETED` |
| supervisor log | `/root/autodl-tmp/outputs/stackcups_joint_pi0_h30.supervisor.log` |
| train log | `/root/autodl-tmp/outputs/stackcups_joint_pi0_h30.train.log` |
| LeRobot dataset | `shaohuan1/stackcups_joint_pi0_lerobot` |
| run / checkpoints | `/root/autodl-tmp/outputs/stackcups_joint_pi0_h30` |
| final checkpoint | `/root/autodl-tmp/outputs/stackcups_joint_pi0_h30/checkpoints/030000/pretrained_model` |
| published adapter | `shaohuan1/stackcups_joint_pi0_h30` |
| max steps | 30,000 |
| checkpoint prediction / train action steps | 50 / 30 |
| current runtime chunk / rate | 15 / locked 30Hz |
| publish | 最后一个 `030000/pretrained_model`，不按 eval 选 best |

`lerobot/pi0_base` 固定 revision `25c379b52ba2ff8788cab921758a3cc3fe3f77f2`，
14,005,618,584-byte 权重的 SHA-256
`8229fd9a7c3c2aafc1e223567b61b5fe3e25eef873bb4233928dbee4bd836303`
已校验。误配的 `n_action_steps=39 / steps=10000` run 在 step 1600 前停止并完整归档到
`/root/autodl-tmp/outputs/failed/stackcups_joint_pi0_wrong-h39-10k-20260729`，不作为候选权重。
正确 trainer 启动时的 PID 为 615676，实际命令行已回读确认
`chunk_size=50`、`n_action_steps=30`、`steps=30000`、W&B online。曲线：
`https://wandb.ai/yangshaohuan720-university/stackcups_joint_pi0_h30/runs/i8tmjy90`。
训练现已跑满 30,000 step；`COMPLETED` 明确记录 `status=completed`、
`checkpoint=.../030000`、`steps=30000`、`chunk_size=50`、`execution_horizon=30`、
`action_space=joint` 和发布仓库。最终 checkpoint 回读为 `type=pi0`、
`use_relative_actions=false`。按用户要求没有做高频轮询，训练过程以在线 W&B 曲线和
完成标记为准。

执行配置为 `configs/piper_single_joint.yaml`：从 YAML 自动选择 `action_space: joint`，
`horizon: 15`、`control_freq: 30`、`control_freq_locked: true`，服务端口为 8081。
服务端会在回包声明动作空间，客户端发现 EEF/joint 误连会在执行器前硬失败；客户端若
尝试用 `--control-freq` 覆盖为其他频率会直接报错。
远端关节服务 PID 586280，启动参数含 `--horizon 15`，日志
`/root/autodl-tmp/outputs/eval-services/pi0-joint-h30.log`；本机
`127.0.0.1:8081` SSH 隧道 PID 61995。H=30 阶段的历史测试曾确认 π0 flow 采样存在
随机越界；用户随后明确决定 joint 专用配置关闭 ActionGuard，并将有限 6-D 关节目标
逐轴钳到最近机械限位执行，详细当前契约见下方。
当前 8080 继续保留给 step-12000 PI0.5 EEF 同步服务。

### 现场测试回读（2026-07-29 06:52）

用户测试产生的 trace 已定向核验：

- `runs/deploy/20260729-065225.jsonl` 是 **π0 关节角 dry-run**：1 次推理，
  `chunk_shape=(30,7)`，延迟 395.5ms，30/30 行执行器检查完成，0 abort、0 CAN 命令。
  六轴范围分别为 j1 `[-83.084,-82.573]°`、j2 `[0.748,35.434]°`、
  j3 `[-44.680,-0.151]°`、j4 `[-1.270,-0.233]°`、j5 `[18.670,45.525]°`、
  j6 `[4.520,11.863]°`，全部在配置硬限位内；夹爪 `[0.1875,0.2043]`。
- 随后的 `runs/deploy/20260729-065250.jsonl` 是 **旧 EEF 真机路径**，不是 joint：
  trace 含 `base/delta/target` 和 `pinocchio_ik_move_j` 实际命令。共 30 次推理、
  896 条 CAN 关节目标，0 abort、0 violation，用户停止后已无 deploy_client 进程。
  该次使用了 best-effort/no-guard 语义：165 条 saturated，最大理想目标残差
  **9.395mm / 13.949°**，最大跟踪距离 17.099mm、最大关节步 5.000°。所以 0 abort
  只说明 no-guard 允许最近可行解继续发送，不代表每个 realized EEF 都忠实于 policy
  的理想目标；trace 本身也不能判断杯子任务是否成功。

当前结论：关节权重已经通过零运动接口/限位检查，但**尚未做 joint 真机下发**。若下一步
要评估关节权重，应继续使用 `configs/piper_single_joint.yaml`，不要把旧 EEF/no-guard
结果混进 joint 权重结论。

### π0 joint 首行限位中止（2026-07-29 06:58）

`runs/deploy/pi0-joint-dry-run.jsonl` 精确记录本次问题：服务返回正确的
`joint/(30,7)`，但第 0 行为 j2=-1.189°、j3=0.911°，超出配置硬限位
j2 `[0,180]°`、j3 `[-170,0]°`。ActionGuard 在 `move_joint`/CAN 之前以
`joint_limit` 中止，0 条运动命令；这不是 IK 求解失败。

joint 执行链为 `ChunkExecutor._step_joint → PiperArm.move_joint → JointCtrl`，模型给出的
绝对关节角不会进入 EEF 累加、host IK 或 Pinocchio IK。根因是 π0 flow 随机采样在训练
数据的 0°边界附近产生了小幅外插；此前同一观测已有合法和非法两类随机结果。当时契约
是 fail-closed，不能把问题误报为 IK 故障。

配置中原来的 `eef_backend: host_ik` 虽不参与 joint 动作，却会在 PiperArm 可写连接时被
构造。按用户要求，`configs/piper_single_joint.yaml` 已明确改为
`eef_backend: pinocchio_ik`、`allow_best_effort: false`；这消除备用 EEF 后端歧义，
但不会改变模型 joint 输出。

### joint 解除 ActionGuard + 限位钳制（2026-07-29）

用户随后明确要求：解除这套 joint 执行配置的动作安全护栏，超过机械限位的轴按最近限位
执行。当前实现：

- `configs/piper_single_joint.yaml` 固定 `deploy.safety.enabled=false`、
  `gripper_rate=0`、`joint_limit_mode=clip`；运行时无需额外传 `--no-guard`。
- 只对 `PiperArm.move_joint` 的有限 shape=(6,) 直接关节目标逐轴 `np.clip` 到
  `joint_limits_deg`，再按 0.001° 量化并复验精确 CAN 目标。EEF/Pinocchio IK 路径仍为
  strict reject，不共享该 clip 开关。
- trace 的 `arm_command` 记录原始 `joint_requested_deg`、是否钳制、钳制轴、
  `joint_limit_clip_delta_deg` 和最终 `joint_target_deg`；每次钳制同时打印 WARNING。
- dry-run 调用同一套纯软件预处理并生成上述元数据，但不切运动模式、不写夹爪、不发 CAN。
- 不解除基础完整性与硬件故障：错误维度、NaN/Inf、反馈陈旧、CAN/驱动器/控制器异常仍中止。
  相机运行时检查也保持。

边界测试覆盖六轴同时越界 `[-160,-1.2,0.9,110,-80,190]°` 精确变为
`[-150,0,0,100,-70,180]°`、默认 reject 配置仍拒绝、clip 模式仍拒绝 NaN、dry-run
preview 为 0 CAN。当前 scripts 单测 43/43、π0 契约 4/4、PI0.5 契约 8/8 通过。

端到端零运动验证 `runs/deploy/pi0-joint-clip-dry-run.jsonl`：
8081 推理 1 次、416ms、`(30,7)`，30/30 行完成、0 abort；20 行钳制了 j2/j3。
模型原始 j2 最低 -1.896°、j3 最高 1.709°，最终命令预览 j2 最低 0°、j3 最高 0°，
六轴最终值全部在机械限位内。客户端以 read-only 连接 Piper，0 CAN 运动帧。

### 当前运行时 chunk=15 + 永久锁定30Hz（2026-07-29）

用户要求把运行 chunk 改为15并永久固定30Hz。训练 checkpoint 保持
`chunk_size=50 / n_action_steps=30`，不重训；`Pi0LeRobotPolicy` 现在只返回前15行，
8081 服务已用 `--horizon 15` 重启为 PID 586280。`piper_single_joint.yaml` 固定
`action_horizon: 15`、`control_freq: 30`、`control_freq_locked: true`；非30Hz的
命令行覆盖会在连接相机/机械臂之前直接失败。

锁定模式使用绝对 deadline 调度，扣除每条 joint 预处理/钳制耗时，过期时不突发补发。
端到端 read-only trace `runs/deploy/pi0-joint-h15-30hz-fixed-dry-run.jsonl`：
服务回包 `(15,7)`、15行、0 abort、0 CAN，15行钳制后全部合法；平均行间隔
33.395ms，即29.944Hz。旧 EEF 非锁定配置保持原调度，不受影响。

## 旧 EEF 权重服务已恢复（2026-07-29）

远端 8080 正在运行已审查的 PI0.5 step 12000，同步模式、EEF delta、H=30；本机
`127.0.0.1:8080` 隧道已用 `92.hdf5` 离线帧实测回包 `(30,7)`、全有限，
`server_mode=sync`、`action_space=eef_delta`。该健康检查没有连接 CAN。

## 当前状态：policy 控制 EEF 已跑通

真机 573 步连续执行，host_ik 零拒绝、guard 零拦截，位姿残差 **0.002mm / 0.001°**。
执行链路 `EEF 目标 → 主机连续有界 IK → MOVE_J/JointCtrl` 端到端成立。

配套证据：

| 项 | 结果 |
|---|---|
| 主机 IK 全量离线扫描 | 74 段 **15447/15447** 通过，最小关节限位余量 0.000° |
| 关节回放真机 | ep0 **314/314** 帧，speed=100，末段跟踪误差 0.06° |
| 单测 | 当前 41 个全过（`python -m unittest discover -s scripts -p "test_*.py"`） |

## Piper IK 社区核验与全量边界（2026-07-29）

“官方 IK 有问题”需要拆成两件事。`piper_sdk.EndPoseCtrl` 自己不做 IK，只发末端
CAN 目标；主控 `MOVE_P` 使用解析逆解。松灵维护者在
[piper_sdk #96](https://github.com/agilexrobotics/piper_sdk/issues/96) 明确确认：
邻近目标在多解点可能解析到不同关节分支，产生角度突变；该模式不会缓存上一解析结果，
并建议使用 `piper_ros` noetic 的 Pinocchio 主机逆解。另有
[末端位姿推理偶发乱扭](https://github.com/agilexrobotics/piper_sdk/issues/111)
和 [RPY 数值突变](https://github.com/agilexrobotics/piper_sdk/issues/106) 的独立反馈。
后者是欧拉表示不连续，不能和真实姿态/关节跳支混为一谈；全链仍按外旋 `xyz` 与
SO(3) 误差处理。

`piper_ros` 的 Pinocchio + CasADi/IPOPT 示例可作为建模来源，不能原样用于 PI0.5
30 Hz 真机闭环：平滑项被注释，目标函数正则到零位而非上一解；大于 30° 的候选仍先
返回，只把下一次 seed 清零；没有量化后 FK 残差验收；其 RPY 轴约定还曾由
`rzyx` 修为 `sxyz`（[PR #30](https://github.com/agilexrobotics/piper_ros/pull/30)）。
此前 `PiperContinuousIK` 用 `piper_sdk` FK + SciPy 有界最小二乘，保留了生产所需的
上一条已接受 seed、±5° 局部硬边界、0.03 连续性、SO(3) 残差、0.001° 量化后复验、
失败不提交 seed/保持上一目标，因此比直接复制 demo 更适合本仓库。

202 段 / 45,294 动作的 `recursive, H=30` 生产路径复扫结果：

| 配置 | 结果 |
|---|---|
| 当前 1mm / 1° 阈值 | 201/202 段完整通过；45,107 条已下发；唯一 `128.hdf5` 在第 10 条之后以 0.006mm / 1.137° fail-closed，下一条 CAN 未发送 |
| 只把旋转阈值临时改为 1.5° | `128.hdf5` 197/197 通过；最大 0.078mm / 1.137°，最大关节步 2.926° |

这组结果现作为旧 `host_ik` A/B 基线保留；默认生产后端已换成下面的真正 Pinocchio IK。

## 真正 Pinocchio IK 后端与全量验收（2026-07-29）

新增 `robokit/arms/piper_pinocchio_ik.py` 与版本化的纯运动学 URDF
`robokit/assets/piper_description.urdf`，配置名是显式的
`eef_backend: pinocchio_ik`。它实际导入 Pinocchio 3.6.0，绝不把该配置转给旧
`PiperContinuousIK`；旧 `host_ik` 只保留作显式 A/B。

模型契约在构造时 fail-closed：

- `nq=nv=6`，活动关节顺序必须严格为 `joint1..joint6`，末端 frame 必须是 `link6`；
- URDF 限位必须与当前 Flash 实读 `joint_limits_deg` 完全相同；
- 四组关节样本与 SDK FK 对拍，最坏差 **0.1268mm / 0.0037°**；
- 实际求解、LOCAL Jacobian、`log6(T_current⁻¹T_target)` 残差和 realized FK 全部来自
  Pinocchio；SDK FK 只参与启动时模型核验。

每条 `move_eef` 都读取最新物理关节反馈。反馈相对机械限位逐轴超限 `≤10°` 时只把
seed 投影到最近限位，`>10°` abort；这不是 policy 笛卡尔目标或无约束理论解。
优化变量直接受“六轴机械限位 ∩ feedback seed±5°”约束，并带 0.03 连续性代价，
不是求完再裁单轴。JointCtrl 0.001° 量化后再次验限，再用 Pinocchio FK 生成
`eef_target_realized` 供 arrival 使用。

no-guard 配置的 `allow_best_effort=true` 把 pose residual、限位饱和和 max iterations
只作为 trace 元数据；有限、限位内、单步 `≤5.001°` 的 best candidate 仍可发送。
普通伺服 lag 仅记录 `seed_tracking_deg`，不 abort。仍硬中止：非有限/维度错误、反馈陈旧、
反馈超限 `>10°`、无任何合法候选、量化越限、CAN/驱动器/控制器异常。

`scripts/validate_hdf5_policy_path.py` 已改为调用 `PiperArm._create_eef_ik` 与生产
`PiperArm.move_eef`，不再单独构造或复制 IK。按 `recursive + H=30 + 30Hz + no guard +
no arrival wait` 全量扫描 `datasets/stack cups`：

| 项 | Pinocchio 结果 |
|---|---|
| episode / action / command | **202/202；45,294/45,294；45,294/45,294** |
| abort / JointCtrl 越限 | **0 / 0** |
| 最大位置/旋转 residual | **0.6044mm / 0.6693°** |
| 最大单步关节变化 | **5.0000°**（要求 ≤5.001°） |
| 物理反馈最大超限投影 | 0.0280° |
| saturated / unconverged | 449 / 0 |
| 最大 iterations | 21 |

同一次扫描逐帧比较 Pinocchio JointCtrl 与 HDF5 下一帧记录关节角：六轴合并 MAE
**0.0315°**、RMSE **0.1678°**，绝对误差 P95/P99 为 **0.0980°/0.4510°**。
逐轴 MAE（j1..j6）为 `0.0293/0.0143/0.0209/0.0530/0.0138/0.0579°`。最大值
9.643° 出现在 `31.hdf5` action 37 的 j6；此时 j4/j6 虽分别相差 8.952°/9.643°，
两组关节角 FK 的末端仅差 0.036mm/0.025°，Jacobian 条件数约 2.17e4。这是腕部奇异点
附近同一末端位姿的零空间放大，而非模型坐标不一致或随机换支；EEF policy 本身没有
示教参考关节角来唯一指定该零空间。

报告：`runs/validation/hdf5-policy-path-pinocchio.json`。新增边界测试覆盖 10°、
10.001°、不可达目标最近可行解、量化限位、max-iterations best candidate、
realized-FK arrival、单位错误和生产 trace 字段；脚本单测 37 项全过。

同步 PI0.5 dry-run 曾连续暴露两项只适用于真运动的反馈检查：第一条虚拟动作后，
到位轮询经 `_wait_arrival → assert_healthy → _assert_writable` 报只读会话不可写；
关闭到位轮询后，静止反馈又在第 29 条虚拟目标累计到 81.6mm 时触发 80mm 跟踪闸。
这两项都是 dry-run 假阳性：dry-run 正确地以 `read_only=True` 连接且不发送目标，反馈
本来就不可能到达或跟随虚拟轨迹。现由 `deploy_client.py` 在任何 `--dry-run` 下强制
关闭到位等待，并让执行器只跳过 feedback tracking 闸；动作幅度、SO(3) 旋转、工作空间、
相邻目标跳变和夹爪检查仍全部保留。`run_eval_client.sh sync` 同时显式附加
`--no-wait-arrival`。真实 `--execute` 会继续启用到位等待和 80mm 跟踪中止，阈值未放宽。

## π0.5 单相机 LoRA + RTC 训练已完成，评测服务已恢复

`pi0.5/` 已包含 HDF5→LeRobot v3、HF 预下载、PEFT LoRA 训练、LeRobot adapter 加载和
robokit TCP 推理服务；另已按 arXiv:2506.07339 增加 PI05 专用 RTC 异步推理。采用
**LeRobot MP4+Parquet**，不采用 RLDS：普通自定义数据是 openpi/LeRobot 的主路径，
且服务器只需从 HF 下载训练消费格式。

单图不是阻塞项：PI05 可从数据集推断只有 `observation.images.cam_high` 这一项；额外相机
缺失时模型也有 mask 机制。训练脚本显式传 `--policy.input_features=null`，推理适配器反向
断言 checkpoint 仍是单图配置，防止训练/部署悄悄换字段。

当前 `datasets/stack cups` 全量重新清洗和 dry-run：

| 项 | 结果 |
|---|---|
| 源数据 | 202 段、45,496 原始时刻、10,505,312,600 bytes、320×240、30 Hz |
| 清洗 | 36 ok / 166 warn / **0 bad** |
| LeRobot 目标 | 45,294 帧、单 `right_arm` / `cam_high`、任务均为 `stack cups` |
| 最大局部 action | xyz 14.35 mm；rpy 4.60°，均在现有采集清洗阈值内 |
| π0.5 契约测试 | 8/8 通过；含 EEF delta、RTC 状态机/缓存及服务动作空间握手 |

166 段 warn 主要是欧拉数值 wrap；转换使用相邻旋转的 SO(3) 相对量，不直接相减欧拉数，
因此不会制造 ±2π 假动作。完整操作和逐文件用途已维护在 README 的「π0.5 单相机 LoRA」。

2026-07-28 在 RTX 6000D 服务器启动、并于 2026-07-29 完成
`stackcups_one_task` 守护任务：

| 项 | 当前值 |
|---|---|
| supervisor | 已完成；服务器已按成功条件关机 |
| 源 | `shaohuan1/lememory` 的 202 个 HDF5；不下载 RLDS |
| 派生数据 | 私有 `shaohuan1/stackcups_one_task_lerobot` |
| 最终权重 | 私有 `shaohuan1/stackcups_one_task` |
| 训练 | PI05 base + LoRA r32/alpha32，batch 8，最多 30k step |
| 曲线 | W&B [run fnbhi812](https://wandb.ai/yangshaohuan720-university/stackcups_one_task/runs/fnbhi812)，train 每 50 step、held-out eval 每 1000 step |
| 日志 | `/root/autodl-tmp/outputs/stackcups_one_task.supervisor.log` |
| PID 文件 | `/root/autodl-tmp/outputs/stackcups_one_task.supervisor.pid` |

环境实测为 LeRobot 0.6.1、torch 2.11.0+cu130，RTX 6000D `sm_120` BF16 matmul 通过；
HF token 身份为 `shaohuan1`，W&B API 鉴权通过。单段真实数据完成
HDF5→AV1 MP4→PyAV 解码与 7 维契约验证。服务器无系统 FFmpeg shared libs，因此数据验证、
预取和训练已显式固定 `video_backend=pyav`，不能改回自动选择的 TorchCodec。

首次准备已完成：HDF5→LeRobot 转换/发布是 202 episodes、45,294 frames；PI05 base
14,467,165,872 bytes 通过 HF LFS SHA-256 校验。HF token 没有 gated
`google/paligemma-3b-pt-224` 权限，第一次模型加载因此 403，失败输出完整归档在
`/root/autodl-tmp/outputs/failed/stackcups_one_task.gated-tokenizer/`。现使用 OpenPI 官方
匿名公开 PaliGemma SentencePiece（4,264,023 bytes、固定 SHA-256），Transformers
token IDs 已与官方模型逐 prompt 对齐，不替换 PI05 权重。

本次训练为 2,574,336 个 LoRA 可训练参数 / 4,145,979,152 总参数。W&B API 已回读
train step 24,400 及全部 24 个 held-out eval 点；step 16,000 的
`eval_loss=0.14999647438526154` 是按 0.5% 有效改善阈值选出的 best，17k–24k 连续
8 次未有效改善，收敛成立。step 19,000 的绝对值只低约 0.0099%，不改变平台选择。

收敛守护使用 held-out eval loss：至少 10k step/6 次 evaluation，连续 4 次相对改善
不足 0.5% 才判平台。最终已发布 step 16,000 到 HF 根目录，提交
`47690b157cafc54abbf0ccab33b930f21f62e972`；根文件与 16k checkpoint blob/LFS SHA
完全一致，之后服务器已关机。训练阶段已经结束，后续是最终权重的数字孪生人工审查和真机成功率验收。

RTC 已实现论文 Algorithm 1 的两部分：服务端用固定 LeRobot revision 的
`RTCProcessor` 在每个 flow step 做 ΠGDM prefix guidance；机器人端在旧 chunk 执行期间
异步请求新 chunk，以最近 10 次延迟最大值预测 `d`，回包后跳过已过去的 action。默认实机
参数为 `H=50, s_min=25, n=5, beta=5, b=10, EXP soft mask`。超过
`d <= s <= H-d` 或耗尽旧 chunk 会硬中止，不会重复末动作。

动作控制仍是原来的单臂 7 维局部 EEF delta：RTC 只改 chunk 调度和去噪引导，
`apply_local_delta_pose`、到位等待、生产 IK、安全闸与夹爪逻辑均未换。新增入口：
服务器 `pi0.5/serve_rtc.py`，机器人 `pi0.5/rtc_deploy_client.py`；完整命令见 README。

2026-07-29 为收敛前紧急真机评测增加统一入口：`run_eval_server.sh` 和
`run_eval_client.sh`。已审查部署候选固定为 step 12000（`eval_loss=0.1520`）；
训练 checkpoint 的 `chunk_size=50`，按用户要求同步服务和客户端都执行前 30 步。
正式命令在 README：原 `deploy_client.py` / `ChunkExecutor`、recursive EEF delta，
`--unsafe-unbounded --eef-backend pinocchio_ik --max-steps 0` 表示加载 no-guard 配置、
显式选择真正 Pinocchio 后端、关闭全部软件护栏且不限轮数；
单独的 `--no-guard` 只关闭 ActionGuard。最终 step 16000 尚未经过用户数字孪生审查；
训练结束时服务器曾按成功条件关机；2026-07-29 已重新开机，默认入口仍为 step 12000，
避免静默切换。RTC 因逐步局部 delta
缺少绝对轨迹累积/重锚适配而暂禁真机。

## HF 分批发布已跑通：202/202 段在 `shaohuan1/lememory`

`scripts/publish_batch.sh` 一条命令跑完 冻结快照 → 清洗 → 转 RLDS → **HDF5 与 RLDS 一起**
上传 → 逐文件比对远端大小 → 记台账。发布期间可以继续采集，新段留给下一批。用法与实测
开销见 README「分批发布到 HuggingFace」。

| 批次 | episode | HDF5 | RLDS | 状态 |
|---|---|---|---|---|
| `stack_cups_b0` | 0..65（66 段） | 3.07 GiB | 66 ep / 18 文件 / 1.76 GiB | 已校验 |
| `stack_cups_b1` | 66..201（136 段） | 6.71 GiB | 136 ep / 34 文件 / 3.74 GiB | 已校验 |

2026-07-28 全量核对：**254 个文件本地与远端大小逐一比对，全部一致**；采集 202 段、
已发布 202 段、未发布 0 段；两批的 RLDS episode 数都等于各自 HDF5 段数。

### 排除性结论：「50~100 的 HDF5 没上传」是误判，不要再去查

远端一个都不缺。看起来缺是两件事叠加：

1. **50~100 跨了两个批次目录**：50..65 在 `stack_cups_b0/hdf5/`，66..100 在
   `stack_cups_b1/hdf5/`。只看其中一个目录必然看到"半截"。
2. **HF 网页按字典序排**，不是数字序。`stack_cups_b1/hdf5/` 的第一屏是
   `100.hdf5 101.hdf5 …`，而 `66.hdf5` 排在 **第 103 位（共 136 个）**，要翻到最后才看得到。

**判据永远用 `--audit` 而不是网页目测**：

```bash
./scripts/publish_batch.sh --audit "stack cups"
```

它逐文件比对本地与远端大小、报缺号、报 RLDS 与 HDF5 段数是否一致，只读不写。
2026-07-28 实跑输出：`b0 66/66 已传`、`b1 136/136 已传`、`未发布 0 段`、`核对通过`。

发布流程第 7 步还会把「批次 → episode 范围」表刷进仓库首页 `README.md`，
所以在 HF 网页上也能直接看出每批应该有哪些段，不必翻文件列表。

**交接给下一个人（含 codex）**：发布这条线目前**没有待办**。新采了段之后跑
`./scripts/publish_batch.sh "stack cups"` 即可，台账会自动从 202 接着发；
怀疑没传全时先跑 `--audit`，它给的是判据，网页目测不是。

### 已修的两个真 bug

1. **台账记的是意图范围而不是快照实际内容。** 断点重跑时快照已存在会跳过冻结，而这期间
   `datasets/` 里又多了几十段，于是 `0..65` 被记成 `0..110`，下一批会从 111 开始、
   **把 66~110 永久跳过**。已改为一律以快照实际内容为准，并修正了历史台账。
2. **转换完成判定只看目录存不存在。** 半截产物（TFDS 中断留 `.incomplete*`）会被当成
   已完成然后上传垃圾。现在要求 `dataset_info.json` 存在**且** episode 数 == 快照段数，
   否则整体重转。

### 与 π0.5 路线的关系

上传的 **HDF5 是两条路线共同的源数据**，删不得（RLDS 不含关节角和各路时间戳）。
RLDS 服务的是 MemoryVLA / X-embodiment 那条路；π0.5 走 LeRobot MP4+Parquet，
两者都从同一批 HDF5 派生，互不冲突。

## 已排除的两个假因

这两条反复被怀疑过，都已用数据否掉，**不要再回去查**：

1. **不是模型的问题。** j5 贴着 70° 限位是 stack cups 的正常工作姿态——示教 15447 帧里
   41% 的帧 j5>60°、85% 的帧 j5>40°，最大值 70.01°。policy 在忠实复现这个姿态。
2. **不是 IK 的问题。** 顶住限位时位姿残差只有 0.002mm/0.001°，IK 在精确求解。

早期真因是 `reset_seed` 的越界容差取成了 JointCtrl 的**指令分辨率** 0.001°，而伺服停在
限位上时编码器会读出略微越界的值；当时先修为 0.3°。2026-07-29 的长时程 policy trace
进一步证明 0.3° 仍不足：15 次 `host_ik_input` 前一条命令都已把 `j5` 合法限制在
70.000°，abort 发生在读取贴限位反馈并重置 seed 时，而不是 IK 发送了越限目标。

旧 `host_ik` 已把 seed 反馈容差参数化；当前 `pinocchio_ik` 的正常与 no-guard 配置均以
`seed_limit_tolerance_deg=10.0` 执行用户指定规则：逐轴反馈超限 `≤10°` 时投影到最近限位
继续有界 IK，`>10°` 才 abort；所有 IK 候选和 JointCtrl 下发值始终受六轴硬边界约束。
每条 trace 保留 `seed_limit_overrun_deg`。

### HDF5 与 policy 控制路径全量验证（2026-07-29）

新增 `scripts/validate_hdf5_policy_path.py`，用现有 HDF5 构造训练契约相同的 local EEF
delta，并按正式同步推理相同的 `recursive + H=30 + no guard + no arrival wait` 进入生产
`ChunkExecutor → PiperArm.move_eef → PiperPinocchioIK → JointCtrl`。不是直接回放
HDF5 关节角；内存 SDK 只替代最后的 CAN 接收，因此不会运动真机或启动数字孪生。

实跑命令：

```bash
python scripts/validate_hdf5_policy_path.py \
  --config configs/piper_single_noguard.yaml \
  --data-dir "datasets/stack cups" --horizon 30 \
  --report runs/validation/hdf5-policy-path-pinocchio.json
```

权威 Pinocchio 结果：**202/202 episode、45,294/45,294 EEF commands、0 abort，
0 JointCtrl 越限，PASS**。最坏位置/旋转残差 0.604mm/0.669°，最大关节步长 5.000°，
数据反馈最大超限 0.028°；449 条 saturated、0 条 unconverged。该验证证明现有 HDF5
轨迹不会再因本机 IK 软件判据 abort，但不替代 CAN/驱动/碰撞/真实跟踪的真机验收。
同一报告的关节参考比较为六轴 MAE 0.0315°、P99 0.4510°；最大 9.643° 离群点来自
腕部奇异处的零空间差异，末端 FK 仍只差 0.036mm/0.025°。

## 下一步

1. **跑长时程 policy，看能否完成 stack cups 任务。** 执行链路已通，剩下的是策略本身
   在新场景的表现。注意 3a365c8 的离线判定：新场景位姿 97% 落在旧数据分布外。
2. **复看 30Hz 闭环速率。** `--no-wait-arrival` 下 trace 的实际步频是否接近训练的 30Hz。
3. **做一轮 π0 joint 真机测试。** 8081 已验证 joint/H=15，绝对deadline调度实测
   29.944Hz；当前 joint 专用配置按用户决定关闭 ActionGuard，超限轴钳到机械限位。
   真机从 `--max-steps 1` 开始。

## 待解决

1. **`max_seed_tracking_deg: 1.0` 只在 `--wait-arrival` 模式下自洽。** 该闸比的是
   「反馈关节 vs 上一条命令关节」。74 段示教相邻帧关节步长 p50=0.855°、**p90=2.01°**、p99=2.72°。
   30Hz 流式下真机只要落后一个命令步，p90 那些帧就超 1° 直接拒发。要跑流式必须放到 ≥3°。
2. **现场光照要先确认。** 2026-07-27 实测 luminance 5.7 vs 训练现场 118，暗场闸会直接
   拦下。真机跑之前先看一眼 trace 里的 `luminance`（正常约 120）。
   > 原先挂在这条下面的 `validate_policy_delta_real.py` 已随清理删除。
3. ~~训练/推理图像几何不一致~~ —— **已关闭**，见下方「图像预处理已对齐训练」。
   > 另两条历史 warn 也已关闭：G1b 双锚定属于已删除的 review 机制；G6 旧数据重复帧
   > 40.8% 随旧数据作废而失效（现在只用 `datasets/`）。

## 图像预处理已对齐训练（2026-07-28）

### 采集预览使用 leMemory train_aligned（2026-08-02）

`configs/piper_single.yaml` 的采集预览与 MemoryVLA 推理现在共用
`robokit.image.lememory_preprocess`：完整 `240x320` 帧直接拉伸到 `224x224`，再做
`scale=0.9, ratio=1.0` 的中心裁剪并放回 `224x224`。采集窗口显示的是这个模型构图，
HDF5 仍保存原始 `240x320`，不把预览图反写进训练数据。

`DEFAULT_PREPROCESS` = **`train_aligned`**（`robokit/policies/memvla_lora.py`）。

**上游 laMemoryVLA 仓库内部两条路径自己就不一致**，逐层追过：

- 训练 `vla/datasets/datasets.py:140` → `obs_transforms.decode_and_resize` →
  `dlimp.resize_image` = `tf.image.resize(img, (224,224), "lanczos3")`，**无裁剪、不保长宽比**；
  再 `image_aug`（`train.py:54` 默认 True）做 `random_resized_crop(scale=[0.9,0.9], ratio=[1.0,1.0])`。
- 部署 `deploy.py:156` 多做一步 `image.crop((left_margin,0,left_margin+h,h))` 中心方形裁剪，
  **训练侧没有任何对应物**。上游自己的注释也只解释了 0.9 那一步。

0.9 裁剪两边是对得上的（部署取中心 = 随机位置的期望值）。多出来的只有那行方形裁剪。

**权重的输入域由训练决定，所以部署必须复现训练那条。** 我们当初照抄了上游 deploy.py，
连它这个 bug 一起抄了。修正只改我们自己的默认值，**上游 codebase 一个字节都没动**。

实测（`datasets/stack cups/0.hdf5` f100，320×240）：两条路径像素 MAE **17.57**；
横向视野 `deploy` 71.2% vs `train_aligned` 94.9%，差 **23.7 个百分点**。

> 之前一直没改，是因为旧 checkpoint 是别人训的、不知道其真实训练配置，改默认等于把一个
> 未验证假设换成另一个。现在用本 codebase 自己重训，训练几何是确定的，不需要再 A/B。
> 对照路径 `--policy-arg preprocess=deploy` 保留。

## 参数现状

`configs/piper_single.yaml`：`speed: 100`、`eef_backend: pinocchio_ik`、`reset_on_disconnect: true`、
`joint_limits_deg` 显式写死（与 `piper_limits.py` 一致）。

CAN 初始化必须把“停接口、设置 bitrate、启接口”拆成三步。`can0` 已处于 UP 时执行
`ip link set can0 up type can bitrate 1000000`，内核必然返回
`RTNETLINK answers: Device or resource busy`；这不是进程抢占，也不表示当前波特率错误。
2026-07-28 现场复现时 `can0` 原本就已是 1 Mbps / ERROR-ACTIVE，CAN receive list 无消费者，
总线错误为 0。按 README 三步重配后仍为 1 Mbps / ERROR-ACTIVE，1 秒收到约 3,300 帧，
说明适配器和机械臂总线均正常。重配会短暂断总线，必须先停止真机控制进程。

2026-08-02 采集启动失败的现场判据：`can0` 为 `UP / LOWER_UP / ERROR-ACTIVE / 1Mbps`，
但 RX/TX/全部总线错误计数均为 0，`candump` 连续 3 秒无帧；USB-CAN 已枚举，内核此前出现
过 `failed to re-submit IN URB: -EPERM` 并多次 USB 断连。当前阻塞是机械臂侧没有 CAN
电气流量或适配器接收仍需物理重插，不是 SDK 漏读。无密码 sudo 无法代用户重建接口。
同时已修复连接事务：机械臂反馈超时时 `Robot.connect` 回滚相机与全部臂，Piper 只读连接
自身也关闭 SDK；实机复跑从原来的 C++ abort/core dump 变为干净 `exit=1`。新增 3 个纯软件
回滚测试通过。

`configs/piper_single_noguard.yaml`：全部安全闸关闭的诊断对照，**只保留** `max_step_deg: 5.0`
和 `continuity_weight: 0.03` 两项 IK 分支选择约束。去掉这两项实测单步关节跳变到 103°
（j1 一步从 -71.8° 甩到 -142.0°，j6 从 +41.5° 甩到 -62.1°），22 步后撞限位中止。

`continuity_weight` 扫描（生产参数，74 段全量）：

| cw | 通过 | 最坏残差 | 最大步长 | >4°帧 |
|---|---|---|---|---|
| 0.03 | 15447/15447 | 0.7406mm / 0.8504° | 5.000° | 33 |
| 0.1 | 15447/15447 | 0.7736mm / 0.9010° | 5.000° | 15 |
| 0.3 | **11733/15447** | 0.8558mm / 0.9906° | 4.093° | 1 |
| 1.0 | **1021/15447** | 0.7256mm / 0.9979° | 1.529° | 0 |

调大能压住腕部零空间漂移，但过了 0.1 就够不到位姿，通过率断崖下跌。

## 仓库收口（2026-07-28 第二轮，删到底）

项目只保留两个功能：**采集数据 pipeline** 与 **执行推理任务**。
当前 `robokit/` 3895 行 / 26 个 Python 文件，`scripts/` 4077 行 / 13 个 Python 文件
（含 6 个单测）。

删除清单与每个存活文件的作用、依赖，见 [PROJECT_MEMORY.md](PROJECT_MEMORY.md) 的
「代码地图」。要点：

- **单测保留**（6 文件 37 用例，纯软件约 0.2 秒）。它们是改代码后唯一的自动回归手段，
  守的正是两个核心功能：清洗漏判 = 坏数据进训练集，reset 生命周期错 = 机械臂失力。
- **原有 policy 注册表只剩 `dummy` + `memvla_lora`**，`memoryvla.py` / `openvla.py` 已删；
  新 PI05 adapter 独立放在 `pi0.5/`，由自己的 `serve_robokit.py` 启动，不改旧注册表。
- `checkpoints/` 4.0G、训练脚本、全部一次性探针与诊断脚本已删。
- `robokit/offline/`（数字孪生/回放）已删，见下方那条教训。

### 教训：删跨仓库被依赖的模块，仓库内 grep 查不出来

删 `robokit/offline/` 时**打断了外部工具** `/home/ysh/piper_data_reviewer`——它
`importlib.import_module("robokit.offline.simarm")` 动态导入，而且是另一个仓库，
在 robokit 里怎么 grep 都找不到。同时它还 import 了 `policies.memoryvla._preprocess`。

处理：把 `simarm.py` / `clock.py` 从 git 历史取出，迁进
`piper_data_reviewer/reviewer/_robokit_offline/`，reviewer 从此自包含；`_preprocess`
的 import 改指向 `memvla_lora`。两边都验证过 import 可解析。

**以后删 robokit 模块前，先 grep `/home/ysh/piper_data_reviewer`。**

### reviewer 已支持边采集边远程审查（2026-08-02）

`piper_data_reviewer` 现在会周期重扫数据目录，`collect.py` 每落盘一条新 episode，
同学已经打开的页面会在几秒内自己列出来。三条实测出来的关键事实：

- **`--share` 原先打印的是错的 IP。** 旧的 `lan_ipv4_addresses()` 用 UDP connect 探
  默认路由，本机常开 Mihomo TUN，探到的是 `198.18.0.1`——发给同学 100% 连不上。现在
  按网卡枚举并明确把代理 TUN / docker 网桥列为「不要发」。
- **本机 ufw 是 active 的。** 不放行端口时对方的连接被内核直接丢弃，现象和校园网
  客户端隔离完全一样，极易误判成「工具坏了」。`run.py --check-network` 会打印现成的
  `sudo ufw allow ...`，可以提前一天跑。
- **录制中的 `.hdf5.tmp` 绝不能当 episode 打开**（HDF5 没写完）。reviewer 只把它显示成
  「● 正在录制」，`close()` 重命名成 `N.hdf5` 那一刻才进入可审查列表。

**按序号缓存是这里的坑。** 重扫可能给 episode 重新编号（新任务目录会排在已有目录
之前），原先 `SourceManager._cache` 和 `encoded_image` 的 lru_cache 都按序号存，
换代后同一序号会返回上一代文件的图像——画面和文件名对不上且完全不报错。现在缓存改按
文件路径存，图像缓存加 `generation` 进 key，两条都有回归用例。

实测：`datasets/`（20 GB、800+ 文件）重扫一次约 7 ms；真实 episode 首次加载 408 帧
0.25 s、4148 帧 4.4 s（EEF→关节 IK 重建）。

### `--preprocess` 必须留在我们这侧

`memvla_lora.py` 原先**完全不做图像预处理**，把原始 4:3 帧直接丢给 `predict_action`。
删掉 `memoryvla.py` 会把刚做好的对齐一起丢掉，所以 `_preprocess` 已移植进
`memvla_lora.py`（实测 MAE 17.57，与迁移前一致）。详见上方「图像预处理已对齐训练」。
