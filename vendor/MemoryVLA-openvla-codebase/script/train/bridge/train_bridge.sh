#!/bin/bash

# H800 NCCL 配置（单节点 8 卡，RoCE 网络）
export NCCL_DEBUG=WARN
export NCCL_SOCKET_IFNAME=eth0     # rendezvous 走管理网
export NCCL_IB_DISABLE=0           # 保留 RoCE（bond0~bond7）
export NCCL_IB_GID_INDEX=3         # RoCE v2
export NCCL_NET_GDR_LEVEL=2
export NCCL_P2P_DISABLE=0          # 启用 NVLink P2P
export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=1
export WANDB_MODE=offline   # 避免 wandb 网络连接阻塞 rank 0

pretrained_ckpt='./pretrained/CogACT-Large/checkpoints/CogACT-Large.pt'
hf_token='YOUR_HF_TOKEN'

data_root_dir='./data/bridge-rlds'
data_mix='bridge'

n_gpu=8
bs=32
shuffle_buffer_size=0 # if your memory is limited, try smaller value

save_interval=2500
dp_step=4
future_action_window_size=15

image_aug=True
run_root_dir='./log/bridge'
run_id='memvla_bridge'

is_resume=False
resume_step=0
resume_epoch=0

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
torchrun --nproc_per_node=8 --master_port=29501 train.py \
  --pretrained_checkpoint ${pretrained_ckpt} \
  --vla.type prism-dinosiglip-224px+oxe+diffusion \
  --vla.data_mix ${data_mix} \
  --vla.expected_world_size ${n_gpu} \
  --vla.per_device_batch_size ${bs} \
  --vla.global_batch_size $((n_gpu * bs)) \
  --vla.learning_rate 2e-5 \
  --vla.max_steps 50000 \
  --data_root_dir ${data_root_dir} \
  --run_root_dir ${run_root_dir} \
  --run_id ${run_id} \
  --image_aug ${image_aug} \
  --save_interval ${save_interval} \
  --repeated_diffusion_steps ${dp_step} \
  --future_action_window_size ${future_action_window_size} \
  --action_model_type 'DiT-L' \
  --dataloader_type 'stream' \
  --is_resume ${is_resume} \
  --resume_step ${resume_step} \
  --resume_epoch ${resume_epoch} \
  --wandb_project 'memvla' \
  --wandb_entity 'YOUR_WANDB_ENTITY' \
  --hf_token ${hf_token} \
  --vla.shuffle_buffer_size ${shuffle_buffer_size} \
