#!/bin/bash
set -eux

export NCCL_SOCKET_IFNAME="^docker0,lo"
export NCCL_IB_DISABLE=0
export NCCL_TIMEOUT=1800000

# if node rank 0
axolotl preprocess train.yaml --output-dir=$BT_CHECKPOINT_DIR --dataset-prepared-path=${BT_CHECKPOINT_DIR}/last_run_prepared

torchrun --nnodes=$BT_GROUP_SIZE --nproc-per-node=$BT_NUM_GPUS --node-rank=$BT_NODE_RANK --rdzv-backend=c10d --rdzv-id=$BT_TRAINING_JOB_ID --rdzv-endpoint=${BT_LEADER_ADDR}:29400  -m axolotl.cli.train train.yaml --output-dir=$BT_CHECKPOINT_DIR --dataset-prepared-path=${BT_CHECKPOINT_DIR}/last_run_prepared
