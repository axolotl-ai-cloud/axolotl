#!/bin/bash
# Prior to running this script, export your HF_TOKEN and WANDB_API_KEY to your environment; i.e.
# export HF_TOKEN="..."
# export WANDB_API_KEY="..."
#

# ---------- SBATCH commands ---------- #
#SBATCH --job-name=axolotl-slurm-multinode
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=$NUM_NODES
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=128

export TORCH_DIST_INIT_BARRIER=0

srun axolotl preprocess train.yaml

srun axolotl train train.yaml --launcher torchrun -- \
    --nproc_per_node=$NUM_TRAINERS --nnodes=$NUM_NODES \
    --rdzv_id axolotl-cli --rdzv_backend c10d --rdzv_endpoint "${PRIMARY_ADDR}:${PRIMARY_PORT}" --rdzv-conf="join_timeout=1800"
