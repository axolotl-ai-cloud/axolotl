#!/bin/bash

export WANDB_MODE=offline
export WANDB_CACHE_DIR=/workspace/data/wandb-cache
mkdir -p $WANDB_CACHE_DIR

mkdir -p /workspace/data/huggingface-cache/{hub,datasets}
export HF_DATASETS_CACHE="/workspace/data/huggingface-cache/datasets"
export HUGGINGFACE_HUB_CACHE="/workspace/data/huggingface-cache/hub"
export TRANSFORMERS_CACHE="/workspace/data/huggingface-cache/hub"
export NCCL_P2P_DISABLE=1

nvidia-smi
num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
gpu_indices=$(seq 0 $((num_gpus - 1)) | paste -sd "," -)
export CUDA_VISIBLE_DEVICES=$gpu_indices
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

apt-get update
apt-get install -y build-essential ninja-build vim git-lfs
git lfs install
pip3 install --force-reinstall https://download.pytorch.org/whl/nightly/cu117/torch-2.0.0.dev20230301%2Bcu117-cp38-cp38-linux_x86_64.whl --index-url https://download.pytorch.org/whl/nightly/cu117
if [ -z "${TORCH_CUDA_ARCH_LIST}" ]; then # only set this if not set yet
    # this covers most common GPUs that the installed version of pytorch supports
    # python -c "import torch; print(torch.cuda.get_arch_list())"
    export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
fi

# install flash-attn and deepspeed from pre-built wheels for this specific container b/c these take forever to install
mkdir -p /workspace/wheels
cd /workspace/wheels
curl -L -O https://github.com/OpenAccess-AI-Collective/axolotl/raw/wheels/wheels/deepspeed-0.9.2%2B7ddc3b01-cp38-cp38-linux_x86_64.whl
curl -L -O https://github.com/OpenAccess-AI-Collective/axolotl/raw/wheels/wheels/flash_attn-1.0.4-cp38-cp38-linux_x86_64.whl
pip install deepspeed-0.9.2%2B7ddc3b01-cp38-cp38-linux_x86_64.whl
pip install flash_attn-1.0.4-cp38-cp38-linux_x86_64.whl
pip install "peft @ git+https://github.com/huggingface/peft.git@main" --force-reinstall --no-dependencies

cd /workspace/
git clone https://github.com/OpenAccess-AI-Collective/axolotl.git
cd axolotl
pip install -e .[int4]
mkdir -p ~/.cache/huggingface/accelerate/
cp configs/accelerate/default_config.yaml ~/.cache/huggingface/accelerate/default_config.yaml
