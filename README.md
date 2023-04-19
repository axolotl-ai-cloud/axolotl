# Axolotl

#### Go ahead and axolotl questions

## Support Matrix

|          | fp16/fp32 | fp16/fp32 w/ lora | 4bit-quant | 4bit-quant w/flash attention | flash attention | xformers attention |
|----------|:----------|:------------------|------------|------------------------------|-----------------|--------------------|
| llama    | ✅         | ✅                 | ✅          | ✅                            | ✅               | ✅                  |
| Pythia   | ✅         | ✅                 | ❌          | ❌                            | ❌               | ❓                  |
| cerebras | ✅         | ✅                 | ❌          | ❌                            | ❌               | ❓                  |


## Getting Started

- Point the config you are using to a huggingface hub dataset (see [configs/llama_7B_4bit.yml](https://github.com/winglian/axolotl/blob/main/configs/llama_7B_4bit.yml#L6-L8))

```yaml
datasets:
  - path: vicgalle/alpaca-gpt4
    type: alpaca
```

- Optionally Download some datasets, see [data/README.md](data/README.md)


- Create a new or update the existing YAML config [config/sample.yml](config/sample.yml)

```yaml
# this is the huggingface model that contains *.pt, *.safetensors, or *.bin files
# this can also be a relative path to a model on disk
base_model: decapoda-research/llama-7b-hf-int4
# you can specify an ignore pattern if the model repo contains more than 1 model type (*.pt, etc)
base_model_ignore_patterns:
# if the base_model repo on hf hub doesn't include configuration .json files,
# you can set that here, or leave this empty to default to base_model
base_model_config: decapoda-research/llama-7b-hf
# If you want to specify the type of model to load, AutoModelForCausalLM is a good choice too
model_type: AutoModelForCausalLM
# Corresponding tokenizer for the model AutoTokenizer is a good choice
tokenizer_type: AutoTokenizer
# whether you are training a 4-bit quantized model
load_4bit: true
# this will attempt to quantize the model down to 8 bits and use adam 8 bit optimizer
load_in_8bit: true
# a list of one or more datasets to finetune the model with
datasets:
  # this can be either a hf dataset, or relative path
  - path: vicgalle/alpaca-gpt4
  # The type of prompt to use for training. [alpaca, sharegpt, gpteacher, oasst, reflection]
    type: alpaca
# axolotl attempts to save the dataset as an arrow after packing the data together so
# subsequent training attempts load faster, relative path
dataset_prepared_path: data/last_run_prepared
# How much of the dataset to set aside as evaluation. 1 = 100%, 0.50 = 50%, etc
val_set_size: 0.04
# if you want to use lora, leave blank to train all parameters in original model
adapter: lora
# if you already have a lora model trained that you want to load, put that here
lora_model_dir:
# the maximum length of an input to train with, this should typically be less than 2048
# as most models have a token/context limit of 2048
sequence_len: 2048
# max sequence length to concatenate training samples together up to
# inspired by StackLLaMA. see https://huggingface.co/blog/stackllama#supervised-fine-tuning
max_packed_sequence_len: 1024
# lora hyperparameters
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
#  - k_proj
#  - o_proj
lora_fan_in_fan_out: false
# wandb configuration if your're using it
wandb_project:
wandb_watch:
wandb_run_id:
wandb_log_model: checkpoint
# where to save the finsihed model to
output_dir: ./completed-model
# training hyperparameters
batch_size: 8
micro_batch_size: 2
num_epochs: 3
warmup_steps: 100
learning_rate: 0.00003
# whether to mask out or include the human's prompt from the training labels
train_on_inputs: false
# don't use this, leads to wonky training (according to someone on the internet)
group_by_length: false
# Use CUDA bf16
bf16: true
# Use CUDA tf32
tf32: true
# does not work with current implementation of 4-bit LoRA
gradient_checkpointing: false
# stop training after this many evaluation losses have increased in a row
# https://huggingface.co/transformers/v4.2.2/_modules/transformers/trainer_callback.html#EarlyStoppingCallback
early_stopping_patience: 3
# specify a scheduler to use with the optimizer. only one_cycle is supported currently
lr_scheduler:
# whether to use xformers attention patch https://github.com/facebookresearch/xformers:
xformers_attention:
# whether to use flash attention patch https://github.com/HazyResearch/flash-attention:
flash_attention:
# resume from a specific checkpoint dir
resume_from_checkpoint:
# if resume_from_checkpoint isn't set and you simply want it to start where it left off
# be careful with this being turned on between different models
auto_resume_from_checkpoints: false
# don't mess with this, it's here for accelerate and torchrun
local_rank:
```

- Install python dependencies with ONE of the following:

    - `pip3 install -e .[int4]` (recommended)
    - `pip3 install -e .[int4_triton]`
    - `pip3 install -e .`
-
- If not using `int4` or `int4_triton`, run `pip install "peft @ git+https://github.com/huggingface/peft.git"`
- Configure accelerate `accelerate config` or update `~/.cache/huggingface/accelerate/default_config.yaml`

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

- Train! `accelerate launch scripts/finetune.py`, make sure to choose the correct YAML config file
- Alternatively you can pass in the config file like: `accelerate launch scripts/finetune.py configs/llama_7B_alpaca.yml`~~


## How to start training on Runpod in under 10 minutes

- Choose your Docker container wisely.
- I recommend `huggingface:transformers-pytorch-deepspeed-latest-gpu` see https://hub.docker.com/r/huggingface/transformers-pytorch-deepspeed-latest-gpu/
- Once you start your runpod, and SSH into it:
```shell
export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
source <(curl -s https://raw.githubusercontent.com/winglian/axolotl/main/scripts/setup-runpod.sh)
```

- Once the setup script completes
```shell
accelerate launch scripts/finetune.py configs/quickstart.yml
```

- Here are some helpful environment variables you'll want to manually set if you open a new shell
```shell
export WANDB_MODE=offline
export WANDB_CACHE_DIR=/workspace/data/wandb-cache
export HF_DATASETS_CACHE="/workspace/data/huggingface-cache/datasets"
export HUGGINGFACE_HUB_CACHE="/workspace/data/huggingface-cache/hub"
export TRANSFORMERS_CACHE="/workspace/data/huggingface-cache/hub"
export NCCL_P2P_DISABLE=1
```

