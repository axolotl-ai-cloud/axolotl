# Axolotl

<div align="center">
  <img src="image/axolotl.png" alt="axolotl" width="160">
  <div>
    <p>
      <b>One repo to finetune them all! </b>
    </p>
    <p>
      Go ahead and axolotl questions!!
    </p>
  </div>
</div>

## Axolotl supports

|          | fp16/fp32 | fp16/fp32 w/ lora | 4bit-quant | 4bit-quant w/flash attention | flash attention | xformers attention |
|----------|:----------|:------------------|------------|------------------------------|-----------------|--------------------|
| llama    | ✅         | ✅                 | ✅          | ✅                            | ✅               | ✅                  |
| Pythia   | ✅         | ✅                 | ❌          | ❌                            | ❌               | ❓                  |
| cerebras | ✅         | ✅                 | ❌          | ❌                            | ❌               | ❓                  |
| mpt      | ✅         | ❌                 | ❌          | ❌                            | ❌               | ❓                  |


## Quickstart ⚡

**Requirements**: Python 3.9. 

```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl

pip3 install -e .[int4]

accelerate config

# finetune
accelerate launch scripts/finetune.py examples/4bit-lora-7b/config.yml

# inference
accelerate launch scripts/finetune.py examples/4bit-lora-7b/config.yml \
    --inference --lora_model_dir="./llama-7b-lora-int4"
```

## Installation

### Environment

- Docker 
  ```bash
  docker run --gpus '"all"' --rm -it winglian/axolotl:main
  ```
  - `winglian/axolotl:dev`: dev branch
  - `winglian/axolotl-runpod:main`: for runpod

- Conda/Pip venv
  1. Install python **3.9**

  2. Install python dependencies with ONE of the following:
      - `pip3 install -e .[int4]` (recommended)
      - `pip3 install -e .[int4_triton]`
      - `pip3 install -e .`

### Dataset

Have dataset(s) in one of the following format (JSONL recommended):

- `alpaca`: instruction; input(optional)
  ```json
  {"instruction": "...", "input": "...", "output": "..."}
  ```
- `sharegpt`: conversations
  ```json
  {"conversations": [{"from": "...", "value": "..."}]}
  ```
- `completion`: raw corpus
  ```json
  {"text": "..."}
  ```

<details>

<summary>See other formats</summary>

- `jeopardy`: question and answer
  ```json
  {"question": "...", "category": "...", "answer": "..."}
  ```
- `oasst`: instruction
  ```json
  {"INSTRUCTION": "...", "RESPONSE": "..."}
  ```
- `gpteacher`: instruction; input(optional)
  ```json
  {"instruction": "...", "input": "...", "response": "..."}
  ```
- `reflection`: instruction with reflect; input(optional)
  ```json
  {"instruction": "...", "input": "...", "output": "...", "reflection": "...", "corrected": "..."}
  ```

> Have some new format to propose? Check if it's already defined in [data.py](src/axolotl/utils/data.py) in `dev` branch!

</details>

Optionally, download some datasets, see [data/README.md](data/README.md)

### Config

See sample configs in [configs](configs) folder or [examples](examples) for quick start. It is recommended to duplicate and modify to your needs. The most important options are:

- model
  ```yaml
  base_model: ./llama-7b-hf # local or huggingface repo
  ```
  Note: The code will load the right architecture.

- dataset
  ```yaml
  datasets:
    - path: vicgalle/alpaca-gpt4 # local or huggingface repo
      type: alpaca # format from earlier
  sequence_len: 2048 # max token length / prompt
  ```

- loading
  ```yaml
  load_4bit: true
  load_in_8bit: true
  bf16: true
  fp16: true
  tf32: true
  ```
  Note: Repo does not do 4-bit quantization.

- lora
  ```yaml
  adapter: lora # blank for full finetune
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  lora_target_modules:
    - q_proj
    - v_proj
  ```

<details>

<summary>All yaml options</summary>

```yaml
# this is the huggingface model that contains *.pt, *.safetensors, or *.bin files
# this can also be a relative path to a model on disk
base_model: ./llama-7b-hf
# you can specify an ignore pattern if the model repo contains more than 1 model type (*.pt, etc)
base_model_ignore_patterns:
# if the base_model repo on hf hub doesn't include configuration .json files,
# you can set that here, or leave this empty to default to base_model
base_model_config: ./llama-7b-hf
# If you want to specify the type of model to load, AutoModelForCausalLM is a good choice too
model_type: AutoModelForCausalLM
# Corresponding tokenizer for the model AutoTokenizer is a good choice
tokenizer_type: AutoTokenizer

# whether you are training a 4-bit quantized model
load_4bit: true
gptq_groupsize: 128 # group size
gptq_model_v1: false # v1 or v2

# this will attempt to quantize the model down to 8 bits and use adam 8 bit optimizer
load_in_8bit: true

# Use CUDA bf16
bf16: true
# Use CUDA fp16
fp16: true
# Use CUDA tf32
tf32: true

# a list of one or more datasets to finetune the model with
datasets:
  # this can be either a hf dataset, or relative path
  - path: vicgalle/alpaca-gpt4
  # The type of prompt to use for training. [alpaca, sharegpt, gpteacher, oasst, reflection]
    type: alpaca
    data_files: # path to source data files

# axolotl attempts to save the dataset as an arrow after packing the data together so
# subsequent training attempts load faster, relative path
dataset_prepared_path: data/last_run_prepared
# push prepared dataset to hub
push_dataset_to_hub: # repo path
# How much of the dataset to set aside as evaluation. 1 = 100%, 0.50 = 50%, etc
val_set_size: 0.04

# the maximum length of an input to train with, this should typically be less than 2048
# as most models have a token/context limit of 2048
sequence_len: 2048
# max sequence length to concatenate training samples together up to
# inspired by StackLLaMA. see https://huggingface.co/blog/stackllama#supervised-fine-tuning
max_packed_sequence_len: 1024

# if you want to use lora, leave blank to train all parameters in original model
adapter: lora
# if you already have a lora model trained that you want to load, put that here
# lora hyperparameters
lora_model_dir:
lora_r: 8
lora_alpha: 16
lora_dropout: 0.05
lora_target_modules:
  - q_proj
  - v_proj
#  - k_proj
#  - o_proj
#  - gate_proj
#  - down_proj
#  - up_proj
lora_modules_to_save:
#  - embed_tokens
#  - lm_head
lora_out_dir:
lora_fan_in_fan_out: false

# wandb configuration if you're using it
wandb_project:
wandb_watch:
wandb_run_id:
wandb_log_model: # 'checkpoint'

# where to save the finished model to
output_dir: ./completed-model

# training hyperparameters
batch_size: 8
micro_batch_size: 2
eval_batch_size: 2
num_epochs: 3
warmup_steps: 100
learning_rate: 0.00003
logging_steps:

# whether to mask out or include the human's prompt from the training labels
train_on_inputs: false
# don't use this, leads to wonky training (according to someone on the internet)
group_by_length: false

# does not work with current implementation of 4-bit LoRA
gradient_checkpointing: false

# stop training after this many evaluation losses have increased in a row
# https://huggingface.co/transformers/v4.2.2/_modules/transformers/trainer_callback.html#EarlyStoppingCallback
early_stopping_patience: 3
# specify a scheduler to use with the optimizer. only one_cycle is supported currently
lr_scheduler:
# specify optimizer
optimizer:
# specify weight decay
weight_decay:

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

# add or change special tokens
special_tokens:
  # bos_token: "<s>"
  # eos_token: "</s>"
  # unk_token: "<unk>"
# add extra tokens
tokens:

# FSDP
fsdp:
fsdp_config:

# Deepspeed
deepspeed:

# TODO
torchdistx_path:

# Debug mode
debug:
```

</details>

### Accelerate

Configure accelerate 

```bash
accelerate config

# Edit manually
# nano ~/.cache/huggingface/accelerate/default_config.yaml
```

### Train

Run
```bash
accelerate launch scripts/finetune.py configs/your_config.yml
```

### Inference

Add `--inference` flag to train command above

If you are inferencing a pretrained LORA, pass 
```bash
--lora_model_dir ./completed-model
```

### Merge LORA to base (Dev branch 🔧 )

Add below flag to train command above

```bash
--merge_lora --lora_model_dir="./completed-model"
```

## Common Errors 🧰

> Cuda out of memory

Please reduce any below
  - `micro_batch_size`
  - `eval_batch_size`
  - `sequence_len`

## Contributing 🤝

Bugs? Please check for open issue else create a new [Issue](https://github.com/OpenAccess-AI-Collective/axolotl/issues/new).

PRs are **greatly welcome**!