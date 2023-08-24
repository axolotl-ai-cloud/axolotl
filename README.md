# Axolotl

Axolotl is a tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures.

<table>
<tr>
<td>

## Table of Contents
- [Introduction](#axolotl)
- [Supported Features](#axolotl-supports)
- [Quickstart](#quickstart-)
- [Installation](#installation)
  - [Docker Installation](#environment)
  - [Conda/Pip venv Installation](#condapip-venv)
  - [LambdaLabs Installation](#lambdalabs)
- [Dataset](#dataset)
  - [How to Add Custom Prompts](#how-to-add-custom-prompts)
  - [How to Use Custom Pretokenized Dataset](#how-to-use-your-custom-pretokenized-dataset)
- [Config](#config)
  - [Train](#train)
  - [Inference](#inference)
  - [Merge LORA to Base](#merge-lora-to-base)
- [Common Errors](#common-errors-)
- [Need Help?](#need-help-)
- [Badge](#badge-)
- [Community Showcase](#community-showcase)
- [Contributing](#contributing-)

</td>
<td>

<div align="center">
  <img src="image/axolotl.png" alt="axolotl" width="160">
  <div>
    <p>
      <b>Axolotl provides a unified repository for fine-tuning <br />a variety of AI models with ease</b>
    </p>
    <p>
      Go ahead and axolotl questions!!
    </p>
    <img src="https://github.com/OpenAccess-AI-Collective/axolotl/actions/workflows/pre-commit.yml/badge.svg?branch=main" alt="pre-commit">
    <img alt="PyTest Status" src="https://github.com/OpenAccess-AI-Collective/axolotl/actions/workflows/tests.yml/badge.svg?branch=main">
  </div>
</div>

</td>
</tr>
</table>

## Axolotl supports

|          | fp16/fp32 | lora | qlora | gptq | gptq w/flash attn | flash attn | xformers attn |
|----------|:----------|:-----|-------|------|-------------------|------------|---------------|
| llama    | ✅         | ✅    | ✅     | ✅             | ✅                 | ✅          | ✅             |
| Pythia   | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓             |
| cerebras | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓             |
| mpt      | ✅         | ❌    | ❓     | ❌             | ❌                 | ❌          | ❓             |
| falcon   | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓             |
| gpt-j    | ✅         | ✅    | ✅     | ❌             | ❌                 | ❓          | ❓             |
| XGen     | ✅         | ❓    | ✅     | ❓             | ❓                 | ❓          | ✅             |


## Quickstart ⚡

Get started with Axolotl in just a few steps! This quickstart guide will walk you through setting up and running a basic fine-tuning task.

**Requirements**: Python >=3.9 and Pytorch >=2.0.

```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip3 install -e .[flash-attn]
pip3 install -U git+https://github.com/huggingface/peft.git

# finetune lora
accelerate launch scripts/finetune.py examples/openllama-3b/lora.yml

# inference
accelerate launch scripts/finetune.py examples/openllama-3b/lora.yml \
    --inference --lora_model_dir="./lora-out"
```

## Installation

### Environment

- Docker
  ```bash
  docker run --gpus '"all"' --rm -it winglian/axolotl:main-py3.10-cu118-2.0.1
  ```
  - `winglian/axolotl-runpod:main-py3.10-cu118-2.0.1`: for runpod
  - `winglian/axolotl-runpod:main-py3.9-cu118-2.0.1-gptq`: for gptq

  Or run on the current files for development:

  ```sh
  docker compose up -d
  ```

- Conda/Pip venv
  1. Install python >=**3.9**

  2. Install pytorch stable https://pytorch.org/get-started/locally/

  3. Install python dependencies with ONE of the following:
      - Recommended, supports QLoRA, NO gptq/int4 support
        ```bash
        pip3 install -e .
        pip3 install -U git+https://github.com/huggingface/peft.git
        ```
      - gptq/int4 support, NO QLoRA
        ```bash
        pip3 install -e .[gptq]
        ```
      - same as above but not recommended
        ```bash
        pip3 install -e .[gptq_triton]
        ```

- LambdaLabs
  <details>

  <summary>Click to Expand</summary>

  1. Install python
  ```bash
  sudo apt update
  sudo apt install -y python3.9

  sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1
  sudo update-alternatives --config python # pick 3.9 if given option
  python -V # should be 3.9

  ```

  2. Install pip
  ```bash
  wget https://bootstrap.pypa.io/get-pip.py
  python get-pip.py
  ```

  3. Install torch
  ```bash
  pip3 install -U torch --index-url https://download.pytorch.org/whl/cu118
  ```

  4. Axolotl
  ```bash
  git clone https://github.com/OpenAccess-AI-Collective/axolotl
  cd axolotl

  pip3 install -e . # change depend on needs
  pip3 install protobuf==3.20.3
  pip3 install -U --ignore-installed requests Pillow psutil scipy
  pip3 install git+https://github.com/huggingface/peft.git # not for gptq
  ```

  5. Set path
  ```bash
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  ```
  </details>

### Dataset

Axolotl supports a variety of dataset formats. Below are some of the formats you can use.
Have dataset(s) in one of the following format (JSONL recommended):

- `alpaca`: instruction; input(optional)
  ```json
  {"instruction": "...", "input": "...", "output": "..."}
  ```
- `sharegpt:chat`: conversations where `from` is `human`/`gpt`
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
- `explainchoice`: question, choices, (solution OR explanation)
  ```json
  {"question": "...", "choices": ["..."], "solution": "...", "explanation": "..."}
  ```
- `concisechoice`: question, choices, (solution OR explanation)
  ```json
  {"question": "...", "choices": ["..."], "solution": "...", "explanation": "..."}
  ```
- `summarizetldr`: article and summary
  ```json
  {"article": "...", "summary": "..."}
  ```
- `alpaca_chat`: basic instruct for alpaca chat
  ```json
  {"instruction": "...", "input": "...", "response": "..."}
  ```
- `alpaca_chat.load_qa`: question and answer for alpaca chat
  ```json
  {"question": "...", "answer": "..."}
  ```
- `alpaca_chat.load_concise`: question and answer for alpaca chat, for concise answers
  ```json
  {"instruction": "...", "input": "...", "response": "..."}
  ```
- `alpaca_chat.load_camel_ai`: question and answer for alpaca chat, for load_camel_ai
  ```json
  {"message_1": "...", "message_2": "..."}
  ```
- `alpaca_w_system.load_open_orca`: support for open orca datasets with included system prompts, instruct
  ```json
  {"system_prompt": "...", "question": "...", "response": "..."}
  ```
- `context_qa`: in context question answering from an article
  ```json
  {"article": "...", "question": "...", "answer": "..."}
  ```
- `context_qa.load_404`: in context question answering from an article, with default response for no answer from context
  ```json
  {"article": "...", "unanswerable_question": "..."}
  ```
- `creative_acr.load_answer`: instruction and revision
  ```json
  {"instruction": "...", "revision": "..."}
  ```
- `creative_acr.load_critique`: critique
  ```json
  {"scores": "...", "critiques": "...", "instruction": "...", "answer": "..."}
  ```
- `creative_acr.load_revise`: critique and revise
  ```json
  {"scores": "...", "critiques": "...", "instruction": "...", "answer": "...", "revision": "..."}
  ```
- `pygmalion`: pygmalion
  ```json
  {"conversations": [{"role": "...", "value": "..."}]}
  ```
- `metharme`: instruction, adds additional eos tokens
  ```json
  {"prompt": "...", "generation": "..."}
  ```
- `sharegpt_simple.load_role`: conversations where `role` is used instead of `from`
  ```json
  {"conversations": [{"role": "...", "value": "..."}]}
  ```
- `sharegpt_simple.load_guanaco`: conversations where `from` is `prompter`/`assistant` instead of default sharegpt
  ```json
  {"conversations": [{"from": "...", "value": "..."}]}
  ```
- `sharegpt_jokes`: creates a chat where bot is asked to tell a joke, then explain why the joke is funny
  ```json
  {"conversations": [{"title": "...", "text": "...", "explanation": "..."}]}
  ```

</details>

#### How to add custom prompts

Using yaml. Example:
```yaml
datasets:
  - path: repo
    type:
      system_prompt: ""
      no_input_format: |-
        User: {instruction}<|end_of_turn|>
        Assistant:
      format: |-
        User: {instruction}
        {input}<|end_of_turn|>
        Assistant:
```

Using file:
1. Add your method to a file in [prompt_strategies](src/axolotl/prompt_strategies). Please see other files as example.
2. Use your custom file name as the dataset type `<prompt_strategies_file>.load_<load_fn>`.

#### How to use your custom pretokenized dataset

- Do not pass a `type:`
- Dataset must contain `input_ids`, `attention_mask`, `labels` in columns


### Config

See [examples](examples) for quick start. It is recommended to duplicate and modify to your needs. The most important options are:

- model
  ```yaml
  base_model: ./llama-7b-hf # local or huggingface repo
  ```
  Note: The code will load the right architecture.

- dataset
  ```yaml
  sequence_len: 2048 # max token length for prompt

  # huggingface repo
  datasets:
    - path: vicgalle/alpaca-gpt4
      type: alpaca # format from earlier

  # huggingface repo with specific configuration/subset
  datasets:
    - path: EleutherAI/pile
      name: enron_emails
      type: completion # format from earlier

  # local
  datasets:
    - path: data.jsonl # or json
      ds_type: json # see other options below
      type: alpaca
  ```

- loading
  ```yaml
  load_in_4bit: true
  load_in_8bit: true
  bf16: true # require >=ampere
  fp16: true
  tf32: true # require >=ampere
  bfloat16: true # require >=ampere, use instead of bf16 when you don't want AMP (automatic mixed precision)
  float16: true # use instead of fp16 when you don't want AMP
  ```
  Note: Repo does not do 4-bit quantization.

- lora
  ```yaml
  adapter: lora # qlora or leave blank for full finetune
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
# you can specify to choose a specific model revision from huggingface hub
model_revision:
# Optional tokenizer configuration override in case you want to use a different tokenizer
# than the one defined in the base model
tokenizer_config:
# If you want to specify the type of model to load, AutoModelForCausalLM is a good choice too
model_type: AutoModelForCausalLM
# Corresponding tokenizer for the model AutoTokenizer is a good choice
tokenizer_type: AutoTokenizer
# Trust remote code for untrusted source
trust_remote_code:
# use_fast option for tokenizer loading from_pretrained, default to True
tokenizer_use_fast:
# Whether to use the legacy tokenizer setting, defaults to True
tokenizer_legacy:
# resize the model embeddings when new tokens are added to multiples of 32
# this is reported to improve training speed on some models
resize_token_embeddings_to_32x:

# whether you are training a 4-bit GPTQ quantized model
gptq: true
gptq_groupsize: 128 # group size
gptq_model_v1: false # v1 or v2

# this will attempt to quantize the model down to 8 bits and use adam 8 bit optimizer
load_in_8bit: true
# use bitsandbytes 4 bit
load_in_4bit:

# Use CUDA bf16
bf16: true # bool or 'full' for `bf16_full_eval`. require >=ampere
# Use CUDA fp16
fp16: true
# Use CUDA tf32
tf32: true # require >=ampere

# a list of one or more datasets to finetune the model with
datasets:
  # hf dataset repo | "json" for local dataset, make sure to fill data_files
  - path: vicgalle/alpaca-gpt4
  # The type of prompt to use for training. [alpaca, sharegpt, gpteacher, oasst, reflection]
    type: alpaca # format | format:<prompt_style> (chat/instruct) | <prompt_strategies>.load_<load_fn>
    ds_type: # Optional[str] (json|arrow|parquet) defines the datatype when path is a file
    data_files: # path to source data files
    shards: # number of shards to split data into
    name: # name of dataset configuration to load

  # custom user prompt
  - path: repo
    type:
      # the below are defaults. only set what's needed.
      system_prompt: ""
      field_system: system
      field_instruction: instruction
      field_output: input

      # customizable to be single line or multi-line
      system_format: "{system}"
      # 'format' can include {input}
      format: |-
        User: {instruction} {input}
        Assistant:
      # 'no_input_format' cannot include {input}
      no_input_format: "{instruction} "

# axolotl attempts to save the dataset as an arrow after packing the data together so
# subsequent training attempts load faster, relative path
dataset_prepared_path: data/last_run_prepared
# push prepared dataset to hub
push_dataset_to_hub: # repo path
# push checkpoints to hub
hub_model_id: # repo path to push finetuned model
# how to push checkpoints to hub
# https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/trainer#transformers.TrainingArguments.hub_strategy
hub_strategy:
# whether to use hf `use_auth_token` for loading datasets. Useful for fetching private datasets
# required to be true when used in combination with `push_dataset_to_hub`
hf_use_auth_token: # boolean
# How much of the dataset to set aside as evaluation. 1 = 100%, 0.50 = 50%, etc. 0 for no eval.
val_set_size: 0.04
# Num shards for whole dataset
dataset_shard_num:
# Index of shard to use for whole dataset
dataset_shard_idx:

# the maximum length of an input to train with, this should typically be less than 2048
# as most models have a token/context limit of 2048
sequence_len: 2048
# max sequence length to concatenate training samples together up to
# inspired by StackLLaMA. see https://huggingface.co/blog/stackllama#supervised-fine-tuning
# FutureWarning: This will soon be DEPRECATED
max_packed_sequence_len: 1024
# use efficient multi-packing with block diagonal attention and per sequence position_ids. Recommend set to 'true'
sample_packing:
# you can set these packing optimizations AFTER starting a training at least once.
# The trainer will provide recommended values for these values.
sample_packing_eff_est:
total_num_tokens:

# if you want to use 'lora' or 'qlora' or leave blank to train all parameters in original model
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
lora_target_linear: # if true, will target all linear layers
lora_modules_to_save:
#  - embed_tokens
#  - lm_head
lora_out_dir:
lora_fan_in_fan_out: false

# ReLoRA configuration
# must use either 'lora' or 'qlora' adapter, and does not support fsdp or deepspeed
relora_steps: # number of steps per ReLoRA restart
relora_warmup_steps: # number of per-restart warmup steps
relora_cpu_offload: # true to perform lora weight merges on cpu during restarts, for modest gpu memory savings

# wandb configuration if you're using it
wandb_mode: # "offline" to save run metadata locally and not sync to the server, "disabled" to turn off wandb
wandb_project: # your wandb project name
wandb_entity: # a wandb Team name if using a Team
wandb_watch:
wandb_run_id: # set the name of your wandb run
wandb_log_model: # "checkpoint" to log model to wandb Artifacts every `save_steps` or "end" to log only at the end of training

# where to save the finished model to
output_dir: ./completed-model

# training hyperparameters
gradient_accumulation_steps: 1
micro_batch_size: 2
eval_batch_size: 2
num_epochs: 3
warmup_steps: 100
learning_rate: 0.00003
lr_quadratic_warmup:
logging_steps:
save_strategy: # set to `no` to skip checkpoint saves
save_steps: # leave empty to save at each epoch
eval_steps:
save_total_limit: # checkpoints saved at a time
max_steps:

# save model as safetensors (require safetensors package)
save_safetensors:

# whether to mask out or include the human's prompt from the training labels
train_on_inputs: false
# group similarly sized data to minimize padding
# may be slower to start, as it must download and sort the entire dataset
# note that training loss may have an oscillating pattern with this enabled
group_by_length: false

# Whether to use gradient checkpointing https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing
gradient_checkpointing: false

# stop training after this many evaluation losses have increased in a row
# https://huggingface.co/transformers/v4.2.2/_modules/transformers/trainer_callback.html#EarlyStoppingCallback
early_stopping_patience: 3

# specify a scheduler and kwargs to use with the optimizer
lr_scheduler: # 'one_cycle' | 'log_sweep' | empty for cosine
lr_scheduler_kwargs:

# for one_cycle optim
lr_div_factor: # learning rate div factor

# for log_sweep optim
log_sweep_min_lr:
log_sweep_max_lr:

# specify optimizer
optimizer:
# specify weight decay
weight_decay:
# adamw hyperparams
adam_beta1:
adam_beta2:
adam_epsilon:
# Gradient clipping max norm
max_grad_norm:

# whether to bettertransformers
flash_optimum:
# whether to use xformers attention patch https://github.com/facebookresearch/xformers:
xformers_attention:
# whether to use flash attention patch https://github.com/Dao-AILab/flash-attention:
flash_attention:
# whether to use scaled-dot-product attention
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
sdp_attention:
# Landmark attention (only llama)
landmark_attention:
# xpos RoPE see https://github.com/kaiokendev/cutoff-len-is-context-len/blob/main/util/xpos_rope_llama_monkey_patch.py
# llama only
xpos_rope:
# RoPE Scaling https://github.com/huggingface/transformers/pull/24653
rope_scaling:
  type: # linear | dynamic
  factor: # float

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

# Deepspeed config path
deepspeed:

# Path to torch distx for optim 'adamw_anyprecision'
torchdistx_path:

# Set padding for data collator to 'longest'
collator_pad_to_longest:

# Set to HF dataset for type: 'completion' for streaming instead of pre-tokenize
pretraining_dataset:

# Debug mode
debug:

# Seed
seed:

# Allow overwrite yml config using from cli
strict:
```

</details>

### Train

Run
```bash
accelerate launch scripts/finetune.py your_config.yml
```

#### Multi-GPU

You can optionally pre-tokenize dataset with the following before finetuning:
```bash
CUDA_VISIBLE_DEVICES="" accelerate ... --prepare_ds_only
```

##### Config

- llama FSDP
```yaml
fsdp:
  - full_shard
  - auto_wrap
fsdp_config:
  fsdp_offload_params: true
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_transformer_layer_cls_to_wrap: LlamaDecoderLayer
```

- llama Deepspeed
```yaml
deepspeed: deepspeed/zero3.json
```

##### Weights & Biases Logging

- wandb options
```yaml
wandb_mode:
wandb_project:
wandb_entity:
wandb_watch:
wandb_run_id:
wandb_log_model:
```

### Inference

Pass the appropriate flag to the train command:

- Pretrained LORA:
  ```bash
  --inference --lora_model_dir="./lora-output-dir"
  ```
- Full weights finetune:
  ```bash
  --inference --base_model="./completed-model"
  ```
- Full weights finetune w/ a prompt from a text file:
  ```bash
  cat /tmp/prompt.txt | python scripts/finetune.py configs/your_config.yml \
    --base_model="./completed-model" --inference --prompter=None --load_in_8bit=True
  ```

### Merge LORA to base

Add below flag to train command above

```bash
--merge_lora --lora_model_dir="./completed-model" --load_in_8bit=False --load_in_4bit=False
```

If you run out of CUDA memory, you can try to merge in system RAM with

```bash
CUDA_VISIBLE_DEVICES="" python3 scripts/finetune.py ...
```

## Common Errors 🧰

> If you encounter a 'Cuda out of memory' error, it means your GPU ran out of memory during the training process. Here's how to resolve it:

Please reduce any below
  - `micro_batch_size`
  - `eval_batch_size`
  - `gradient_accumulation_steps`
  - `sequence_len`

> `failed (exitcode: -9)`

Usually means your system has run out of system memory.
Similarly, you should consider reducing the same settings as when you run out of VRAM.
Additionally, look into upgrading your system RAM which should be simpler than GPU upgrades.

> RuntimeError: expected scalar type Float but found Half

Try set `fp16: true`

> NotImplementedError: No operator found for `memory_efficient_attention_forward` ...

Try to turn off xformers.

> accelerate config missing

It's safe to ignore it.

## Need help? 🙋♂️

Join our [Discord server](https://discord.gg/HhrNrHJPRb) where we can help you

## Badge ❤🏷️

Building something cool with Axolotl? Consider adding a badge to your model card.

```markdown
[<img src="https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/OpenAccess-AI-Collective/axolotl)
```

[<img src="https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/OpenAccess-AI-Collective/axolotl)

## Community Showcase

Check out some of the projects and models that have been built using Axolotl! Have a model you'd like to add to our Community Showcase? Open a PR with your model.

Open Access AI Collective
- [Minotaur 13b](https://huggingface.co/openaccess-ai-collective/minotaur-13b)
- [Manticore 13b](https://huggingface.co/openaccess-ai-collective/manticore-13b)
- [Hippogriff 30b](https://huggingface.co/openaccess-ai-collective/hippogriff-30b-chat)

PocketDoc Labs
- [Dan's PersonalityEngine 13b LoRA](https://huggingface.co/PocketDoc/Dans-PersonalityEngine-13b-LoRA)

## Contributing 🤝

Please read the [contributing guide](./.github/CONTRIBUTING.md)

Bugs? Please check the [open issues](https://github.com/OpenAccess-AI-Collective/axolotl/issues/bug) else create a new Issue.

PRs are **greatly welcome**!

Please run below to setup env
```bash
pip3 install -r requirements-dev.txt -r requirements-tests.txt
pre-commit install

# test
pytest tests/
```
