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
    <img src="https://github.com/OpenAccess-AI-Collective/axolotl/actions/workflows/pre-commit.yml/badge.svg?branch=main" alt="pre-commit">
    <img alt="PyTest Status" src="https://github.com/OpenAccess-AI-Collective/axolotl/actions/workflows/tests.yml/badge.svg?branch=main">
  </div>
</div>

## Axolotl supports

|         | fp16/fp32 | fp16/fp32 w/ lora | qlora | 4bit-quant | 4bit-quant w/flash attention | flash attention | xformers attention |
|---------|:----------|:------------------|------|------------|------------------------------|-----------------|--------------------|
| llama   | ✅         | ✅                 | ✅  | ✅          | ✅                            | ✅               | ✅                  |
| Pythia  | ✅         | ✅                 | ❓  | ❌          | ❌                            | ❌               | ❓                  |
| cerebras | ✅         | ✅                 | ❓  | ❌          | ❌                            | ❌               | ❓                  |
| mpt     | ✅         | ❌                 | ❓  | ❌          | ❌                            | ❌               | ❓                  |
| falcon  | ✅         | ✅                 | ✅  | ❌          | ❌                            | ❌               | ❓                  |


## Quickstart ⚡

**Requirements**: Python 3.9 and Pytorch 2.0.

```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl

pip3 install -e .
pip3 install -U git+https://github.com/huggingface/peft.git

accelerate config

# finetune lora
accelerate launch scripts/finetune.py examples/lora-openllama-3b/config.yml

# inference
accelerate launch scripts/finetune.py examples/lora-openllama-3b/config.yml \
    --inference --lora_model_dir="./lora-out"
```

## Installation

### Environment

- Docker
  ```bash
  docker run --gpus '"all"' --rm -it winglian/axolotl:main-py3.9-cu118-2.0.0
  ```
  - `winglian/axolotl-runpod:main-py3.9-cu118-2.0.0`: for runpod
  - `winglian/axolotl-runpod:main-py3.9-cu118-2.0.0-gptq`: for gptq
  - `winglian/axolotl:dev`: dev branch (not usually up to date)

  Or run on the current files for development:

  ```sh
  docker compose up -d
  ```

- Conda/Pip venv
  1. Install python **3.9**

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
  pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
  ```

  4. Axolotl
  ```bash
  git clone https://github.com/OpenAccess-AI-Collective/axolotl
  cd axolotl

  pip3 install -e . # change depend on needs
  pip3 install protobuf==3.20.3
  pip3 install -U requests
  pip3 install -U --ignore-installed psutil
  pip3 install -U scipy
  pip3 install git+https://github.com/huggingface/peft.git # not for gptq
  ```

  5. Set path
  ```bash
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  ```
  </details>

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

</details>

#### How to add custom prompts

  1. Add your method to a file in [prompt_strategies](src/axolotl/prompt_strategies). Please see other files as example.
  2. Use your custom file name as the dataset type.

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
  load_in_4bit: true
  load_in_8bit: true
  bf16: true # require >=ampere
  fp16: true
  tf32: true # require >=ampere
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
# Optional tokenizer configuration override in case you want to use a different tokenizer
# than the one defined in the base model
tokenizer_config:
# If you want to specify the type of model to load, AutoModelForCausalLM is a good choice too
model_type: AutoModelForCausalLM
# Corresponding tokenizer for the model AutoTokenizer is a good choice
tokenizer_type: AutoTokenizer
# Trust remote code for untrusted source
trust_remote_code:

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
  # this can be either a hf dataset, or relative path
  - path: vicgalle/alpaca-gpt4
  # The type of prompt to use for training. [alpaca, sharegpt, gpteacher, oasst, reflection]
    type: alpaca # format OR format:prompt_style (chat/instruct)
    data_files: # path to source data files
    shards: # number of shards to split data into

# axolotl attempts to save the dataset as an arrow after packing the data together so
# subsequent training attempts load faster, relative path
dataset_prepared_path: data/last_run_prepared
# push prepared dataset to hub
push_dataset_to_hub: # repo path
# whether to use hf `use_auth_token` for loading datasets. Useful for fetching private datasets
# required to be true when used in combination with `push_dataset_to_hub`
hf_use_auth_token: # boolean
# How much of the dataset to set aside as evaluation. 1 = 100%, 0.50 = 50%, etc
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
max_packed_sequence_len: 1024

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

# wandb configuration if you're using it
wandb_mode:
wandb_project:
wandb_watch:
wandb_run_id:
wandb_log_model: # 'checkpoint'

# where to save the finished model to
output_dir: ./completed-model

# training hyperparameters
gradient_accumulation_steps: 1
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

# whether to use xformers attention patch https://github.com/facebookresearch/xformers:
xformers_attention:
# whether to use flash attention patch https://github.com/HazyResearch/flash-attention:
flash_attention:  # require a100 for llama
# whether to use scaled-dot-product attention
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
sdp_attention:
# Landmark attention (only llama)
landmark_attention:
# xpos RoPE see https://github.com/kaiokendev/cutoff-len-is-context-len/blob/main/util/xpos_rope_llama_monkey_patch.py
# llama only
xpos_rope:

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

# Path to torch distx for optim 'adamw_anyprecision'
torchdistx_path:

# Set padding for data collator to 'longest'
collator_pad_to_longest:

# Debug mode
debug:

# Seed
seed:

# Allow overwrite yml config using from cli
strict:
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

Pass the appropriate flag to the train command:

- Pretrained LORA:
  ```bash
  --inference --lora_model_dir ./completed-model
  ```
- Full weights finetune:
  ```bash
  --inference --base_model ./completed-model
  ```

### Merge LORA to base

Add below flag to train command above

```bash
--merge_lora --lora_model_dir="./completed-model" --load_in_8bit=False --load_in_4bit=False
```

## Common Errors 🧰

> Cuda out of memory

Please reduce any below
  - `micro_batch_size`
  - `eval_batch_size`
  - `gradient_accumulation_steps`
  - `sequence_len`

> RuntimeError: expected scalar type Float but found Half

Try set `fp16: true`

> NotImplementedError: No operator found for `memory_efficient_attention_forward` ...

Try to turn off xformers.

## Need help? 🙋‍♂️

Join our [Discord server](https://discord.gg/HhrNrHJPRb) where we can help you

## Badge ❤🏷️

Building something cool with Axolotl? Consider adding a badge to your model card.

```markdown
[<img src="https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/OpenAccess-AI-Collective/axolotl)
```

[<img src="https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/OpenAccess-AI-Collective/axolotl)

## Contributing 🤝

Bugs? Please check for open issue else create a new [Issue](https://github.com/OpenAccess-AI-Collective/axolotl/issues/new).

PRs are **greatly welcome**!

Please run below to setup env
```bash
pip3 install -r requirements-dev.txt -r requirements-tests.txt
pre-commit install

# test
pytest tests/
```
