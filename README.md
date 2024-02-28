# Axolotl

Axolotl is a tool designed to streamline the fine-tuning of various AI models, offering support for multiple configurations and architectures.

Features:
- Train various Huggingface models such as llama, pythia, falcon, mpt
- Supports fullfinetune, lora, qlora, relora, and gptq
- Customize configurations using a simple yaml file or CLI overwrite
- Load different dataset formats, use custom formats, or bring your own tokenized datasets
- Integrated with xformer, flash attention, rope scaling, and multipacking
- Works with single GPU or multiple GPUs via FSDP or Deepspeed
- Easily run with Docker locally or on the cloud
- Log results and optionally checkpoints to wandb or mlflow
- And more!


<table>
<tr>
<td>

## Table of Contents
- [Introduction](#axolotl)
- [Supported Features](#axolotl-supports)
- [Quickstart](#quickstart-)
- [Environment](#environment)
  - [Docker](#docker)
  - [Conda/Pip venv](#condapip-venv)
  - [Cloud GPU](#cloud-gpu) - Latitude.sh, RunPod
  - [Bare Metal Cloud GPU](#bare-metal-cloud-gpu)
  - [Windows](#windows)
  - [Launching on public clouds via SkyPilot](#launching-on-public-clouds-via-skypilot)
- [Dataset](#dataset)
  - [How to Add Custom Prompts](#how-to-add-custom-prompts)
  - [How to Use Custom Pretokenized Dataset](#how-to-use-your-custom-pretokenized-dataset)
- [Config](#config)
  - [Train](#train)
  - [Inference](#inference-playground)
  - [Merge LORA to Base](#merge-lora-to-base)
  - [Special Tokens](#special-tokens)
- Advanced Topics
  - [Multipack](./docs/multipack.md)<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#666" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>
  - [RLHF & DPO](./docs/rlhf.md)<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#666" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>
- [Common Errors](#common-errors-)
  - [Tokenization Mismatch b/w Training & Inference](#tokenization-mismatch-bw-inference--training)
- [Debugging Axolotl](#debugging-axolotl)
- [Need Help?](#need-help-)
- [Badge](#badge-)
- [Community Showcase](#community-showcase)
- [Contributing](#contributing-)
- [Sponsors](#sponsors-)

</td>
<td>

<div align="center">
  <img src="image/axolotl.png" alt="axolotl" width="160">
  <div>
    <p>
      <b>Axolotl provides a unified repository for fine-tuning <br />a variety of AI models with ease</b>
    </p>
    <p>
      Go ahead and Axolotl questions!!
    </p>
    <img src="https://github.com/OpenAccess-AI-Collective/axolotl/actions/workflows/pre-commit.yml/badge.svg?branch=main" alt="pre-commit">
    <img alt="PyTest Status" src="https://github.com/OpenAccess-AI-Collective/axolotl/actions/workflows/tests.yml/badge.svg?branch=main">
  </div>
</div>

</td>
</tr>
</table>

## Axolotl supports

|             | fp16/fp32 | lora | qlora | gptq | gptq w/flash attn | flash attn | xformers attn |
|-------------|:----------|:-----|-------|------|-------------------|------------|--------------|
| llama       | ✅         | ✅    | ✅     | ✅             | ✅                 | ✅          | ✅            |
| Mistral     | ✅         | ✅    | ✅     | ✅             | ✅                 | ✅          | ✅            |
| Mixtral-MoE | ✅         | ✅    | ✅     | ❓             | ❓                 | ❓          | ❓            |
| Pythia      | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓            |
| cerebras    | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓            |
| btlm        | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓            |
| mpt         | ✅         | ❌    | ❓     | ❌             | ❌                 | ❌          | ❓            |
| falcon      | ✅         | ✅    | ✅     | ❌             | ❌                 | ❌          | ❓            |
| gpt-j       | ✅         | ✅    | ✅     | ❌             | ❌                 | ❓          | ❓            |
| XGen        | ✅         | ❓    | ✅     | ❓             | ❓                 | ❓          | ✅            |
| phi         | ✅         | ✅    | ✅     | ❓             | ❓                 | ❓          | ❓            |
| RWKV        | ✅         | ❓    | ❓     | ❓             | ❓                 | ❓          | ❓            |
| Qwen        | ✅         | ✅    | ✅     | ❓             | ❓                 | ❓          | ❓            |
| Gemma       | ✅         | ✅    | ✅     | ❓             | ❓                 | ✅          | ❓            |

✅: supported
❌: not supported
❓: untested

## Quickstart ⚡

Get started with Axolotl in just a few steps! This quickstart guide will walk you through setting up and running a basic fine-tuning task.

**Requirements**: Python >=3.9 and Pytorch >=2.1.1.

### For developers
```bash
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip3 install packaging
```

General case:
```
pip3 install -e '.[flash-attn,deepspeed]'
```

Mac: see https://github.com/OpenAccess-AI-Collective/axolotl/blob/13199f678b9aab39e92961323bdbce3234ee4b2b/docs/mac.md
```
pip3 install -e '.'
```

### Usage
```bash
# preprocess datasets - optional but recommended
CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess examples/openllama-3b/lora.yml

# finetune lora
accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml

# inference
accelerate launch -m axolotl.cli.inference examples/openllama-3b/lora.yml \
    --lora_model_dir="./lora-out"

# gradio
accelerate launch -m axolotl.cli.inference examples/openllama-3b/lora.yml \
    --lora_model_dir="./lora-out" --gradio

# remote yaml files - the yaml config can be hosted on a public URL
# Note: the yaml config must directly link to the **raw** yaml
accelerate launch -m axolotl.cli.train https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/examples/openllama-3b/lora.yml
```

## Advanced Setup

### Environment

#### Docker

  ```bash
  docker run --gpus '"all"' --rm -it winglian/axolotl:main-latest
  ```

  Or run on the current files for development:

  ```sh
  docker compose up -d
  ```

>[!Tip]
> If you want to debug axolotl or prefer to use Docker as your development environment, see the [debugging guide's section on Docker](docs/debugging.md#debugging-with-docker).

  <details>

  <summary>Docker advanced</summary>

  A more powerful Docker command to run would be this:

  ```bash
docker run --privileged --gpus '"all"' --shm-size 10g --rm -it --name axolotl --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --mount type=bind,src="${PWD}",target=/workspace/axolotl -v ${HOME}/.cache/huggingface:/root/.cache/huggingface winglian/axolotl:main-latest
  ```

  It additionally:
  * Prevents memory issues when running e.g. deepspeed (e.g. you could hit SIGBUS/signal 7 error) through `--ipc` and `--ulimit` args.
  * Persists the downloaded HF data (models etc.) and your modifications to axolotl code through `--mount`/`-v` args.
  * The `--name` argument simply makes it easier to refer to the container in vscode (`Dev Containers: Attach to Running Container...`) or in your terminal.
  * The `--privileged` flag gives all capabilities to the container.
  * The `--shm-size 10g` argument increases the shared memory size. Use this if you see `exitcode: -7` errors using deepspeed.

  [More information on nvidia website](https://docs.nvidia.com/deeplearning/frameworks/user-guide/index.html#setincshmem)

  </details>

#### Conda/Pip venv
  1. Install python >=**3.9**

  2. Install pytorch stable https://pytorch.org/get-started/locally/

  3. Install Axolotl along with python dependencies
        ```bash
        pip3 install packaging
        pip3 install -e '.[flash-attn,deepspeed]'
        ```
  4. (Optional) Login to Huggingface to use gated models/datasets.
        ```bash
        huggingface-cli login
        ```
        Get the token at huggingface.co/settings/tokens

#### Cloud GPU

For cloud GPU providers that support docker images, use [`winglian/axolotl-cloud:main-latest`](https://hub.docker.com/r/winglian/axolotl-cloud/tags)

- on Latitude.sh use this [direct link](https://latitude.sh/blueprint/989e0e79-3bf6-41ea-a46b-1f246e309d5c)
- on RunPod use this [direct link](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)

#### Bare Metal Cloud GPU

##### LambdaLabs

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

  pip3 install packaging
  pip3 install -e '.[flash-attn,deepspeed]'
  pip3 install protobuf==3.20.3
  pip3 install -U --ignore-installed requests Pillow psutil scipy
  ```

  5. Set path
  ```bash
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  ```
  </details>

#### Windows
Please use WSL or Docker!


#### Launching on public clouds via SkyPilot
To launch on GPU instances (both on-demand and spot instances) on 7+ clouds (GCP, AWS, Azure, OCI, and more), you can use [SkyPilot](https://skypilot.readthedocs.io/en/latest/index.html):

```bash
pip install "skypilot-nightly[gcp,aws,azure,oci,lambda,kubernetes,ibm,scp]"  # choose your clouds
sky check
```

Get the [example YAMLs](https://github.com/skypilot-org/skypilot/tree/master/llm/axolotl) of using Axolotl to finetune `mistralai/Mistral-7B-v0.1`:
```
git clone https://github.com/skypilot-org/skypilot.git
cd skypilot/llm/axolotl
```

Use one command to launch:
```bash
# On-demand
HF_TOKEN=xx sky launch axolotl.yaml --env HF_TOKEN

# Managed spot (auto-recovery on preemption)
HF_TOKEN=xx BUCKET=<unique-name> sky spot launch axolotl-spot.yaml --env HF_TOKEN --env BUCKET
```

### Dataset

Axolotl supports a variety of dataset formats. Below are some of the formats you can use.
Have dataset(s) in one of the following format (JSONL recommended):

#### Pretraining

- `completion`: raw corpus
  ```json
  {"text": "..."}
  ```

Note: Axolotl usually loads the entire dataset into memory. This will be challenging for large datasets. Use the following config to enable streaming:

```yaml
pretraining_dataset: # hf path only
```

#### Supervised finetuning

##### Instruction

- `alpaca`: instruction; input(optional)
  ```json
  {"instruction": "...", "input": "...", "output": "..."}
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
- `context_qa.load_v2`: in context question answering (alternate)
  ```json
  {"context": "...", "question": "...", "answer": "..."}
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
- `metharme`: instruction, adds additional eos tokens
  ```json
  {"prompt": "...", "generation": "..."}
  ```

</details>

##### Conversation

- `sharegpt`: conversations where `from` is `human`/`gpt`. (optional: first row with role `system` to override default system prompt)
  ```json
  {"conversations": [{"from": "...", "value": "..."}]}
  ```

<details>

<summary>See other formats</summary>

- `pygmalion`: pygmalion
  ```json
  {"conversations": [{"role": "...", "value": "..."}]}
  ```
- `sharegpt.load_role`: conversations where `role` is used instead of `from`
  ```json
  {"conversations": [{"role": "...", "value": "..."}]}
  ```
- `sharegpt.load_guanaco`: conversations where `from` is `prompter`/`assistant` instead of default sharegpt
  ```json
  {"conversations": [{"from": "...", "value": "..."}]}
  ```
- `sharegpt_jokes`: creates a chat where bot is asked to tell a joke, then explain why the joke is funny
  ```json
  {"conversations": [{"title": "...", "text": "...", "explanation": "..."}]}
  ```

</details>

Note: `type: sharegpt` opens a special config `conversation:` that enables conversions to many Conversation types. See dataset section under [all yaml options](#all-yaml-options).

#### How to add custom prompts

For a dataset that is preprocessed for instruction purposes:

```json
{"input": "...", "output": "..."}
```

You can use this example in your YAML config:

```yaml
datasets:
  - path: repo
    type:
      system_prompt: ""
      field_system: system
      field_instruction: input
      field_output: output
      format: "[INST] {instruction} [/INST]"
      no_input_format: "[INST] {instruction} [/INST]"
```
See full config options under [all yaml options](#all-yaml-options).

#### How to use your custom pretokenized dataset

- Do not pass a `type:`
- Columns in Dataset must be exactly `input_ids`, `attention_mask`, `labels`

```yaml
- path: ...
```

### Config

See [examples](examples) for quick start. It is recommended to duplicate and modify to your needs. The most important options are:

- model
  ```yaml
  base_model: ./llama-7b-hf # local or huggingface repo
  ```
  Note: The code will load the right architecture.

- dataset
  ```yaml
  datasets:
      # huggingface repo
    - path: vicgalle/alpaca-gpt4
      type: alpaca

      # huggingface repo with specific configuration/subset
    - path: EleutherAI/pile
      name: enron_emails
      type: completion # format from earlier
      field: text # Optional[str] default: text, field to use for completion data

      # huggingface repo with multiple named configurations/subsets
    - path: bigcode/commitpackft
      name:
        - ruby
        - python
        - typescript
      type: ... # unimplemented custom format

      # fastchat conversation
      # See 'conversation' options: https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
    - path: ...
      type: sharegpt
      conversation: chatml # default: vicuna_v1.1

      # local
    - path: data.jsonl # or json
      ds_type: json # see other options below
      type: alpaca

      # dataset with splits, but no train split
    - path: knowrohit07/know_sql
      type: context_qa.load_v2
      train_on_split: validation

      # loading from s3 or gcs
      # s3 creds will be loaded from the system default and gcs only supports public access
    - path: s3://path_to_ds # Accepts folder with arrow/parquet or file path like above. Supports s3, gcs.
      ...

      # Loading Data From a Public URL
      # - The file format is `json` (which includes `jsonl`) by default. For different formats, adjust the `ds_type` option accordingly.
    - path: https://some.url.com/yourdata.jsonl # The URL should be a direct link to the file you wish to load. URLs must use HTTPS protocol, not HTTP.
      ds_type: json # this is the default, see other options below.
  ```

- loading
  ```yaml
  load_in_4bit: true
  load_in_8bit: true

  bf16: auto # require >=ampere, auto will detect if your GPU supports this and choose automatically.
  fp16: # leave empty to use fp16 when bf16 is 'auto'. set to false if you want to fallback to fp32
  tf32: true # require >=ampere

  bfloat16: true # require >=ampere, use instead of bf16 when you don't want AMP (automatic mixed precision)
  float16: true # use instead of fp16 when you don't want AMP
  ```
  Note: Repo does not do 4-bit quantization.

- lora
  ```yaml
  adapter: lora # 'qlora' or leave blank for full finetune
  lora_r: 8
  lora_alpha: 16
  lora_dropout: 0.05
  lora_target_modules:
    - q_proj
    - v_proj
  ```

<details id="all-yaml-options">

<summary>All yaml options (click to expand)</summary>

```yaml
# This is the huggingface model that contains *.pt, *.safetensors, or *.bin files
# This can also be a relative path to a model on disk
base_model: ./llama-7b-hf
# You can specify an ignore pattern if the model repo contains more than 1 model type (*.pt, etc)
base_model_ignore_patterns:
# If the base_model repo on hf hub doesn't include configuration .json files,
# You can set that here, or leave this empty to default to base_model
base_model_config: ./llama-7b-hf
# You can specify to choose a specific model revision from huggingface hub
revision_of_model:
# Optional tokenizer configuration path in case you want to use a different tokenizer
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
# Resize the model embeddings when new tokens are added to multiples of 32
# This is reported to improve training speed on some models
resize_token_embeddings_to_32x:

# (Internal use only)
# Used to identify which the model is based on
is_falcon_derived_model:
is_llama_derived_model:
is_qwen_derived_model:
# Please note that if you set this to true, `padding_side` will be set to "left" by default
is_mistral_derived_model:

# optional overrides to the base model configuration
overrides_of_model_config:
  # RoPE Scaling https://github.com/huggingface/transformers/pull/24653
  rope_scaling:
    type: # linear | dynamic
    factor: # float

# optional overrides to the bnb 4bit quantization configuration
# https://huggingface.co/docs/transformers/main/main_classes/quantization#transformers.BitsAndBytesConfig
bnb_config_kwargs:
  # These are default values
  llm_int8_has_fp16_weight: false
  bnb_4bit_quant_type: nf4
  bnb_4bit_use_double_quant: true


# Whether you are training a 4-bit GPTQ quantized model
gptq: true

# This will attempt to quantize the model down to 8 bits and use adam 8 bit optimizer
load_in_8bit: true
# Use bitsandbytes 4 bit
load_in_4bit:

# Use CUDA bf16
bf16: true # bool or 'full' for `bf16_full_eval`. require >=ampere
# Use CUDA fp16
fp16: true
# Use CUDA tf32
tf32: true # require >=ampere

# No AMP (automatic mixed precision)
bfloat16: true # require >=ampere
float16: true

# Limit the memory for all available GPUs to this amount (if an integer, expressed in gigabytes); default: unset
gpu_memory_limit: 20GiB
# Do the LoRA/PEFT loading on CPU -- this is required if the base model is so large it takes up most or all of the available GPU VRAM, e.g. during a model and LoRA merge
lora_on_cpu: true

# A list of one or more datasets to finetune the model with
datasets:
  # HuggingFace dataset repo | s3://,gs:// path | "json" for local dataset, make sure to fill data_files
  - path: vicgalle/alpaca-gpt4
  # The type of prompt to use for training. [alpaca, sharegpt, gpteacher, oasst, reflection]
    type: alpaca # format | format:<prompt_style> (chat/instruct) | <prompt_strategies>.load_<load_fn>
    ds_type: # Optional[str] (json|arrow|parquet|text|csv) defines the datatype when path is a file
    data_files: # Optional[str] path to source data files
    shards: # Optional[int] number of shards to split data into
    name: # Optional[str] name of dataset configuration to load
    train_on_split: train # Optional[str] name of dataset split to load from

    # Optional[str] fastchat conversation type, only used with type: sharegpt
    conversation:  # Options (see Conversation 'name'): https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
    field_human: # Optional[str]. Human key to use for conversation.
    field_model: # Optional[str]. Assistant key to use for conversation.

  # Custom user instruction prompt
  - path: repo
    type:
      # The below are defaults. only set what's needed if you use a different column name.
      system_prompt: ""
      system_format: "{system}"
      field_system: system
      field_instruction: instruction
      field_input: input
      field_output: output

      # Customizable to be single line or multi-line
      # Use {instruction}/{input} as key to be replaced
      # 'format' can include {input}
      format: |-
        User: {instruction} {input}
        Assistant:
      # 'no_input_format' cannot include {input}
      no_input_format: "{instruction} "

      # For `completion` datsets only, uses the provided field instead of `text` column
      field:

# A list of one or more datasets to eval the model with.
# You can use either test_datasets, or val_set_size, but not both.
test_datasets:
  - path: /workspace/data/eval.jsonl
    ds_type: json
    # You need to specify a split. For "json" datasets the default split is called "train".
    split: train
    type: completion
    data_files:
      - /workspace/data/eval.jsonl

# use RL training: 'dpo', 'ipo', 'kto_pair'
rl:

# Saves the desired chat template to the tokenizer_config.json for easier inferencing
# Currently supports chatml and inst (mistral/mixtral)
chat_template: chatml
# Changes the default system message
default_system_message: You are a helpful assistant. Please give a long and detailed answer. # Currently only supports chatml.
# Axolotl attempts to save the dataset as an arrow after packing the data together so
# subsequent training attempts load faster, relative path
dataset_prepared_path: data/last_run_prepared
# Push prepared dataset to hub
push_dataset_to_hub: # repo path
# The maximum number of processes to use while preprocessing your input dataset. This defaults to `os.cpu_count()`
# if not set.
dataset_processes: # defaults to os.cpu_count() if not set
# Keep dataset in memory while preprocessing
# Only needed if cached dataset is taking too much storage
dataset_keep_in_memory:
# push checkpoints to hub
hub_model_id: # private repo path to push finetuned model
# how to push checkpoints to hub
# https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/trainer#transformers.TrainingArguments.hub_strategy
hub_strategy:
# Whether to use hf `use_auth_token` for loading datasets. Useful for fetching private datasets
# Required to be true when used in combination with `push_dataset_to_hub`
hf_use_auth_token: # boolean
# How much of the dataset to set aside as evaluation. 1 = 100%, 0.50 = 50%, etc. 0 for no eval.
val_set_size: 0.04
# Num shards for whole dataset
dataset_shard_num:
# Index of shard to use for whole dataset
dataset_shard_idx:

# The maximum length of an input to train with, this should typically be less than 2048
# as most models have a token/context limit of 2048
sequence_len: 2048
# Pad inputs so each step uses constant sized buffers
# This will reduce memory fragmentation and may prevent OOMs, by re-using memory more efficiently
pad_to_sequence_len:
# Use efficient multi-packing with block diagonal attention and per sequence position_ids. Recommend set to 'true'
sample_packing:
# Set to 'false' if getting errors during eval with sample_packing on.
eval_sample_packing:
# You can set these packing optimizations AFTER starting a training at least once.
# The trainer will provide recommended values for these values.
sample_packing_eff_est:
total_num_tokens:

# Passed through to transformers when loading the model when launched without accelerate
# Use `sequential` when training w/ model parallelism to limit memory
device_map:
# Defines the max memory usage per gpu on the system. Passed through to transformers when loading the model.
max_memory:

# If you want to use 'lora' or 'qlora' or leave blank to train all parameters in original model
adapter: lora
# If you already have a lora model trained that you want to load, put that here.
# This means after training, if you want to test the model, you should set this to the value of `output_dir`.
# Note that if you merge an adapter to the base model, a new subdirectory `merged` will be created under the `output_dir`.
lora_model_dir:

# LoRA hyperparameters
# For more details about the following options, see:
# https://www.anyscale.com/blog/fine-tuning-llms-lora-or-full-parameter-an-in-depth-analysis-with-llama-2
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
lora_target_linear: # If true, will target all linear modules
peft_layers_to_transform: # The layer indices to transform, otherwise, apply to all layers

# If you added new tokens to the tokenizer, you may need to save some LoRA modules because they need to know the new tokens.
# For LLaMA and Mistral, you need to save `embed_tokens` and `lm_head`. It may vary for other models.
# `embed_tokens` converts tokens to embeddings, and `lm_head` converts embeddings to token probabilities.
# https://github.com/huggingface/peft/issues/334#issuecomment-1561727994
lora_modules_to_save:
#  - embed_tokens
#  - lm_head

lora_fan_in_fan_out: false

peft:
  # Configuration options for loftq initialization for LoRA
  # https://huggingface.co/docs/peft/developer_guides/quantization#loftq-initialization
  loftq_config:
    loftq_bits:  # typically 4 bits

# ReLoRA configuration
# Must use either 'lora' or 'qlora' adapter, and does not support fsdp or deepspeed
relora_steps: # Number of steps per ReLoRA restart
relora_warmup_steps: # Number of per-restart warmup steps
relora_anneal_steps: # Number of anneal steps for each relora cycle
relora_prune_ratio: # threshold for optimizer magnitude when pruning
relora_cpu_offload: # True to perform lora weight merges on cpu during restarts, for modest gpu memory savings

# wandb configuration if you're using it
# Make sure your `WANDB_API_KEY` environment variable is set (recommended) or you login to wandb with `wandb login`.
wandb_mode: # "offline" to save run metadata locally and not sync to the server, "disabled" to turn off wandb
wandb_project: # Your wandb project name
wandb_entity: # A wandb Team name if using a Team
wandb_watch:
wandb_name: # Set the name of your wandb run
wandb_run_id: # Set the ID of your wandb run
wandb_log_model: # "checkpoint" to log model to wandb Artifacts every `save_steps` or "end" to log only at the end of training

# mlflow configuration if you're using it
mlflow_tracking_uri: # URI to mlflow
mlflow_experiment_name: # Your experiment name
hf_mlflow_log_artifacts:  # set to true to copy each saved checkpoint on each save to mlflow artifact registry

# Where to save the full-finetuned model to
output_dir: ./completed-model

# Whether to use torch.compile and which backend to use
torch_compile:  # bool
torch_compile_backend:  # Optional[str]

# Training hyperparameters

# If greater than 1, backpropagation will be skipped and the gradients will be accumulated for the given number of steps.
gradient_accumulation_steps: 1
# The number of samples to include in each batch. This is the number of samples sent to each GPU.
micro_batch_size: 2
eval_batch_size:
num_epochs: 4
warmup_steps: 100  # cannot use with warmup_ratio
warmup_ratio: 0.05  # cannot use with warmup_steps
learning_rate: 0.00003
lr_quadratic_warmup:
logging_steps:
eval_steps: # Leave empty to eval at each epoch, integers for every N steps. decimal for fraction of total steps
evals_per_epoch: # number of times per epoch to run evals, mutually exclusive with eval_steps
save_strategy: # Set to `no` to skip checkpoint saves
save_steps: # Leave empty to save at each epoch
saves_per_epoch: # number of times per epoch to save a checkpoint, mutually exclusive with save_steps
save_total_limit: # Checkpoints saved at a time
# Maximum number of iterations to train for. It precedes num_epochs which means that
# if both are set, num_epochs will not be guaranteed.
# e.g., when 1 epoch is 1000 steps => `num_epochs: 2` and `max_steps: 100` will train for 100 steps
max_steps:

eval_table_size: # Approximate number of predictions sent to wandb depending on batch size. Enabled above 0. Default is 0
eval_max_new_tokens: # Total number of tokens generated for predictions sent to wandb. Default is 128
eval_causal_lm_metrics: # HF evaluate metrics used during evaluation. Default is ["sacrebleu", "comet", "ter", chrf]

loss_watchdog_threshold: # High loss value, indicating the learning has broken down (a good estimate is ~2 times the loss at the start of training)
loss_watchdog_patience: # Number of high-loss steps in a row before the trainer aborts (default: 3)

# Save model as safetensors (require safetensors package)
save_safetensors:

# Whether to mask out or include the human's prompt from the training labels
train_on_inputs: false
# Group similarly sized data to minimize padding.
# May be slower to start, as it must download and sort the entire dataset.
# Note that training loss may have an oscillating pattern with this enabled.
group_by_length: false

# Whether to use gradient checkpointing https://huggingface.co/docs/transformers/v4.18.0/en/performance#gradient-checkpointing
gradient_checkpointing: false
# additional kwargs to pass to the trainer for gradient checkpointing
# gradient_checkpointing_kwargs:
#   use_reentrant: false

# Stop training after this many evaluation losses have increased in a row
# https://huggingface.co/transformers/v4.2.2/_modules/transformers/trainer_callback.html#EarlyStoppingCallback
early_stopping_patience: 3

# Specify a scheduler and kwargs to use with the optimizer
lr_scheduler: # 'one_cycle' | 'log_sweep' | empty for cosine
lr_scheduler_kwargs:
cosine_min_lr_ratio: # decay lr to some percentage of the peak lr, e.g. cosine_min_lr_ratio=0.1 for 10% of peak lr
cosine_constant_lr_ratio: # freeze lr at some percentage of the step, e.g. cosine_constant_lr_ratio=0.8 means start cosine_min_lr at 80% of training step (https://arxiv.org/pdf/2308.04014.pdf)

# For one_cycle optim
lr_div_factor: # Learning rate div factor

# Specify optimizer
# Valid values are driven by the Transformers OptimizerNames class, see:
# https://github.com/huggingface/transformers/blob/95b374952dc27d8511541d6f5a4e22c9ec11fb24/src/transformers/training_args.py#L134
#
# Note that not all optimizers may be available in your environment, ex: 'adamw_anyprecision' is part of
# torchdistx, 'adamw_bnb_8bit' is part of bnb.optim.Adam8bit, etc. When in doubt, it is recommended to start with the optimizer used
# in the examples/ for your model and fine-tuning use case.
#
# Valid values for 'optimizer' include:
# - adamw_hf
# - adamw_torch
# - adamw_torch_fused
# - adamw_torch_xla
# - adamw_apex_fused
# - adafactor
# - adamw_anyprecision
# - sgd
# - adagrad
# - adamw_bnb_8bit
# - lion_8bit
# - lion_32bit
# - paged_adamw_32bit
# - paged_adamw_8bit
# - paged_lion_32bit
# - paged_lion_8bit
optimizer:
# Specify weight decay
weight_decay:
# adamw hyperparams
adam_beta1:
adam_beta2:
adam_epsilon:
# Gradient clipping max norm
max_grad_norm:

# Augmentation techniques
# NEFT https://arxiv.org/abs/2310.05914, set this to a number (paper default is 5) to add noise to embeddings
# currently only supported on Llama and Mistral
neftune_noise_alpha:

# Whether to bettertransformers
flash_optimum:
# Whether to use xformers attention patch https://github.com/facebookresearch/xformers:
xformers_attention:
# Whether to use flash attention patch https://github.com/Dao-AILab/flash-attention:
flash_attention:
flash_attn_cross_entropy:  # Whether to use flash-attention cross entropy implementation - advanced use only
flash_attn_rms_norm:  # Whether to use flash-attention rms norm implementation - advanced use only
flash_attn_fuse_qkv: # Whether to fuse QKV into a single operation
flash_attn_fuse_mlp: # Whether to fuse part of the MLP into a single operation
# Whether to use scaled-dot-product attention
# https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
sdp_attention:
# Shifted-sparse attention (only llama) - https://arxiv.org/pdf/2309.12307.pdf
s2_attention:
# Resume from a specific checkpoint dir
resume_from_checkpoint:
# If resume_from_checkpoint isn't set and you simply want it to start where it left off.
# Be careful with this being turned on between different models.
auto_resume_from_checkpoints: false

# Don't mess with this, it's here for accelerate and torchrun
local_rank:

# Add or change special tokens.
# If you add tokens here, you don't need to add them to the `tokens` list.
special_tokens:
  # bos_token: "<s>"
  # eos_token: "</s>"
  # unk_token: "<unk>"

# Add extra tokens.
tokens:

# FSDP
fsdp:
fsdp_config:

# Deepspeed config path. e.g., deepspeed_configs/zero3.json
deepspeed:

# Advanced DDP Arguments
ddp_timeout:
ddp_bucket_cap_mb:
ddp_broadcast_buffers:

# Path to torch distx for optim 'adamw_anyprecision'
torchdistx_path:

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

<details>
<summary> Understanding of batch size and gradient accumulation steps </summary>
<br/>
Gradient accumulation means accumulating gradients over several mini-batches and updating the model weights afterward. When the samples in each batch are diverse, this technique doesn't significantly impact learning.

This method allows for effective training with larger effective batch sizes without needing proportionally larger memory. Here's why:

1. **Memory Consumption with Batch Size**: The primary reason increasing the batch size impacts memory is due to the storage requirements for intermediate activations. When you forward propagate a batch through a network, you have to store the activations at each layer for each sample in the batch, because these activations are used during backpropagation to compute gradients. Therefore, larger batches mean more activations, leading to greater GPU memory consumption.

2. **Gradient Accumulation**: With gradient accumulation, you're effectively simulating a larger batch size by accumulating gradients over several smaller batches (or micro-batches). However, at any given time, you're only forward and backward propagating a micro-batch. This means you only store activations for the micro-batch, not the full accumulated batch. As a result, you can simulate the effect of a larger batch size without the memory cost of storing activations for a large batch.

**Example 1:**
Micro batch size: 3
Gradient accumulation steps: 2
Number of GPUs: 3
Total batch size = 3 * 2 * 3 = 18

```
| GPU 1          | GPU 2          | GPU 3          |
|----------------|----------------|----------------|
| S1, S2, S3     | S4, S5, S6     | S7, S8, S9     |
| e1, e2, e3     | e4, e5, e6     | e7, e8, e9     |
|----------------|----------------|----------------|
| → (accumulate) | → (accumulate) | → (accumulate) |
|----------------|----------------|----------------|
| S10, S11, S12  | S13, S14, S15  | S16, S17, S18  |
| e10, e11, e12  | e13, e14, e15  | e16, e17, e18  |
|----------------|----------------|----------------|
| → (apply)      | → (apply)      | → (apply)      |

Accumulated gradient for the weight w1 after the second iteration (considering all GPUs):
Total gradient for w1 = e1 + e2 + e3 + e4 + e5 + e6 + e7 + e8 + e9 + e10 + e11 + e12 + e13 + e14 + e15 + e16 + e17 + e18

Weight update for w1:
w1_new = w1_old - learning rate x (Total gradient for w1 / 18)
```

**Example 2:**
Micro batch size: 2
Gradient accumulation steps: 1
Number of GPUs: 3
Total batch size = 2 * 1 * 3 = 6

```
| GPU 1     | GPU 2     | GPU 3     |
|-----------|-----------|-----------|
| S1, S2    | S3, S4    | S5, S6    |
| e1, e2    | e3, e4    | e5, e6    |
|-----------|-----------|-----------|
| → (apply) | → (apply) | → (apply) |

Accumulated gradient for the weight w1 (considering all GPUs):
Total gradient for w1 = e1 + e2 + e3 + e4 + e5 + e6

Weight update for w1:
w1_new = w1_old - learning rate × (Total gradient for w1 / 6)
```

</details>

### Train

Run
```bash
accelerate launch -m axolotl.cli.train your_config.yml
```

> [!TIP]
> You can also reference a config file that is hosted on a public URL, for example `accelerate launch -m axolotl.cli.train https://yourdomain.com/your_config.yml`

#### Preprocess dataset

You can optionally pre-tokenize dataset with the following before finetuning.
This is recommended for large datasets.

- Set `dataset_prepared_path:` to a local folder for saving and loading pre-tokenized dataset.
- (Optional): Set `push_dataset_to_hub: hf_user/repo` to push it to Huggingface.
- (Optional): Use `--debug` to see preprocessed examples.

```bash
python -m axolotl.cli.preprocess your_config.yml
```

#### Multi-GPU

Below are the options available in axolotl for training with multiple GPUs. Note that DeepSpeed
is the recommended multi-GPU option currently because FSDP may experience
[loss instability](https://github.com/huggingface/transformers/issues/26498).

##### DeepSpeed

Deepspeed is an optimization suite for multi-gpu systems allowing you to train much larger models than you
might typically be able to fit into your GPU's VRAM. More information about the various optimization types
for deepspeed is available at https://huggingface.co/docs/accelerate/main/en/usage_guides/deepspeed#what-is-integrated

We provide several default deepspeed JSON configurations for ZeRO stage 1, 2, and 3.

```yaml
deepspeed: deepspeed_configs/zero1.json
```

```shell
accelerate launch -m axolotl.cli.train examples/llama-2/config.py --deepspeed deepspeed_configs/zero1.json
```

##### FSDP

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

##### Weights & Biases Logging

Make sure your `WANDB_API_KEY` environment variable is set (recommended) or you login to wandb with `wandb login`.

- wandb options
```yaml
wandb_mode:
wandb_project:
wandb_entity:
wandb_watch:
wandb_name:
wandb_log_model:
```

##### Special Tokens

It is important to have special tokens like delimiters, end-of-sequence, beginning-of-sequence in your tokenizer's vocabulary.  This will help you avoid tokenization issues and help your model train better.  You can do this in axolotl like this:

```yml
special_tokens:
  bos_token: "<s>"
  eos_token: "</s>"
  unk_token: "<unk>"
tokens: # these are delimiters
  - "<|im_start|>"
  - "<|im_end|>"
```

When you include these tokens in your axolotl config, axolotl adds these tokens to the tokenizer's vocabulary.

### Inference Playground

Axolotl allows you to load your model in an interactive terminal playground for quick experimentation.
The config file is the same config file used for training.

Pass the appropriate flag to the inference command, depending upon what kind of model was trained:

- Pretrained LORA:
  ```bash
  python -m axolotl.cli.inference examples/your_config.yml --lora_model_dir="./lora-output-dir"
  ```
- Full weights finetune:
  ```bash
  python -m axolotl.cli.inference examples/your_config.yml --base_model="./completed-model"
  ```
- Full weights finetune w/ a prompt from a text file:
  ```bash
  cat /tmp/prompt.txt | python -m axolotl.cli.inference examples/your_config.yml \
    --base_model="./completed-model" --prompter=None --load_in_8bit=True
  ```
-- With gradio hosting
  ```bash
  python -m axolotl.cli.inference examples/your_config.yml --gradio
  ```

Please use `--sample_packing False` if you have it on and receive the error similar to below:

> RuntimeError: stack expects each tensor to be equal size, but got [1, 32, 1, 128] at entry 0 and [1, 32, 8, 128] at entry 1

### Merge LORA to base

The following command will merge your LORA adapater with your base model. You can optionally pass the argument `--lora_model_dir` to specify the directory where your LORA adapter was saved, otherwhise, this will be inferred from `output_dir` in your axolotl config file.  The merged model is saved in the sub-directory `{lora_model_dir}/merged`.

```bash
python3 -m axolotl.cli.merge_lora your_config.yml --lora_model_dir="./completed-model"
```

You may need to use the `gpu_memory_limit` and/or `lora_on_cpu` config options to avoid running out of memory. If you still run out of CUDA memory, you can try to merge in system RAM with

```bash
CUDA_VISIBLE_DEVICES="" python3 -m axolotl.cli.merge_lora ...
```

although this will be very slow, and using the config options above are recommended instead.

## Common Errors 🧰

See also the [FAQ's](./docs/faq.md) and [debugging guide](docs/debugging.md).

> If you encounter a 'Cuda out of memory' error, it means your GPU ran out of memory during the training process. Here's how to resolve it:

Please reduce any below
  - `micro_batch_size`
  - `eval_batch_size`
  - `gradient_accumulation_steps`
  - `sequence_len`

If it does not help, try running without deepspeed and without accelerate (replace "accelerate launch" with "python") in the command.

Using adamw_bnb_8bit might also save you some memory.

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

> NCCL Timeouts during training

See the [NCCL](docs/nccl.md) guide.


### Tokenization Mismatch b/w Inference & Training

For many formats, Axolotl constructs prompts by concatenating token ids _after_ tokenizing strings.  The reason for concatenating token ids rather than operating on strings is to maintain precise accounting for attention masks.

If you decode a prompt constructed by axolotl, you might see spaces between tokens (or lack thereof) that you do not expect, especially around delimiters and special tokens.  When you are starting out with a new format, you should always do the following:

1. Materialize some data using `python -m axolotl.cli.preprocess your_config.yml --debug`, and then decode the first few rows with your model's tokenizer.
2. During inference, right before you pass a tensor of token ids to your model, decode these tokens back into a string.
3. Make sure the inference string from #2 looks **exactly** like the data you fine tuned on from #1, including spaces and new lines.  If they aren't the same, adjust your inference server accordingly.
4. As an additional troubleshooting step, you can look at the token ids between 1 and 2 to make sure they are identical.

Having misalignment between your prompts during training and inference can cause models to perform very poorly, so it is worth checking this.  See [this blog post](https://hamel.dev/notes/llm/05_tokenizer_gotchas.html) for a concrete example.

## Debugging Axolotl

See [this debugging guide](docs/debugging.md) for tips on debugging Axolotl, along with an example configuration for debugging with VSCode.

## Need help? 🙋

Join our [Discord server](https://discord.gg/HhrNrHJPRb) where we our community members can help you.

Need dedicated support? Please contact us at [✉️wing@openaccessaicollective.org](mailto:wing@openaccessaicollective.org) for dedicated support options.

## Badge ❤🏷️

Building something cool with Axolotl? Consider adding a badge to your model card.

```markdown
[<img src="https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/OpenAccess-AI-Collective/axolotl)
```

[<img src="https://raw.githubusercontent.com/OpenAccess-AI-Collective/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/OpenAccess-AI-Collective/axolotl)

## Community Showcase

Check out some of the projects and models that have been built using Axolotl! Have a model you'd like to add to our Community Showcase? Open a PR with your model.

Open Access AI Collective
- [Minotaur 13b](https://huggingface.co/openaccess-ai-collective/minotaur-13b-fixed)
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
git clone https://github.com/OpenAccess-AI-Collective/axolotl
cd axolotl

pip3 install packaging
pip3 install -e '.[flash-attn,deepspeed]'

pip3 install -r requirements-dev.txt -r requirements-tests.txt
pre-commit install

# test
pytest tests/

# optional: run against all files
pre-commit run --all-files
```

Thanks to all of our contributors to date. Help drive open source AI progress forward by contributing to Axolotl.

<a href="https://github.com/openaccess-ai-collective/axolotl/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=openaccess-ai-collective/axolotl" alt="contributor chart by https://contrib.rocks"/>
</a>

## Sponsors 🤝❤

OpenAccess AI Collective is run by volunteer contributors such as [winglian](https://github.com/winglian),
[NanoCode012](https://github.com/NanoCode012), [tmm1](https://github.com/tmm1),
[mhenrichsen](https://github.com/mhenrichsen), [casper-hansen](https://github.com/casper-hansen),
[hamelsmu](https://github.com/hamelsmu) and many more who help us accelerate forward by fixing bugs, answering
community questions and implementing new features. Axolotl needs donations from sponsors for the compute needed to
run our unit & integration tests, troubleshooting community issues, and providing bounties. If you love axolotl,
consider sponsoring the project via [GitHub Sponsors](https://github.com/sponsors/OpenAccess-AI-Collective),
[Ko-fi](https://ko-fi.com/axolotl_ai) or reach out directly to
[wing@openaccessaicollective.org](mailto:wing@openaccessaicollective.org).

---

#### 💎 Diamond Sponsors - [Contact directly](mailto:wing@openaccessaicollective.org)

---

#### 🥇 Gold Sponsors - $5000/mo

---

#### 🥈 Silver Sponsors - $1000/mo

---

#### 🥉 Bronze Sponsors - $500/mo

---
