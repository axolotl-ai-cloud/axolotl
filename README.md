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

<a href="https://www.phorm.ai/query?projectId=e315ba4a-4e14-421f-ab05-38a1f9076f25">
  <img alt="phorm.ai" src="https://img.shields.io/badge/Phorm-Ask_AI-%23F2777A.svg?&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iNSIgaGVpZ2h0PSI0IiBmaWxsPSJub25lIiB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciPgogIDxwYXRoIGQ9Ik00LjQzIDEuODgyYTEuNDQgMS40NCAwIDAgMS0uMDk4LjQyNmMtLjA1LjEyMy0uMTE1LjIzLS4xOTIuMzIyLS4wNzUuMDktLjE2LjE2NS0uMjU1LjIyNmExLjM1MyAxLjM1MyAwIDAgMS0uNTk1LjIxMmMtLjA5OS4wMTItLjE5Mi4wMTQtLjI3OS4wMDZsLTEuNTkzLS4xNHYtLjQwNmgxLjY1OGMuMDkuMDAxLjE3LS4xNjkuMjQ2LS4xOTFhLjYwMy42MDMgMCAwIDAgLjItLjEwNi41MjkuNTI5IDAgMCAwIC4xMzgtLjE3LjY1NC42NTQgMCAwIDAgLjA2NS0uMjRsLjAyOC0uMzJhLjkzLjkzIDAgMCAwLS4wMzYtLjI0OS41NjcuNTY3IDAgMCAwLS4xMDMtLjIuNTAyLjUwMiAwIDAgMC0uMTY4LS4xMzguNjA4LjYwOCAwIDAgMC0uMjQtLjA2N0wyLjQzNy43MjkgMS42MjUuNjcxYS4zMjIuMzIyIDAgMCAwLS4yMzIuMDU4LjM3NS4zNzUgMCAwIDAtLjExNi4yMzJsLS4xMTYgMS40NS0uMDU4LjY5Ny0uMDU4Ljc1NEwuNzA1IDRsLS4zNTctLjA3OUwuNjAyLjkwNkMuNjE3LjcyNi42NjMuNTc0LjczOS40NTRhLjk1OC45NTggMCAwIDEgLjI3NC0uMjg1Ljk3MS45NzEgMCAwIDEgLjMzNy0uMTRjLjExOS0uMDI2LjIyNy0uMDM0LjMyNS0uMDI2TDMuMjMyLjE2Yy4xNTkuMDE0LjMzNi4wMy40NTkuMDgyYTEuMTczIDEuMTczIDAgMCAxIC41NDUuNDQ3Yy4wNi4wOTQuMTA5LjE5Mi4xNDQuMjkzYTEuMzkyIDEuMzkyIDAgMCAxIC4wNzguNThsLS4wMjkuMzJaIiBmaWxsPSIjRjI3NzdBIi8+CiAgPHBhdGggZD0iTTQuMDgyIDIuMDA3YTEuNDU1IDEuNDU1IDAgMCAxLS4wOTguNDI3Yy0uMDUuMTI0LS4xMTQuMjMyLS4xOTIuMzI0YTEuMTMgMS4xMyAwIDAgMS0uMjU0LjIyNyAxLjM1MyAxLjM1MyAwIDAgMS0uNTk1LjIxNGMtLjEuMDEyLS4xOTMuMDE0LS4yOC4wMDZsLTEuNTYtLjEwOC4wMzQtLjQwNi4wMy0uMzQ4IDEuNTU5LjE1NGMuMDkgMCAuMTczLS4wMS4yNDgtLjAzM2EuNjAzLjYwMyAwIDAgMCAuMi0uMTA2LjUzMi41MzIgMCAwIDAgLjEzOS0uMTcyLjY2LjY2IDAgMCAwIC4wNjQtLjI0MWwuMDI5LS4zMjFhLjk0Ljk0IDAgMCAwLS4wMzYtLjI1LjU3LjU3IDAgMCAwLS4xMDMtLjIwMi41MDIuNTAyIDAgMCAwLS4xNjgtLjEzOC42MDUuNjA1IDAgMCAwLS4yNC0uMDY3TDEuMjczLjgyN2MtLjA5NC0uMDA4LS4xNjguMDEtLjIyMS4wNTUtLjA1My4wNDUtLjA4NC4xMTQtLjA5Mi4yMDZMLjcwNSA0IDAgMy45MzhsLjI1NS0yLjkxMUExLjAxIDEuMDEgMCAwIDEgLjM5My41NzIuOTYyLjk2MiAwIDAgMSAuNjY2LjI4NmEuOTcuOTcgMCAwIDEgLjMzOC0uMTRDMS4xMjIuMTIgMS4yMy4xMSAxLjMyOC4xMTlsMS41OTMuMTRjLjE2LjAxNC4zLjA0Ny40MjMuMWExLjE3IDEuMTcgMCAwIDEgLjU0NS40NDhjLjA2MS4wOTUuMTA5LjE5My4xNDQuMjk1YTEuNDA2IDEuNDA2IDAgMCAxIC4wNzcuNTgzbC0uMDI4LjMyMloiIGZpbGw9IndoaXRlIi8+CiAgPHBhdGggZD0iTTQuMDgyIDIuMDA3YTEuNDU1IDEuNDU1IDAgMCAxLS4wOTguNDI3Yy0uMDUuMTI0LS4xMTQuMjMyLS4xOTIuMzI0YTEuMTMgMS4xMyAwIDAgMS0uMjU0LjIyNyAxLjM1MyAxLjM1MyAwIDAgMS0uNTk1LjIxNGMtLjEuMDEyLS4xOTMuMDE0LS4yOC4wMDZsLTEuNTYtLjEwOC4wMzQtLjQwNi4wMy0uMzQ4IDEuNTU5LjE1NGMuMDkgMCAuMTczLS4wMS4yNDgtLjAzM2EuNjAzLjYwMyAwIDAgMCAuMi0uMTA2LjUzMi41MzIgMCAwIDAgLjEzOS0uMTcyLjY2LjY2IDAgMCAwIC4wNjQtLjI0MWwuMDI5LS4zMjFhLjk0Ljk0IDAgMCAwLS4wMzYtLjI1LjU3LjU3IDAgMCAwLS4xMDMtLjIwMi41MDIuNTAyIDAgMCAwLS4xNjgtLjEzOC42MDUuNjA1IDAgMCAwLS4yNC0uMDY3TDEuMjczLjgyN2MtLjA5NC0uMDA4LS4xNjguMDEtLjIyMS4wNTUtLjA1My4wNDUtLjA4NC4xMTQtLjA5Mi4yMDZMLjcwNSA0IDAgMy45MzhsLjI1NS0yLjkxMUExLjAxIDEuMDEgMCAwIDEgLjM5My41NzIuOTYyLjk2MiAwIDAgMSAuNjY2LjI4NmEuOTcuOTcgMCAwIDEgLjMzOC0uMTRDMS4xMjIuMTIgMS4yMy4xMSAxLjMyOC4xMTlsMS41OTMuMTRjLjE2LjAxNC4zLjA0Ny40MjMuMWExLjE3IDEuMTcgMCAwIDEgLjU0NS40NDhjLjA2MS4wOTUuMTA5LjE5My4xNDQuMjk1YTEuNDA2IDEuNDA2IDAgMCAxIC4wNzcuNTgzbC0uMDI4LjMyMloiIGZpbGw9IndoaXRlIi8+Cjwvc3ZnPgo=">
</a>

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
  - [Cloud GPU](#cloud-gpu) - Latitude.sh, JarvisLabs, RunPod
  - [Bare Metal Cloud GPU](#bare-metal-cloud-gpu)
  - [Windows](#windows)
  - [Mac](#mac)
  - [Google Colab](#google-colab)
  - [Launching on public clouds via SkyPilot](#launching-on-public-clouds-via-skypilot)
  - [Launching on public clouds via dstack](#launching-on-public-clouds-via-dstack)
- [Dataset](#dataset)
- [Config](#config)
  - [Train](#train)
  - [Inference](#inference-playground)
  - [Merge LORA to Base](#merge-lora-to-base)
  - [Special Tokens](#special-tokens)
  - [All Config Options](#all-config-options)
- Advanced Topics
  - [Multipack](./docs/multipack.qmd)<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#666" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>
  - [RLHF & DPO](./docs/rlhf.qmd)<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#666" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>
  - [Dataset Pre-Processing](./docs/dataset_preprocessing.qmd)<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#666" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>
  - [Unsloth](./docs/unsloth.qmd)<svg width="24" height="24" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#666" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path></svg>
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
    <img src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/pre-commit.yml/badge.svg?branch=main" alt="pre-commit">
    <img alt="PyTest Status" src="https://github.com/axolotl-ai-cloud/axolotl/actions/workflows/tests.yml/badge.svg?branch=main">
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
| Mixtral8X22 | ✅         | ✅    | ✅     | ❓             | ❓                 | ❓          | ❓            |
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

**Requirements**: Python >=3.10 and Pytorch >=2.1.1.

```bash
git clone https://github.com/axolotl-ai-cloud/axolotl
cd axolotl

pip3 install packaging ninja
pip3 install -e '.[flash-attn,deepspeed]'
```

### Usage
```bash
# preprocess datasets - optional but recommended
CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess examples/openllama-3b/lora.yml

# finetune lora
accelerate launch -m axolotl.cli.train examples/openllama-3b/lora.yml

# inference
accelerate launch -m axolotl.cli.inference examples/openllama-3b/lora.yml \
    --lora_model_dir="./outputs/lora-out"

# gradio
accelerate launch -m axolotl.cli.inference examples/openllama-3b/lora.yml \
    --lora_model_dir="./outputs/lora-out" --gradio

# remote yaml files - the yaml config can be hosted on a public URL
# Note: the yaml config must directly link to the **raw** yaml
accelerate launch -m axolotl.cli.train https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/examples/openllama-3b/lora.yml
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
> If you want to debug axolotl or prefer to use Docker as your development environment, see the [debugging guide's section on Docker](docs/debugging.qmd#debugging-with-docker).

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
  1. Install python >=**3.10**

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
- on JarvisLabs.ai use this [direct link](https://jarvislabs.ai/templates/axolotl)
- on RunPod use this [direct link](https://runpod.io/gsc?template=v2ickqhz9s&ref=6i7fkpdz)

#### Bare Metal Cloud GPU

##### LambdaLabs

  <details>

  <summary>Click to Expand</summary>

  1. Install python
  ```bash
  sudo apt update
  sudo apt install -y python3.10

  sudo update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1
  sudo update-alternatives --config python # pick 3.10 if given option
  python -V # should be 3.10

  ```

  2. Install pip
  ```bash
  wget https://bootstrap.pypa.io/get-pip.py
  python get-pip.py
  ```

  3. Install Pytorch https://pytorch.org/get-started/locally/

  4. Follow instructions on quickstart.

  5. Run
  ```bash
  pip3 install protobuf==3.20.3
  pip3 install -U --ignore-installed requests Pillow psutil scipy
  ```

  6. Set path
  ```bash
  export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
  ```
  </details>

##### GCP

<details>

<summary>Click to Expand</summary>

Use a Deeplearning linux OS with cuda and pytorch installed. Then follow instructions on quickstart.

Make sure to run the below to uninstall xla.
```bash
pip uninstall -y torch_xla[tpu]
```

</details>

#### Windows
Please use WSL or Docker!

#### Mac

Use the below instead of the install method in QuickStart.
```
pip3 install -e '.'
```
More info: [mac.md](/docs/mac.qmd)

#### Google Colab

Please use this example [notebook](examples/colab-notebooks/colab-axolotl-example.ipynb).

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

#### Launching on public clouds via dstack
To launch on GPU instance (both on-demand and spot instances) on public clouds (GCP, AWS, Azure, Lambda Labs, TensorDock, Vast.ai, and CUDO), you can use [dstack](https://dstack.ai/).

Write a job description in YAML as below:

```yaml
# dstack.yaml
type: task

image: winglian/axolotl-cloud:main-20240429-py3.11-cu121-2.2.2

env:
  - HUGGING_FACE_HUB_TOKEN
  - WANDB_API_KEY

commands:
  - accelerate launch -m axolotl.cli.train config.yaml

ports:
  - 6006

resources:
  gpu:
    memory: 24GB..
    count: 2
```

then, simply run the job with `dstack run` command. Append `--spot` option if you want spot instance. `dstack run` command will show you the instance with cheapest price across multi cloud services:

```bash
pip install dstack
HUGGING_FACE_HUB_TOKEN=xxx WANDB_API_KEY=xxx dstack run . -f dstack.yaml # --spot
```

For further and fine-grained use cases, please refer to the official [dstack documents](https://dstack.ai/docs/) and the detailed description of [axolotl example](https://github.com/dstackai/dstack/tree/master/examples/fine-tuning/axolotl) on the official repository.

### Dataset

Axolotl supports a variety of dataset formats.  It is recommended to use a JSONL.  The schema of the JSONL depends upon the task and the prompt template you wish to use.  Instead of a JSONL, you can also use a HuggingFace dataset with columns for each JSONL field.

See [these docs](https://axolotl-ai-cloud.github.io/axolotl/docs/dataset-formats/) for more information on how to use different dataset formats.

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

#### All Config Options

See [these docs](docs/config.qmd) for all config options.

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
accelerate launch -m axolotl.cli.train examples/llama-2/config.yml --deepspeed deepspeed_configs/zero1.json
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

##### FSDP + QLoRA

Axolotl supports training with FSDP and QLoRA, see [these docs](docs/fsdp_qlora.qmd) for more information.

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

See also the [FAQ's](./docs/faq.qmd) and [debugging guide](docs/debugging.qmd).

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

See the [NCCL](docs/nccl.qmd) guide.


### Tokenization Mismatch b/w Inference & Training

For many formats, Axolotl constructs prompts by concatenating token ids _after_ tokenizing strings.  The reason for concatenating token ids rather than operating on strings is to maintain precise accounting for attention masks.

If you decode a prompt constructed by axolotl, you might see spaces between tokens (or lack thereof) that you do not expect, especially around delimiters and special tokens.  When you are starting out with a new format, you should always do the following:

1. Materialize some data using `python -m axolotl.cli.preprocess your_config.yml --debug`, and then decode the first few rows with your model's tokenizer.
2. During inference, right before you pass a tensor of token ids to your model, decode these tokens back into a string.
3. Make sure the inference string from #2 looks **exactly** like the data you fine tuned on from #1, including spaces and new lines.  If they aren't the same, adjust your inference server accordingly.
4. As an additional troubleshooting step, you can look at the token ids between 1 and 2 to make sure they are identical.

Having misalignment between your prompts during training and inference can cause models to perform very poorly, so it is worth checking this.  See [this blog post](https://hamel.dev/notes/llm/finetuning/05_tokenizer_gotchas.html) for a concrete example.

## Debugging Axolotl

See [this debugging guide](docs/debugging.qmd) for tips on debugging Axolotl, along with an example configuration for debugging with VSCode.

## Need help? 🙋

Join our [Discord server](https://discord.gg/HhrNrHJPRb) where we our community members can help you.

Need dedicated support? Please contact us at [✉️wing@openaccessaicollective.org](mailto:wing@openaccessaicollective.org) for dedicated support options.

## Badge ❤🏷️

Building something cool with Axolotl? Consider adding a badge to your model card.

```markdown
[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)
```

[<img src="https://raw.githubusercontent.com/axolotl-ai-cloud/axolotl/main/image/axolotl-badge-web.png" alt="Built with Axolotl" width="200" height="32"/>](https://github.com/axolotl-ai-cloud/axolotl)

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

Bugs? Please check the [open issues](https://github.com/axolotl-ai-cloud/axolotl/issues/bug) else create a new Issue.

PRs are **greatly welcome**!

Please run the quickstart instructions followed by the below to setup env:
```bash
pip3 install -r requirements-dev.txt -r requirements-tests.txt
pre-commit install

# test
pytest tests/

# optional: run against all files
pre-commit run --all-files
```

Thanks to all of our contributors to date. Help drive open source AI progress forward by contributing to Axolotl.

<a href="https://github.com/axolotl-ai-cloud/axolotl/graphs/contributors">
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

 - [JarvisLabs.ai](https://jarvislabs.ai)

---
