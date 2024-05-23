# Overview

This is an example of CodeLLaMA configuration for 7b, 13b and 34b.

The 7b variant fits on any 24GB VRAM GPU and will take up about 17 GB of VRAM during training if using qlora and 20 GB if using lora. On a RTX 4090 it trains 3 epochs of the default dataset in about 15 minutes.

The 13b variant will fit if you change these settings to these values:
gradient_accumulation_steps: 2
micro_batch_size: 1

The 34b variant does not fit on 24GB of VRAM - you will need something with +40 gb VRAM that also supports flash attention v2 - A6000 or A100 are good choices.

```shell
accelerate launch scripts/finetune.py examples/code-llama/[MODEL_SIZE]/qlora.yml

```
or

```shell
accelerate launch scripts/finetune.py examples/code-llama/[MODEL_SIZE]/lora.yml

```
