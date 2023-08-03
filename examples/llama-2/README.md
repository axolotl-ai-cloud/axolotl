# Overview

This is an example of a llama-2 configuration for 7b and 13b. The yaml file contains configuration for the 7b variant, but you can just aswell use the same settings for 13b.

The 7b variant fits on any 24GB VRAM GPU and will take up about 17 GB of VRAM during training if using qlora and 20 GB if using lora. On a RTX 4090 it trains 3 epochs of the default dataset in about 15 minutes.

The 13b variant will fit if you change these settings to these values:
gradient_accumulation_steps: 2
micro_batch_size: 1

```shell
accelerate launch scripts/finetune.py examples/llama-2/qlora.yml

```
or

```shell
accelerate launch scripts/finetune.py examples/llama-2/lora.yml

```
