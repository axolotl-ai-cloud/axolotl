# OpenAI's GPT-OSS

GPT-OSS is a 20 billion parameter MoE model trained by OpenAI, released in August 2025.

- 20B Full Parameter SFT can be trained on 8x48GB GPUs (peak reserved memory @ ~36GiB/GPU) - [YAML](./gpt-oss-20b-fft-fsdp2.yaml)
- 20B LoRA SFT (all linear layers, and experts in last two layers) can be trained a single GPU (peak reserved memory @ ~44GiB) - [YAML](./gpt-oss-20b-sft-lora-singlegpu.yaml)
