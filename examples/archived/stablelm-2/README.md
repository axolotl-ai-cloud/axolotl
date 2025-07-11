# StableLM 2

This repository contains examples for training and processing using StableLM-2. It also includes a section to help you estimate the GPU requirements for your specific use case.

## Estimating GPU Requirements

| type          | deepspeed | batch size | context length | vRAM GPU (GBs) |
|---------------|-----------|------------|----------------|----------------|
| full finetune | N/A       | 1          | 4096           | ~21.5GBs       |
| full finetune | zero2     | 1          | 4096           | ~20GBs         |
| lora          | N/A       | 1          | 4096           | ~16.6GBs       |

The above are estimates and might differ slight depending on the setup for example whether you pack your sequence lengths or not (the above assumes you do to length 4096).

This blog post from Hamel Husain was a great resource for estimating these numbers: https://hamel.dev/notes/llm/03_estimating_vram.html

## Training
We have example scripts here for both full finetuning and lora using the popular alpaca dataset:

```shell
# preprocess the dataset
CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess examples/stablelm-2/1.6b/lora.yml
```

Single GPU Training:
```shell
python -m axolotl.cli.train examples/stablelm-2/fft.yml --deepspeed deepspeed_configs/zero2.json
# OR
python -m axolotl.cli.train examples/stablelm-2/1.6b/lora.yml
```

Multinode GPU Training with `accelerate`:
```shell
# make sure you've configured accelerate properly
accelerate launch -m axolotl.cli.train examples/stablelm-2/1.6b/fft.yml --deepspeed deepspeed_configs/zero2.json
```
