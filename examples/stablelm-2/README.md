# StableLM

Determine GPU size requirements: https://hamel.dev/notes/llm/03_estimating_vram.html

Due to some nuances with the phi code, please use deepspeed when training phi for full finetune.

```shell
CUDA_VISIBLE_DEVICES="" python -m axolotl.cli.preprocess examples/stablelm-2/lora.yml
accelerate launch -m axolotl.cli.train examples/phi/phi-ft.yml --deepspeed deepspeed_configs/zero2.json

# OR
export CUDA_VISIBLE_DEVICES=6
CUDA_VISIBLE_DEVICES=0 python -m axolotl.cli.train examples/stablelm-2/lora.yml
```
