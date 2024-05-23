# Phi

Due to some nuances with the phi code, please use deepspeed when training phi for full finetune.

```shell
accelerate launch -m axolotl.cli.train examples/phi/phi-ft.yml --deepspeed deepspeed_configs/zero1.json

# OR

python -m axolotl.cli.train examples/phi/phi-qlora.yml
```
