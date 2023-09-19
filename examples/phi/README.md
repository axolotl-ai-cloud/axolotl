# Phi

Due to some nuances with the phi code, please use deepspeed when training phi for full finetune.

```shell
# You may need to install deepspeed with `pip3 install deepspeed`
accelerate launch -m axolotl.cli.train examples/phi/phi-ft.yml --deepspeed deepspeed/zero1.json

# OR

python -m axolotl.cli.train examples/phi/phi-qlora.yml
```
