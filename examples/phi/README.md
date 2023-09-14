# Phi

Due to some nuances with the phi code, please use deepspeed when training phi.

```shell
accelerate launch scripts/finetune.py examples/phi/phi-ft.yml --deepspeed deepspeed/zero1.json
```
