# openllama-3b

Basic full tune
```shell
accelerate launch scripts/finetune.py examples/openllama-3b/config.yml
```

LoRA
```shell
accelerate launch scripts/finetune.py examples/openllama-3b/lora.yml
```

QLoRA
```shell
accelerate launch scripts/finetune.py examples/openllama-3b/qlora.yml
```
