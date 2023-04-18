# Axolotl

#### Go ahead and axolotl questions

## Support Matrix

|          | fp16/fp32 | fp16/fp32 w/ lora | 4bit-quant | 4bit-quant w/flash attention | flash attention | xformers attention |
|----------|:----------|:------------------|------------|------------------------------|-----------------|--------------------|
| llama    | ✅         | ✅                 | ✅          | ✅                            | ✅               | ✅                  |
| Pythia   | ✅         | ✅                 | ❌          | ❌                            | ❌               | ❓                  |
| cerebras | ✅         | ✅                 | ❌          | ❌                            | ❌               | ❓                  |


## Getting Started

- Point the config you are using to a huggingface hub dataset (see [configs/llama_7B_4bit.yml](https://github.com/winglian/axolotl/blob/main/configs/llama_7B_4bit.yml#L6-L8))

```yaml
datasets:
  - path: vicgalle/alpaca-gpt4
    type: alpaca
```

- Optionally Download some datasets, see [data/README.md](data/README.md)


- Create a new or update the existing YAML config [config/pythia_1_2B_alpaca.yml](config/pythia_1_2B_alpaca.yml)
- Install python dependencies with ONE of the following:

    - `pip3 install -e .[int4]` (recommended)
    - `pip3 install -e .[int4_triton]`
    - `pip3 install -e .`
-
- If not using `int4` or `int4_triton`, run `pip install "peft @ git+https://github.com/huggingface/peft.git"`
- Configure accelerate `accelerate config` or update `~/.cache/huggingface/accelerate/default_config.yaml`

```yaml
compute_environment: LOCAL_MACHINE
distributed_type: MULTI_GPU
downcast_bf16: 'no'
gpu_ids: all
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 4
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

- Train! `accelerate launch scripts/finetune.py`, make sure to choose the correct YAML config file
- Alternatively you can pass in the config file like: `accelerate launch scripts/finetune.py configs/llama_7B_alpaca.yml`~~
