# Axolotl

#### You know you're going to axolotl questions

## Getting Started

- Download some datasets.

```shell
curl https://raw.githubusercontent.com/tloen/alpaca-lora/main/alpaca_data_gpt4.json -o data/raw/alpaca_data_gpt4.json
curl https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json -L -o data/raw/vicuna_cleaned.json
curl https://github.com/teknium1/GPTeacher/blob/main/Instruct/gpt4-instruct-similarity-0.6-dataset.json?raw=true -L -o data/raw/gpt4-instruct-similarity-0.6-dataset.json
curl https://github.com/teknium1/GPTeacher/blob/main/Roleplay/roleplay-similarity_0.6-instruct-dataset.json?raw=true -L -o data/raw/roleplay-similarity_0.6-instruct-dataset.json
```

- Convert the JSON data files to JSONL.

```shell
python3 ./scripts/alpaca_json_to_jsonl.py --input data/alpaca_data_gpt4.json > data/alpaca_data_gpt4.jsonl
python3 ./scripts/alpaca_json_to_jsonl.py --input data/raw/vicuna_cleaned.json > data/vicuna_cleaned.jsonl
python3 ./scripts/alpaca_json_to_jsonl.py --input data/raw/roleplay-similarity_0.6-instruct-dataset.json > data/roleplay-similarity_0.6-instruct-dataset.jsonl
python3 ./scripts/alpaca_json_to_jsonl.py --input data/raw/gpt4-instruct-similarity-0.6-dataset.json > data/gpt4-instruct-similarity-0.6-dataset.jsonl
```

- Using JSONL makes it easier to subset the data if you want a smaller training set, i.e get 2000 random examples.

```shell
shuf -n2000 data/vicuna_cleaned.jsonl > data/vicuna_cleaned.subset0.jsonl
```

- Create a new or update the existing YAML config (config/pythia_1_2B_alpaca.yml)[config/pythia_1_2B_alpaca.yml]
- Install python dependencies `pip3 install -e .[int4_triton]` or `pip3 install -e .[int4]`
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
- Alternatively you can pass in the config file like: `accelerate launch scripts/finetune.py configs/llama_7B_alpaca.yml`
