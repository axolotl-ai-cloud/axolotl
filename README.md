# Axolotl

#### You know you're going to axolotl questions


### Converting JSON data files to JSONL

```shell
python3 ./scripts/alpaca_json_to_jsonl.py --input data/alpaca_data_gpt4.json > data/alpaca_data_gpt4.jsonl
python3 ./scripts/alpaca_json_to_jsonl.py --input data/raw/vicuna_cleaned.json > data/vicuna_cleaned.jsonl
python3 ./scripts/alpaca_json_to_jsonl.py --input data/raw/roleplay-similarity_0.6-instruct-dataset.json > data/roleplay-similarity_0.6-instruct-dataset.jsonl
python3 ./scripts/alpaca_json_to_jsonl.py --input data/raw/gpt4-instruct-similarity-0.6-dataset.json > data/gpt4-instruct-similarity-0.6-dataset.jsonl
```
