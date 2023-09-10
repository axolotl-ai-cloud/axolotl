# pip install auto-gptq --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/

# import debugpy
# debugpy.listen(('0.0.0.0', 5678))
# debugpy.wait_for_client()
# debugpy.breakpoint()

import json
import random
import time
from pathlib import Path
import logging

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, LlamaTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from axolotl.prompters import AlpacaPrompter
from axolotl.utils.models import load_model, load_tokenizer
from axolotl.common.cli import TrainerCliArgs
from axolotl.logging_config import configure_logging
from axolotl.utils.dict import DictDefault
# from scripts.finetune import load_cfg
from finetune import load_cfg, get_merged_out_dir, do_merge_lora_model_and_tokenizer, load_datasets

from axolotl.utils.quantize import load_merged_model, get_quantized_model, quantize_and_save, push_model, get_quantized_model_id, get_quantized_model_dir, get_examples_for_quantization

configure_logging()
LOG = logging.getLogger("axolotl")

# logging.basicConfig(
#     format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.DEBUG, datefmt="%Y-%m-%d %H:%M:%S"
# )

# LOG.setLevel(logging.DEBUG)
# handler = logging.StreamHandler()
# formatter = logging.Formatter('%(asctime)s %(levelname)s [%(name)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
# handler.setFormatter(formatter)
# LOG.addHandler(handler)

print("Done importing...")

## CHANGE BELOW ##
# config_path: Path = Path("./examples/llama-2/lora.yml")
config_path: Path = Path("./examples/llama-2/lora-short.yml")

# pretrained_model_dir = "facebook/opt-125m"
# quantized_model_dir = "opt-125m-4bit"
dataset_name = "teknium/GPT4-LLM-Cleaned"
# huggingface_username = "CHANGE_ME"
## CHANGE ABOVE

def main():
    print("Starting...")
    # return
    # prompt = "<|prompt|>How can entrepreneurs start building their own communities even before launching their product?</s><|answer|>"

    should_quantize = True
    # tokenizer = get_tokenizer()

    cfg = load_cfg(config_path)

    cfg['lora_model_dir'] = cfg['output_dir']

    LOG.info(f"loading tokenizer... {cfg.tokenizer_config or cfg.base_model_config}")
    tokenizer = load_tokenizer(cfg)

    if should_quantize:
        print("Quantizing...")

        print("Loading dataset...")
        datasets = load_datasets(cfg=cfg, cli_args=TrainerCliArgs())
        train_dataset = datasets.train_dataset
        n_samples = 128
        # # n_samples = 2
        # examples = train_dataset.shuffle(seed=42).select(
        #         [
        #             random.randrange(0, len(train_dataset) - 1)  # nosec
        #             for _ in range(n_samples)
        #         ]
        #     )

        LOG.info("loading model and (optionally) peft_config...")
        # model, peft_config = load_model(cfg, tokenizer, inference=True)
        model = load_merged_model(cfg)
        # model = get_model()

        # examples = load_data(dataset_name, tokenizer, n_samples)

        # print(examples)
        # examples_for_quant = [
        #     {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]}
        #     for example in examples
        # ]
        # print(examples_for_quant)
        examples_for_quant = get_examples_for_quantization(train_dataset, n_samples)

        modelq = quantize_and_save(cfg, model, tokenizer, examples_for_quant)
    else:
        print("Loading quantized model...")
        modelq = get_quantized_model(cfg)

    push_model(cfg, modelq, tokenizer)

if __name__ == "__main__":
    main()


# Load configure
# Load dataset
# Load tokenizer
# Prepare database
# Load previous model, final checkpoint


# --merge_lora --lora_model_dir="./completed-model" --load_in_8bit=False --load_in_4bit=False
# accelerate launch ./scripts/finetune.py ./examples/llama-2/lora.yml --merge_lora --lora_model_dir="./lora-out" --load_in_8bit=False --load_in_4bit=False
# CUDA_VISIBLE_DEVICES="1" accelerate launch ./scripts/finetune.py ./examples/llama-2/lora.yml --merge_lora --lora_model_dir="./lora-out" --load_in_8bit=False --load_in_4bit=False

# HUB_MODEL_ID="Glavin001/llama-2-7b-alpaca_2k_test" accelerate launch ./scripts/quantize.py

