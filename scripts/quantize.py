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
config_path: Path = Path("./examples/llama-2/lora.yml")

# pretrained_model_dir = "facebook/opt-125m"
# quantized_model_dir = "opt-125m-4bit"
dataset_name = "teknium/GPT4-LLM-Cleaned"
# huggingface_username = "CHANGE_ME"
## CHANGE ABOVE

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad 
)

# TEMPLATE = "<|prompt|>{instruction}</s><|answer|>"
prompter = AlpacaPrompter()

# def load_data(data_path, tokenizer, n_samples, template=TEMPLATE):
def load_data(data_path, tokenizer, n_samples):
    # Load dataset
    dataset = load_dataset(data_path)
    
    if "train" in dataset:
        raw_data = dataset["train"]
    else:
        raw_data = dataset

    # Sample from the dataset if n_samples is provided and less than the dataset size
    if n_samples is not None and n_samples < len(raw_data):
        raw_data = raw_data.shuffle(seed=42).select(range(n_samples))

    def tokenize(examples):
        instructions = examples["instruction"]
        outputs = examples["output"]

        prompts = []
        texts = []
        input_ids = []
        attention_mask = []
        for input_text, output_text in zip(instructions, outputs):
            # prompt = template.format(instruction=input_text)
            # prompt = next(prompter.build_prompt(instruction=input_text, output=output_text))
            prompt = next(prompter.build_prompt(instruction=input_text))
            text = prompt + output_text

            if len(tokenizer(prompt)["input_ids"]) >= tokenizer.model_max_length:
                continue

            tokenized_data = tokenizer(text)

            input_ids.append(tokenized_data["input_ids"][: tokenizer.model_max_length])
            attention_mask.append(tokenized_data["attention_mask"][: tokenizer.model_max_length])
            prompts.append(prompt)
            texts.append(text)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt": prompts,
            "text": texts,
        }

    raw_data = raw_data.map(
        tokenize,
        batched=True,
        batch_size=len(raw_data),
        num_proc=1,
        keep_in_memory=True,
        load_from_cache_file=False,
        # remove_columns=["instruction", "input"]
    )

    # Convert to PyTorch tensors
    raw_data.set_format(type='torch', columns=['input_ids', 'attention_mask'])

    # for sample in dataset:
    #     sample["input_ids"] = torch.LongTensor(sample["input_ids"])
    #     sample["attention_mask"] = torch.LongTensor(sample["attention_mask"])

    return raw_data


# def get_tokenizer():
#     print("Loading tokenizer...")
#     # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
#     tokenizer = LlamaTokenizer.from_pretrained(pretrained_model_dir, use_fast=True)
#     return tokenizer

# def get_model():
def load_merged_model(cfg: DictDefault):
    print("Loading model...")

    merged_out_dir = get_merged_out_dir(cfg)
    
    # Check if the merged model exists
    if not merged_out_dir.exists():
        # If not, merge the model
        print("Merged model not found. Merging...")
        # model, tokenizer = load_model(cfg, inference=True)
        # do_merge_lora_model_and_tokenizer(cfg=cfg, model=model, tokenizer=tokenizer)
        raise NotImplementedError("Merging model is not implemented yet.")

    # load un-quantized model, by default, the model will always be loaded into CPU memory
    model = AutoGPTQForCausalLM.from_pretrained(merged_out_dir, quantize_config)
    # model = AutoGPTQForCausalLM.from_pretrained(pretrained_model_dir, quantize_config)
    print("Model loaded.")
    return model

def get_quantized_model(cfg: DictDefault):
    print("Loading quantized model...")
    quantized_model_dir = get_quantized_model_dir(cfg)
    model = AutoGPTQForCausalLM.from_quantized(quantized_model_dir, device="cuda:0", use_safetensors=True)
    print("Model loaded.")
    return model

def quantize_and_save(cfg: DictDefault, model, tokenizer, examples_for_quant):
    print("Quantize...")
    start = time.time()
    # quantize model, the examples should be list of dict whose keys can only be "input_ids" and "attention_mask"
    model.quantize(
        examples_for_quant,
        batch_size=1,
        # batch_size=args.quant_batch_size,
        # use_triton=args.use_triton,
        # autotune_warmup_after_quantized=args.use_triton
    )
    end = time.time()
    print(f"quantization took: {end - start: .4f}s")

    # save quantized model
    print("Saving quantized model...")
    # model.save_quantized(quantized_model_dir)
    quantized_model_dir = get_quantized_model_dir(cfg)
    model.save_quantized(quantized_model_dir, use_safetensors=True)
    print("Saving tokenizer...")
    tokenizer.save_pretrained(quantized_model_dir)
    print("Saved.")

    return model

def push_model(cfg: DictDefault, model, tokenizer):
# def push_model(model):
    # push quantized model to Hugging Face Hub. 
    # to use use_auth_token=True, Login first via huggingface-cli login.
    # or pass explcit token with: use_auth_token="hf_xxxxxxx"
    # (uncomment the following three lines to enable this feature)
    # repo_id = f"YourUserName/{quantized_model_dir}"
    print("Pushing to Huggingface hub...")
    # repo_id = f"{huggingface_username}/{quantized_model_dir}"
    repo_id = get_quantized_model_id(cfg)
    pretrained_model_dir = cfg['base_model']
    commit_message = f"AutoGPTQ model for {pretrained_model_dir}: {quantize_config.bits}bits, gr{quantize_config.group_size}, desc_act={quantize_config.desc_act}"
    # model.push_to_hub(repo_id, commit_message=commit_message, use_auth_token=True, use_safetensors=True, safe_serialization=True)
    # model.push_to_hub(repo_id, commit_message=commit_message, use_auth_token=True, safe_serialization=True)
    model.push_to_hub(repo_id, commit_message=commit_message, use_auth_token=True, use_safetensors=True)
    tokenizer.push_to_hub(repo_id, commit_message=commit_message, use_auth_token=True)
    print("Pushed.")

# def push_tokenizer(tokenizer):

def get_quantized_model_id(cfg: DictDefault):
# def get_quantized_model_id(cfg: DictDefault, quantize_config):
    # return f"{cfg.hub_model_id}-{quantize_config.bits}bits-gr{quantize_config.group_size}-desc_act{quantize_config.desc_act}"
    if not cfg.hub_model_id:
        raise ValueError("Missing hub_model_id in the configuration.")
    return f"{cfg.hub_model_id}-GPTQ"

def get_quantized_model_dir(cfg: DictDefault):
# def get_quantized_model_dir(cfg: DictDefault, quantize_config):
    if not cfg.output_dir:
        raise ValueError("Missing output_dir in the configuration.")
    return f"{cfg.output_dir.lstrip('./')}-GPTQ"

def main():
    print("Starting...")
    # return
    # prompt = "<|prompt|>How can entrepreneurs start building their own communities even before launching their product?</s><|answer|>"

    should_quantize = False
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
        # n_samples = 2
        examples = train_dataset.shuffle(seed=42).select(
                [
                    random.randrange(0, len(train_dataset) - 1)  # nosec
                    for _ in range(n_samples)
                ]
            )

        LOG.info("loading model and (optionally) peft_config...")
        # model, peft_config = load_model(cfg, tokenizer, inference=True)
        model = load_merged_model(cfg)
        # model = get_model()

        # examples = load_data(dataset_name, tokenizer, n_samples)

        # print(examples)
        examples_for_quant = [
            {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]}
            for example in examples
        ]
        # print(examples_for_quant)

        modelq = quantize_and_save(cfg, model, tokenizer, examples_for_quant)
    else:
        print("Loading quantized model...")
        modelq = get_quantized_model(cfg)

    push_model(cfg, modelq, tokenizer)

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

