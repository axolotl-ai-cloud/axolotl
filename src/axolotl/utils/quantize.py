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

# import torch
# from datasets import load_dataset, Dataset
# from transformers import AutoTokenizer, LlamaTokenizer, TextGenerationPipeline
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from axolotl.prompters import AlpacaPrompter
from axolotl.utils.models import load_model, load_tokenizer
from axolotl.common.cli import TrainerCliArgs
from axolotl.logging_config import configure_logging
from axolotl.utils.dict import DictDefault
# from finetune import load_cfg, get_merged_out_dir, do_merge_lora_model_and_tokenizer

# configure_logging()
# LOG = logging.getLogger("axolotl")

quantize_config = BaseQuantizeConfig(
    bits=4,  # quantize model to 4-bit
    group_size=128,  # it is recommended to set the value to 128
    desc_act=False,  # set to False can significantly speed up inference but the perplexity may slightly bad 
)

def get_merged_out_dir(cfg: DictDefault):
    return Path(cfg.output_dir) / "merged"

def load_merged_model(cfg: DictDefault):
    print("Loading merged model...")

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

def get_examples_for_quantization(dataset, n_samples):
    print("Loading dataset...")
    examples = dataset.shuffle(seed=42).select(
            [
                random.randrange(0, len(dataset) - 1)  # nosec
                for _ in range(n_samples)
            ]
        )

    examples_for_quant = [
        {"input_ids": example["input_ids"], "attention_mask": example["attention_mask"]}
        for example in examples
    ]
    return examples_for_quant
