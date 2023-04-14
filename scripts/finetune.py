import os
import sys
from pathlib import Path

import fire
import torch
import transformers
import yaml
from attrdict import AttrDict
from datasets import load_dataset, IterableDataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# add src to the pythonpath so we don't need to pip install this
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from axolotl.datasets import TokenizedPromptDataset
from axolotl.prompt_tokenizers import AlpacaPromptTokenizingStrategy, ShareGPTPromptTokenizingStrategy, \
    LLAMA_DEFAULT_PAD_TOKEN, GPTeacherPromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter, GPTeacherPrompter, ShareGPTPrompter

def setup_wandb_env_vars(cfg):
    if len(cfg.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
        cfg.use_wandb = True
        if len(cfg.wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = cfg.wandb_watch
        if len(cfg.wandb_log_model) > 0:
            os.environ["WANDB_LOG_MODEL"] = cfg.wandb_log_model


def load_model(base_model, model_type, tokenizer_type, cfg, adapter="lora"):
    if adapter != "lora":
        raise NotImplementedError(f"{adapter} peft adapter not available")
    try:
        model = getattr(transformers, model_type).from_pretrained(
            base_model,
            load_in_8bit=cfg.load_in_8bit,
            torch_dtype=torch.float16 if cfg.load_in_8bit else torch.float32,
            device_map=cfg.device_map,
        )
    except:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=cfg.load_in_8bit,
            torch_dtype=torch.float16 if cfg.load_in_8bit else torch.float32,
            device_map=cfg.device_map,
        )

    try:
        tokenizer = getattr(transformers, tokenizer_type).from_pretrained(model)
    except:
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    if tokenizer.__class__.__name__ == "LlamaTokenizer":
        tokenizer.pad_token = LLAMA_DEFAULT_PAD_TOKEN

    if cfg.load_in_8bit:
        model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if cfg.ddp:
        model.to(f"cuda:{cfg.local_rank}")

    # TODO resume_from_checkpoint handling

    model.print_trainable_parameters()
    return model, tokenizer


def train(
    config: Path = Path('configs/pythia_1_2B_alpaca.yml'),
    **kwargs,
):
    # load the config from the yaml file
    with open(config, 'r') as f:
        cfg: AttrDict = AttrDict(yaml.load(f))
    # if there are any options passed in the cli, if it is something that seems valid from the yaml,
    # then overwrite the value
    for k, v in enumerate(kwargs):
        if k in cfg:
            cfg.k = v

    # setup some derived config / hyperparams
    cfg.gradient_accumulation_steps = cfg.batch_size // cfg.micro_batch_size
    cfg.device_map = "auto"
    cfg.world_size = int(os.environ.get("WORLD_SIZE", 1))
    cfg.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    cfg.ddp = cfg.world_size != 1
    if cfg.ddp:
        cfg.device_map = {"": int(os.environ.get("LOCAL_RANK", 0))}
        cfg.gradient_accumulation_steps = cfg.gradient_accumulation_steps // cfg.world_size
    setup_wandb_env_vars(cfg)

    # Load the model and tokenizer
    model, tokenizer = load_model(cfg.base_model, cfg.model_type, cfg.tokenizer_type, cfg, adapter=cfg.adapter)
    datasets = []
    for d in cfg.datasets:
        ds: IterableDataset = load_dataset("json", data_files=d.path, streaming=True, num_proc=4, split=None)
        if d.type == "alpaca":
            ds_strategy = AlpacaPromptTokenizingStrategy(AlpacaPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len)
            ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
            datasets.append(ds_wrapper)
        elif d.type == "gpteacher":
            ds_strategy = GPTeacherPromptTokenizingStrategy(GPTeacherPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len)
            ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
            datasets.append(ds_wrapper)
        elif d.type == "sharegpt":
            ds_strategy = ShareGPTPromptTokenizingStrategy(ShareGPTPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len)
            ds_wrapper = TokenizedPromptDataset(ds_strategy, ds)
            datasets.append(ds_wrapper)


if __name__ == "__main__":
    fire.Fire(train)
