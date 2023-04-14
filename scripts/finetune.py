import math
import os
import signal
import sys
from pathlib import Path

import bitsandbytes as bnb
import fire
import torch
import transformers
import yaml
from attrdict import AttrDict
from datasets import load_dataset, IterableDataset, Dataset
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_int8_training, get_peft_model_state_dict,
)
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# add src to the pythonpath so we don't need to pip install this
from transformers.trainer_pt_utils import get_parameter_names

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(project_root, 'src')
sys.path.insert(0, src_dir)

from axolotl.datasets import TokenizedPromptDataset, ConstantLengthDataset
from axolotl.prompt_tokenizers import AlpacaPromptTokenizingStrategy, ShareGPTPromptTokenizingStrategy, \
    LLAMA_DEFAULT_PAD_TOKEN, GPTeacherPromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter, GPTeacherPrompter, ShareGPTPrompter

def setup_wandb_env_vars(cfg):
    if len(cfg.wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = cfg.wandb_project
        cfg.use_wandb = True
        if cfg.wandb_watch and len(cfg.wandb_watch) > 0:
            os.environ["WANDB_WATCH"] = cfg.wandb_watch
        if cfg.wandb_log_model and len(cfg.wandb_log_model) > 0:
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

    if tokenizer.__class__.__name__ == "GPTNeoXTokenizerFast":
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if cfg.load_in_8bit:
        model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        target_modules=cfg.lora_target_modules,
        lora_dropout=cfg.lora_dropout,
        fan_in_fan_out=cfg.lora_fan_in_fan_out,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if cfg.ddp:
        model.to(f"cuda:{cfg.local_rank}")

    # TODO resume_from_checkpoint handling

    model.print_trainable_parameters()
    return model, tokenizer, lora_config


def train(
    config: Path = Path('configs/pythia_1_2B_alpaca.yml'),
    **kwargs,
):
    # load the config from the yaml file
    with open(config, 'r') as f:
        cfg: AttrDict = AttrDict(yaml.load(f, Loader=yaml.Loader))
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
    model, tokenizer, lora_config = load_model(cfg.base_model, cfg.model_type, cfg.tokenizer_type, cfg, adapter=cfg.adapter)
    datasets = []
    for d in cfg.datasets:
        ds: IterableDataset = load_dataset("json", data_files=d.path, streaming=True, split=None)
        if d.type == "alpaca":
            ds_strategy = AlpacaPromptTokenizingStrategy(AlpacaPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len)
            ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
            datasets.append(ds_wrapper)
        elif d.type == "gpteacher":
            ds_strategy = GPTeacherPromptTokenizingStrategy(GPTeacherPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len)
            ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
            datasets.append(ds_wrapper)
        elif d.type == "sharegpt":
            ds_strategy = ShareGPTPromptTokenizingStrategy(ShareGPTPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len)
            ds_wrapper = TokenizedPromptDataset(ds_strategy, ds["train"])
            datasets.append(ds_wrapper)
    constant_len_dataset = ConstantLengthDataset(tokenizer, datasets, seq_length=cfg.sequence_len)
    constant_len_dataset = Dataset.from_list([_ for _ in constant_len_dataset]).train_test_split(
        test_size=cfg.val_set_size, shuffle=True, seed=42
    )

    print(constant_len_dataset)
    train_dataset = constant_len_dataset["train"]
    eval_dataset = constant_len_dataset["test"]

    total_num_steps = int(math.ceil(len(train_dataset) * cfg.num_epochs / cfg.batch_size))
    warmup_steps = min(int(0.03 * total_num_steps), 100)
    logging_steps = min(int(0.005 * total_num_steps), 10)
    save_steps = eval_steps = min(int(0.05 * total_num_steps), 200)

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        warmup_steps=warmup_steps,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        logging_steps=logging_steps,
        evaluation_strategy="steps" if cfg.val_set_size > 0 else "no",
        save_strategy="steps",
        eval_steps=eval_steps if cfg.val_set_size > 0 else None,
        save_steps=save_steps,
        output_dir=cfg.output_dir,
        save_total_limit=3,
        load_best_model_at_end=True if cfg.val_set_size > 0 else False,
        ddp_find_unused_parameters=False if cfg.ddp else None,
        group_by_length=cfg.group_by_length,
        report_to="wandb" if cfg.use_wandb else None,
        run_name=cfg.wandb_run_name if cfg.use_wandb else None,
    )

    decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if n in decay_parameters],
            "weight_decay": training_args.weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    adam_bnb_optim = bnb.optim.Adam8bit(
        optimizer_grouped_parameters,
        betas=(training_args.adam_beta1, training_args.adam_beta2),
        eps=training_args.adam_epsilon,
        lr=training_args.learning_rate,
    )

    lr_scheduler = transformers.get_cosine_schedule_with_warmup(
        adam_bnb_optim,
        training_args.warmup_steps,
        total_num_steps,
    )

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        optimizers=(adam_bnb_optim, lr_scheduler),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    old_state_dict = model.state_dict
    model.state_dict = (
        lambda self, *_, **__: get_peft_model_state_dict(
            self, old_state_dict()
        )
    ).__get__(model, type(model))

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    signal.signal(signal.SIGINT, lambda signal, frame: (
        model.save_pretrained(cfg.output_dir),
        exit(0)
    ))

    # go ahead and presave the adapter config
    lora_config.save_pretrained(cfg.output_dir)
    trainer.train(resume_from_checkpoint=cfg.resume_from_checkpoint)

    model.save_pretrained(cfg.output_dir)

if __name__ == "__main__":
    fire.Fire(train)
