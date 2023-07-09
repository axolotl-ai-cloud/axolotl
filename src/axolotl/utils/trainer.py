"""Module containing the Trainer class and related functions"""

import importlib
import logging
import math
import os
import sys
from pathlib import Path
from typing import Optional

import bitsandbytes as bnb
import torch.cuda
import transformers
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from transformers import EarlyStoppingCallback, Trainer
from transformers.trainer_pt_utils import get_parameter_names

from axolotl.utils.callbacks import (
    SaveBetterTransformerModelCallback,
    SavePeftModelCallback,
)
from axolotl.utils.schedulers import InterpolatingLogScheduler


class OneCycleLRSchedulerTrainer(Trainer):
    """
    Trainer subclass that uses the OneCycleLR scheduler
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lr_scheduler = None

    def create_scheduler(
        self,
        num_training_steps: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        optimizer = self.optimizer if optimizer is None else optimizer
        num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
        pct_start = num_warmup_steps / num_training_steps

        self.lr_scheduler = OneCycleLR(
            optimizer,
            max_lr=self.args.learning_rate,
            total_steps=num_training_steps,
            pct_start=pct_start,
            div_factor=6,
        )

        return self.lr_scheduler


def setup_trainer(cfg, train_dataset, eval_dataset, model, tokenizer):
    total_num_steps = int(
        math.ceil(len(train_dataset) * cfg.num_epochs / cfg.batch_size)
    )
    warmup_steps = (
        cfg.warmup_steps
        if cfg.warmup_steps is not None
        else min(int(0.03 * total_num_steps), 100)
    )
    logging_steps = (
        cfg.logging_steps
        if cfg.logging_steps is not None
        else max(min(int(0.005 * total_num_steps), 10), 1)
    )

    training_arguments_kwargs = {}
    if cfg.bf16 == "full":
        training_arguments_kwargs["bf16_full_eval"] = True
    else:
        training_arguments_kwargs["bf16"] = cfg.bf16
    training_arguments_kwargs["fp16"] = (cfg.fp16 and not cfg.bf16) or False
    training_arguments_kwargs["tf32"] = cfg.tf32
    training_arguments_kwargs["warmup_steps"] = warmup_steps
    training_arguments_kwargs["logging_steps"] = logging_steps

    if cfg.seed:
        training_arguments_kwargs["seed"] = cfg.seed

    if cfg.gradient_checkpointing:
        if cfg.gptq:
            from alpaca_lora_4bit.gradient_checkpointing import (
                apply_gradient_checkpointing,
            )

            gradient_checkpointing_ratio = (
                cfg.gradient_checkpointing_ratio
                if cfg.gradient_checkpointing_ratio
                else 1.0
            )
            apply_gradient_checkpointing(
                model, checkpoint_ratio=gradient_checkpointing_ratio
            )
        else:
            training_arguments_kwargs[
                "gradient_checkpointing"
            ] = cfg.gradient_checkpointing
    if cfg.fsdp:
        training_arguments_kwargs["fsdp"] = cfg.fsdp
        if cfg.fsdp_config:
            training_arguments_kwargs["fsdp_config"] = dict(cfg.fsdp_config)

    # deepspeed
    if (
        os.environ.get("ACCELERATE_USE_DEEPSPEED") == "true"
        and torch.cuda.device_count() > 1
    ):
        if cfg.deepspeed:
            training_arguments_kwargs["deepspeed"] = cfg.deepspeed
        else:
            # make a guess here
            # TODO search Path("./") for one
            training_arguments_kwargs["deepspeed"] = "./ds_config.json"

    if cfg.adam_beta1:
        training_arguments_kwargs["adam_beta1"] = cfg.adam_beta1
    if cfg.adam_beta2:
        training_arguments_kwargs["adam_beta2"] = cfg.adam_beta2
    if cfg.adam_epsilon:
        training_arguments_kwargs["adam_epsilon"] = cfg.adam_epsilon
    if cfg.max_grad_norm:
        training_arguments_kwargs["max_grad_norm"] = cfg.max_grad_norm

    if cfg.hub_model_id:
        training_arguments_kwargs["hub_model_id"] = cfg.hub_model_id
        training_arguments_kwargs["push_to_hub"] = True

    training_args = transformers.TrainingArguments(
        per_device_train_batch_size=cfg.micro_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size
        if cfg.eval_batch_size is not None
        else cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        evaluation_strategy="steps" if cfg.val_set_size > 0 else "no",
        save_strategy="steps" if cfg.save_steps else "epoch",
        eval_steps=cfg.eval_steps if cfg.val_set_size > 0 else None,
        save_steps=cfg.save_steps,
        output_dir=cfg.output_dir,
        save_total_limit=3,
        load_best_model_at_end=(
            cfg.load_best_model_at_end is not False
            and cfg.val_set_size > 0
            and cfg.save_steps
            and cfg.save_steps % cfg.eval_steps == 0
            and cfg.load_in_8bit is not True
        )
        or False,
        ddp_find_unused_parameters=False if cfg.ddp else None,
        group_by_length=cfg.group_by_length,
        report_to="wandb" if cfg.use_wandb else None,
        run_name=cfg.wandb_run_id if cfg.use_wandb else None,
        optim=cfg.optimizer if cfg.optimizer else "adamw_hf",
        lr_scheduler_type=cfg.lr_scheduler
        if cfg.lr_scheduler and cfg.lr_scheduler not in ("one_cycle", "log_sweep")
        else "cosine",
        weight_decay=cfg.weight_decay if cfg.weight_decay is not None else 0.0,
        **training_arguments_kwargs,
    )

    trainer_kwargs = {}

    if cfg.optimizer == "adamw_anyprecision":
        if Path(cfg.torchdistx_path).exists():
            sys.path.append(cfg.torchdistx_path)
            importlib.import_module("torchdistx")
    if (
        cfg.optimizer == "adamw_bnb_8bit"
        and not cfg.gptq
        and "deepspeed" not in training_arguments_kwargs
        and not cfg.fsdp
    ):
        decay_parameters = get_parameter_names(model, [nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (n in decay_parameters and p.requires_grad)
                ],
                "weight_decay": training_args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in model.named_parameters()
                    if (n not in decay_parameters and p.requires_grad)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = bnb.optim.Adam8bit(
            optimizer_grouped_parameters,
            betas=(training_args.adam_beta1, training_args.adam_beta2),
            eps=training_args.adam_epsilon,
            lr=training_args.learning_rate,
        )

        if cfg.lr_scheduler == "one_cycle":
            lr_scheduler_kwargs = (
                cfg.lr_scheduler_kwargs if cfg.lr_scheduler_kwargs else {}
            )
            lr_scheduler = OneCycleLR(
                optimizer,
                cfg.learning_rate,
                total_steps=total_num_steps,
                epochs=cfg.num_epochs,
                div_factor=cfg.lr_div_factor if cfg.lr_div_factor else 6,
                **lr_scheduler_kwargs,
            )
        elif cfg.lr_scheduler == "log_sweep":
            lr_scheduler = InterpolatingLogScheduler(
                optimizer,
                cfg.warmup_steps,
                cfg.log_sweep_min_lr if cfg.log_sweep_min_lr else 1e-10,
                cfg.log_sweep_max_lr if cfg.log_sweep_max_lr else 10,
            )
        else:
            lr_scheduler = transformers.get_cosine_schedule_with_warmup(
                optimizer,
                training_args.warmup_steps,
                total_num_steps,
            )
        trainer_kwargs["optimizers"] = (optimizer, lr_scheduler)

    callbacks = []
    # TODO on_save callback to sync checkpoints to GCP/AWS in background
    if cfg.early_stopping_patience:
        early_stop_cb = EarlyStoppingCallback(
            cfg.early_stopping_patience,
        )
        callbacks.append(early_stop_cb)

    if cfg.local_rank == 0 and cfg.adapter in [
        "lora",
        "qlora",
    ]:  # only save in rank 0
        callbacks.append(SavePeftModelCallback)

    if hasattr(model, "use_bettertransformer") and model.use_bettertransformer is True:
        callbacks.append(SaveBetterTransformerModelCallback)

    data_collator_kwargs = {
        "padding": True,
    }
    if cfg.collator_pad_to_longest:
        data_collator_kwargs["padding"] = "longest"
    else:
        data_collator_kwargs["pad_to_multiple_of"] = 8

    if cfg.is_llama_derived_model and cfg.landmark_attention:
        from functools import partial

        from axolotl.monkeypatch.llama_landmark_attn import (
            add_mem_tokens,
            get_mem_id,
            set_model_mem_id,
        )

        set_model_mem_id(model, tokenizer)

        logging.info("Adding landmark attention tokens to dataset")

        for dataset in [train_dataset, eval_dataset]:
            dataset = dataset.map(
                partial(add_mem_tokens, mem_freq=50, mem_id=get_mem_id(tokenizer)),
                batched=False,
                num_proc=32,
            )

    trainer_cls = (
        OneCycleLRSchedulerTrainer
        if cfg.lr_scheduler == "one_cycle" and (cfg.fsdp or cfg.adapter == "qlora")
        else transformers.Trainer
    )
    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            **data_collator_kwargs,
        ),
        callbacks=callbacks,
        **trainer_kwargs,
    )

    return trainer
