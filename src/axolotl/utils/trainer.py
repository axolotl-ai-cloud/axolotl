"""Module containing the Trainer class and related functions"""
import importlib
import logging
import math
import os
import sys
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional, Union

import bitsandbytes as bnb
import numpy as np
import torch.cuda
import transformers
from datasets import Dataset, set_caching_enabled
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_pt_utils import get_parameter_names

from axolotl.utils.callbacks import (
    GPUStatsCallback,
    SaveBetterTransformerModelCallback,
    SavePeftModelCallback,
)
from axolotl.utils.collators import DataCollatorForSeq2Seq
from axolotl.utils.dataloader import MultipackDistributedDataloader
from axolotl.utils.schedulers import (
    InterpolatingLogScheduler,
    get_cosine_schedule_with_quadratic_warmup,
)

LOG = logging.getLogger("axolotl")


@torch.jit.script
def weighted_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, weights: torch.Tensor
):
    # Flatten the logits, labels, and weights tensors
    logits = logits.view(
        -1, logits.size(-1)
    )  # logits becomes of shape [batch_size*sequence_length, vocab_size]
    labels = labels.view(-1)  # labels becomes of shape [batch_size*sequence_length]
    weights = weights.view(-1)  # weights becomes of shape [batch_size*sequence_length]

    # Compute the unweighted cross entropy loss
    losses = torch.nn.functional.cross_entropy(logits, labels, reduction="none")

    # Apply the weights to the losses and compute their sum
    return (weights * losses).sum()


@torch.jit.script
def create_weighted_mask(labels: torch.Tensor):
    # Check if the tensor is 2D. If not, unsqueeze it to make it 2D
    if len(labels.shape) == 1:
        labels = labels.unsqueeze(0)

    weights = torch.zeros_like(labels).float()
    for i in range(labels.shape[0]):
        mask = labels[i] != -100

        # Create a tensor to track group ids
        group_ids = torch.zeros_like(labels[i]).int()
        curr_group_id = 0

        for j in range(1, len(labels[i])):
            if mask[j] and not mask[j - 1]:  # switch from masked to unmasked label
                curr_group_id += 1  # start new group
            group_ids[j] = (
                curr_group_id if mask[j] else 0
            )  # assign group id if unmasked label

        # Count only unmasked labels in each group
        group_counts = torch.bincount(group_ids[mask])

        mask_weights = torch.zeros_like(labels[i]).float()
        mask_weights[mask] = 1.0 / group_counts[group_ids[mask]]

        weights[i] = mask_weights

    return weights.squeeze()  # squeeze the output to match the input dimension


def trainer_weighted_loss(model_output, labels, shift_labels=True):
    logits = (
        model_output["logits"] if isinstance(model_output, dict) else model_output[0]
    )
    if shift_labels:
        logits = logits[..., :-1, :].contiguous()
        labels = labels[..., 1:].contiguous()

    weights = create_weighted_mask(labels)
    return weighted_cross_entropy(logits, labels, weights)


@dataclass
class AxolotlTrainingArguments(TrainingArguments):
    """
    Extend the base TrainingArguments for axolotl helpers
    """

    lr_quadratic_warmup: bool = field(
        default=False,
        metadata={"help": "Use quadratic warmup for cosine scheduling."},
    )
    sample_packing: bool = field(
        default=False,
        metadata={"help": "Use sample packing for efficient training."},
    )
    sample_packing_efficiency: float = field(
        default=1.0,
        metadata={"help": "Sample packing efficiency for calculating batch length."},
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum sequence length the model can handle"},
    )
    sample_packing_seq_len_multiplier: int = field(
        default=1,
        metadata={"help": "the multiplier for the max len for packed sequences"},
    )


class AxolotlTrainer(Trainer):
    """
    Extend the base Trainer for axolotl helpers
    """

    args = None  # type: AxolotlTrainingArguments

    def create_scheduler(
        self, num_training_steps: int, optimizer: torch.optim.Optimizer = None
    ):
        """
        Setup the scheduler. The optimizer of the trainer must have been set up either before this method is called or
        passed as an argument.

        Args:
            num_training_steps (int): The number of training steps to do.
            optimizer (torch.optim.Optimizer): The training optimizer
        """

        # fmt: off
        if self.lr_scheduler is None:  # type: ignore  # pylint: disable=access-member-before-definition
            # fmt: on
            if (
                self.args.lr_scheduler_type == "cosine"
                and self.args.lr_quadratic_warmup is True
            ):
                self.lr_scheduler = get_cosine_schedule_with_quadratic_warmup(  # pylint: disable=attribute-defined-outside-init
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            else:
                return super().create_scheduler(num_training_steps, optimizer)
        return self.lr_scheduler

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.args.world_size > 1 and self.args.sample_packing:
            return DistributedSampler(
                self.train_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                seed=self.args.seed,
            )
        return super()._get_train_sampler()

    def get_train_dataloader(self) -> Union[DataLoader, MultipackDistributedDataloader]:
        if self.args.sample_packing:
            train_sampler = self._get_train_sampler()
            return self.accelerator.prepare(
                MultipackDistributedDataloader(
                    self.train_dataset,
                    batch_size=self._train_batch_size,
                    seq_max_length=self.args.max_seq_length,
                    collate_fn=self.data_collator,
                    sampler=train_sampler,
                    packing_efficiency_estimate=self.args.sample_packing_efficiency,
                    sample_packing_seq_len_multiplier=self.args.sample_packing_seq_len_multiplier,
                    device_count=int(os.environ.get("WORLD_SIZE", 1)),
                )
            )
        return super().get_train_dataloader()

    def get_eval_dataloader(
        self, eval_dataset: Optional[Dataset] = None
    ) -> Union[DataLoader, MultipackDistributedDataloader]:
        if self.args.sample_packing:
            eval_dataset = (
                eval_dataset if eval_dataset is not None else self.eval_dataset
            )
            eval_sampler = self._get_eval_sampler(eval_dataset)
            return self.accelerator.prepare(
                MultipackDistributedDataloader(
                    eval_dataset,
                    batch_size=self.args.eval_batch_size,
                    seq_max_length=self.args.max_seq_length,
                    collate_fn=self.data_collator,
                    sampler=eval_sampler,
                    packing_efficiency_estimate=self.args.sample_packing_efficiency,
                    sample_packing_seq_len_multiplier=self.args.eval_batch_size,
                    device_count=int(os.environ.get("WORLD_SIZE", 1)),
                )
            )
        return super().get_eval_dataloader(eval_dataset)

    def compute_loss(self, model, inputs, return_outputs=False):
        # use one's weighted cross entropy loss calc
        # if self.args.sample_packing:
        #     labels = inputs.pop("labels")
        #     outputs = model(**inputs)
        #     loss = trainer_weighted_loss(outputs, labels, shift_labels=True)
        #     return (loss, outputs) if return_outputs else loss
        return super().compute_loss(model, inputs, return_outputs=return_outputs)


class OneCycleLRSchedulerTrainer(AxolotlTrainer):
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


def add_position_ids(sample):
    sample["position_ids"] = torch.arange(len(sample["input_ids"]))
    return sample


def drop_long_seq(sample, sequence_len=2048):
    return len(sample["input_ids"]) <= sequence_len


@contextmanager
def disable_datasets_caching():
    try:
        set_caching_enabled(False)
        yield
    finally:
        set_caching_enabled(True)


def process_datasets_for_packing(cfg, train_dataset, eval_dataset):
    if cfg.sample_packing:
        drop_long = partial(drop_long_seq, sequence_len=cfg.sequence_len)
        train_dataset = train_dataset.filter(drop_long, num_proc=os.cpu_count()).map(
            add_position_ids, num_proc=os.cpu_count()
        )
        if eval_dataset:
            eval_dataset = eval_dataset.filter(drop_long, num_proc=os.cpu_count()).map(
                add_position_ids, num_proc=os.cpu_count()
            )
    return train_dataset, eval_dataset


def calculate_total_num_steps(cfg, train_dataset, tokenizer):
    if cfg.sample_packing:
        # we have to drop anything longer then sequence len otherwise
        # flash attention with position ids fails
        if not cfg.total_num_tokens:
            LOG.info("calculating total_num_tokens")
            total_num_tokens = np.sum(
                train_dataset.data.column("input_ids")
                .to_pandas()
                .apply(lambda x: len(x))  # pylint: disable=unnecessary-lambda
                .values
            )
            LOG.info(f"📝 UPDATE CONFIG WITH: `total_num_tokens: {total_num_tokens}`")
            cfg.total_num_tokens = total_num_tokens

        if cfg.sample_packing_eff_est:
            total_num_steps = (
                # match count to len est in dataloader
                (
                    math.floor(
                        0.99
                        * cfg.total_num_tokens
                        / cfg.sample_packing_eff_est
                        / cfg.sequence_len
                        // cfg.batch_size
                        // int(os.environ.get("WORLD_SIZE", 1))
                    )
                    - 1
                )
                * cfg.num_epochs
            )
            LOG.info(
                f"total_num_tokens: {cfg.total_num_tokens}, total_num_steps: {total_num_steps}"
            )
        else:
            sampler = RandomSampler(train_dataset)
            data_loader = MultipackDistributedDataloader(
                train_dataset,
                batch_size=cfg.micro_batch_size,
                seq_max_length=cfg.max_packed_sequence_len or cfg.sequence_len,
                collate_fn=DataCollatorForSeq2Seq(
                    tokenizer,
                    return_tensors="pt",
                    padding="longest",
                ),
                sampler=sampler,
                packing_efficiency_estimate=cfg.sample_packing_eff_est,
                sample_packing_seq_len_multiplier=cfg.micro_batch_size,
                device_count=int(os.environ.get("WORLD_SIZE", 1)),
            )
            data_loader_len = data_loader.len_w_stats()
            actual_eff = data_loader.efficiency()
            LOG.info(f"data_loader_len: {data_loader_len}")
            total_num_steps = int(
                math.floor(
                    data_loader_len
                    * cfg.micro_batch_size
                    * cfg.num_epochs
                    // cfg.batch_size
                )
            )
            LOG.info(
                f"📝 UPDATE CONFIG WITH: `sample_packing_eff_est: {math.ceil(actual_eff * 100.0) / 100.0}`"
            )
            cfg.sample_packing_eff_est = math.ceil(actual_eff * 100.0) / 100.0
    else:
        total_num_steps = int(
            math.ceil(len(train_dataset) * cfg.num_epochs / cfg.batch_size)
        )
    LOG.info(f"total_num_steps: {total_num_steps}")
    return total_num_steps


def setup_fsdp_envs(cfg):
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    if cfg.fsdp_config.fsdp_sync_module_states:
        os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    if cfg.fsdp_config.fsdp_state_dict_type:
        os.environ["FSDP_STATE_DICT_TYPE"] = cfg.fsdp_config.fsdp_state_dict_type


def setup_trainer(cfg, train_dataset, eval_dataset, model, tokenizer, total_num_steps):
    if cfg.fsdp:
        setup_fsdp_envs(cfg)
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

    if cfg.lr_quadratic_warmup is not None:
        training_arguments_kwargs["lr_quadratic_warmup"] = cfg.lr_quadratic_warmup

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
        training_arguments_kwargs["hub_private_repo"] = True

        if cfg.hub_strategy:
            training_arguments_kwargs["hub_strategy"] = cfg.hub_strategy

    if cfg.save_safetensors:
        training_arguments_kwargs["save_safetensors"] = cfg.save_safetensors

    if cfg.sample_packing_eff_est:
        training_arguments_kwargs[
            "sample_packing_efficiency"
        ] = cfg.sample_packing_eff_est

    training_args = AxolotlTrainingArguments(  # pylint: disable=unexpected-keyword-arg
        # max_steps=total_num_steps,  # this is helpful in case we don't actually know total # of steps
        max_seq_length=cfg.sequence_len,
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
        save_total_limit=cfg.save_total_limit if cfg.save_total_limit else 4,
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
        sample_packing=cfg.sample_packing if cfg.sample_packing else False,
        sample_packing_seq_len_multiplier=cfg.micro_batch_size,
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
    callbacks.append(GPUStatsCallback(cfg))
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
        # A100 is best at 64, while others at 8. Let's use the larger so we don't have to check
        # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
        data_collator_kwargs["pad_to_multiple_of"] = 64

    if cfg.is_llama_derived_model and cfg.landmark_attention:
        from axolotl.monkeypatch.llama_landmark_attn import (
            add_mem_tokens,
            get_mem_id,
            set_model_mem_id,
        )

        set_model_mem_id(model, tokenizer)

        LOG.info("Adding landmark attention tokens to dataset")

        for dataset in [train_dataset, eval_dataset]:
            dataset = dataset.map(
                partial(add_mem_tokens, mem_freq=50, mem_id=get_mem_id(tokenizer)),
                batched=False,
                num_proc=32,
            )

    trainer_cls = (
        OneCycleLRSchedulerTrainer
        if cfg.lr_scheduler == "one_cycle" and (cfg.fsdp or cfg.adapter == "qlora")
        else AxolotlTrainer
    )
    trainer = trainer_cls(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        data_collator=DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            **data_collator_kwargs,
        ),
        callbacks=callbacks,
        **trainer_kwargs,
    )

    return trainer
