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
from typing import List, Optional, Union

import numpy as np
import torch
import torch.cuda
import torch.distributed as dist
import transformers
from datasets import Dataset, set_caching_enabled
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import (
    DataLoader,
    DistributedSampler,
    RandomSampler,
    SequentialSampler,
)
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_pt_utils import SequentialDistributedSampler

from axolotl.monkeypatch.relora import ReLoRACallback, ReLoRAScheduler
from axolotl.utils.callbacks import (
    EvalFirstStepCallback,
    GPUStatsCallback,
    SaveAxolotlConfigtoWandBCallback,
    SaveBetterTransformerModelCallback,
    bench_eval_callback_factory,
    log_prediction_callback_factory,
)
from axolotl.utils.collators import DataCollatorForSeq2Seq
from axolotl.utils.dataloader import MultipackDistributedDataloader
from axolotl.utils.distributed import (
    is_distributed,
    is_main_process,
    reduce_and_broadcast,
    zero_first,
)
from axolotl.utils.schedulers import get_cosine_schedule_with_quadratic_warmup

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
    eval_sample_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Use sample packing for efficient evals."},
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
    relora_steps: Optional[int] = field(
        default=None,
        metadata={"help": "how often to reset for ReLoRA"},
    )
    relora_warmup_steps: Optional[int] = field(
        default=None,
        metadata={"help": "how many warmup steps to take after reset for ReLoRA"},
    )
    bench_split: Optional[str] = field(
        default="eval", metadata={"help": "The benchmark split to run on"}
    )
    bench_dataset: Optional[str] = field(
        default="pharaouk/dharma-1/dharma_1_mini.json",
        metadata={
            "help": "Benchmark dataset to use: options are `mmlu-zs`, `mmlu-fs`, or the full path to the dataset file"
        },
    )
    do_bench_eval: Optional[bool] = field(
        default=False, metadata={"help": "Whether to run the Benchmark evaluation."}
    )
    max_bench_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, only evaluates on `max_bench_samples` of the benchmark dataset."
        },
    )
    bench_source_max_len: int = field(
        default=2048, metadata={"help": "Maximum source sequence length for bench."}
    )


class AxolotlTrainer(Trainer):
    """
    Extend the base Trainer for axolotl helpers
    """

    args = None  # type: AxolotlTrainingArguments

    def __init__(self, *args, bench_data_collator=None, **kwargs):
        self.bench_data_collator = bench_data_collator
        super().__init__(*args, **kwargs)

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

    def _get_eval_sampler(
        self, eval_dataset: Dataset
    ) -> Optional[torch.utils.data.Sampler]:
        if (
            self.args.world_size > 1
            and self.args.sample_packing
            and self.args.eval_sample_packing is not False
        ):
            return SequentialDistributedSampler(
                eval_dataset,
                num_replicas=self.args.world_size,
                rank=self.args.process_index,
                batch_size=self.args.per_device_eval_batch_size,
            )
        return super()._get_eval_sampler(eval_dataset)

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
        if self.args.sample_packing and self.args.eval_sample_packing is not False:
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

    def _get_bench_sampler(
        self, bench_dataset: Dataset
    ) -> Optional[torch.utils.data.Sampler]:
        if self.args.world_size <= 1:
            return SequentialSampler(bench_dataset)
        return None

    def get_bench_dataloader(
        self,
        bench_dataset: Dataset,
    ) -> Union[DataLoader, MultipackDistributedDataloader]:
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.bench_data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        if not isinstance(bench_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_bench_sampler(bench_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return DataLoader(bench_dataset, **dataloader_params)
        # return self.accelerator.prepare(DataLoader(bench_dataset, **dataloader_params))

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


class ReLoRATrainer(AxolotlTrainer):
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
        lr_scheduler = super().create_scheduler(num_training_steps, optimizer)

        if self.args.relora_steps:
            warmup_steps = (
                self.args.relora_warmup_steps if self.args.relora_warmup_steps else 10
            )
            self.lr_scheduler = ReLoRAScheduler(
                optimizer,
                lr_scheduler,
                self.args.relora_steps,
                warmup_steps,
            )
        else:
            self.lr_scheduler = lr_scheduler

        return self.lr_scheduler


def add_position_ids(sample):
    sample_len = len(sample["input_ids"])
    sample["position_ids"] = torch.arange(len(sample["input_ids"]))
    sample["length"] = sample_len
    return sample


def add_length(sample):
    sample["length"] = len(sample["input_ids"])
    return sample


def drop_long_seq(sample, sequence_len=2048):
    return len(sample["input_ids"]) <= sequence_len and len(sample["input_ids"]) > 0


@contextmanager
def disable_datasets_caching():
    try:
        set_caching_enabled(False)
        yield
    finally:
        set_caching_enabled(True)


def process_datasets_for_packing(cfg, train_dataset, eval_dataset, tokenizer):
    drop_long = partial(drop_long_seq, sequence_len=cfg.sequence_len)
    with zero_first(is_main_process()):
        train_dataset = train_dataset.filter(drop_long, num_proc=cfg.dataset_processes)
        if eval_dataset:
            eval_dataset = eval_dataset.filter(
                drop_long, num_proc=cfg.dataset_processes
            )

        if cfg.group_by_length:
            train_dataset = train_dataset.map(
                add_length, num_proc=cfg.dataset_processes
            )

        if cfg.sample_packing:
            train_dataset = train_dataset.map(
                add_position_ids, num_proc=cfg.dataset_processes
            )
            if cfg.eval_sample_packing is not False:
                if eval_dataset:
                    eval_dataset = eval_dataset.map(
                        add_position_ids, num_proc=cfg.dataset_processes
                    )

        # Phi doesn't want the attention_mask feature when training
        if "CodeGenTokenizer" in tokenizer.__class__.__name__ or (
            cfg.is_mistral_derived_model and cfg.flash_attention
        ):
            train_dataset = train_dataset.remove_columns("attention_mask")
            if eval_dataset:
                eval_dataset = eval_dataset.remove_columns("attention_mask")

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
            LOG.info(f"total_num_tokens: {total_num_tokens}")
            cfg.total_num_tokens = total_num_tokens

        if not cfg.total_supervised_tokens:
            total_supervised_tokens = (
                train_dataset.data.column("labels")
                .to_pandas()
                .apply(lambda x: np.sum(np.array(x) != -100))
                .sum()
            )
            LOG.info(f"`total_supervised_tokens: {total_supervised_tokens}`")
            cfg.total_supervised_tokens = total_supervised_tokens

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
            if cfg.world_size > 1 and is_distributed():
                sampler = DistributedSampler(
                    train_dataset,
                    num_replicas=cfg.world_size,
                    rank=dist.get_rank(),
                    seed=cfg.seed or 42,
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
            # FIXME: is there a bug here somewhere? the total num steps depends
            # on the agreed on value for sample_packing_eff_est
            total_num_steps = int(math.floor(data_loader_len * cfg.num_epochs))

            def calc_sample_packing_eff_est(estimates: List[float]):
                LOG.info(f"sample_packing_eff_est across ranks: {repr(estimates)}")
                return max(estimates)

            sample_packing_actual_eff_all = reduce_and_broadcast(
                lambda: actual_eff,
                calc_sample_packing_eff_est,
            )
            sample_packing_eff_est = (
                math.ceil(sample_packing_actual_eff_all * 100.0) / 100.0
            )
            cfg.sample_packing_eff_est = sample_packing_eff_est
            LOG.info(f"sample_packing_eff_est: {cfg.sample_packing_eff_est}")
    else:
        total_num_steps = int(
            math.ceil(len(train_dataset) * cfg.num_epochs / cfg.batch_size)
        )
    LOG.info(f"total_num_steps: {total_num_steps}")
    return total_num_steps


def setup_fsdp_envs(cfg):
    os.environ["ACCELERATE_USE_FSDP"] = "true"
    if cfg.fsdp_config.fsdp_offload_params:
        os.environ["FSDP_OFFLOAD_PARAMS"] = "true"
    if cfg.fsdp_config.fsdp_sync_module_states:
        os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    if cfg.fsdp_config.fsdp_state_dict_type:
        os.environ["FSDP_STATE_DICT_TYPE"] = cfg.fsdp_config.fsdp_state_dict_type
    if cfg.fsdp_config.fsdp_transformer_layer_cls_to_wrap:
        os.environ[
            "FSDP_TRANSFORMER_CLS_TO_WRAP"
        ] = cfg.fsdp_config.fsdp_transformer_layer_cls_to_wrap


def setup_trainer(cfg, train_dataset, eval_dataset, model, tokenizer, total_num_steps):
    if cfg.fsdp:
        setup_fsdp_envs(cfg)
    elif cfg.deepspeed:
        os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"

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
        training_arguments_kwargs["gradient_checkpointing"] = cfg.gradient_checkpointing
    if cfg.fsdp:
        training_arguments_kwargs["fsdp"] = cfg.fsdp
        if cfg.fsdp_config:
            training_arguments_kwargs["fsdp_config"] = dict(cfg.fsdp_config)

    # deepspeed
    if cfg.deepspeed:
        training_arguments_kwargs["deepspeed"] = cfg.deepspeed

    if cfg.lr_quadratic_warmup is not None:
        training_arguments_kwargs["lr_quadratic_warmup"] = cfg.lr_quadratic_warmup

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

    if cfg.eval_steps:
        training_arguments_kwargs["evaluation_strategy"] = "steps"
        training_arguments_kwargs["eval_steps"] = cfg.eval_steps
    elif cfg.evaluation_strategy:
        training_arguments_kwargs["evaluation_strategy"] = cfg.evaluation_strategy
    elif cfg.val_set_size == 0:
        # no eval set, so don't eval
        training_arguments_kwargs["evaluation_strategy"] = "no"
    else:
        # we have an eval set, but no steps defined, default to use epoch
        training_arguments_kwargs["evaluation_strategy"] = "epoch"

    if cfg.save_steps:
        training_arguments_kwargs["save_strategy"] = "steps"
        training_arguments_kwargs["save_steps"] = cfg.save_steps
    elif cfg.save_strategy:
        training_arguments_kwargs["save_strategy"] = cfg.save_strategy
    else:
        # default to saving each epoch if not defined
        training_arguments_kwargs["save_strategy"] = "epoch"

    if cfg.do_bench_eval:
        training_arguments_kwargs["do_bench_eval"] = cfg.do_bench_eval
        if cfg.bench_dataset:
            training_arguments_kwargs["bench_dataset"] = cfg.bench_dataset
    if cfg.metric_for_best_model:
        training_arguments_kwargs["metric_for_best_model"] = cfg.metric_for_best_model
    if cfg.greater_is_better:
        training_arguments_kwargs["greater_is_better"] = cfg.greater_is_better

    if cfg.torch_compile:
        if torch.__version__ < "2.1.0":  # pylint: disable=protected-access
            LOG.warning("torch>=2.1.0 required for torch_compile to work properly")
        else:
            import torch._dynamo  # pylint: disable=redefined-outer-name

            torch._dynamo.config.suppress_errors = (  # pylint: disable=protected-access
                True
            )
            training_arguments_kwargs["torch_compile"] = cfg.torch_compile
            if cfg.torch_compile_backend:
                training_arguments_kwargs[
                    "torch_compile_backend"
                ] = cfg.torch_compile_backend

    # DDP Config
    if cfg.ddp_timeout:
        training_arguments_kwargs["ddp_timeout"] = cfg.ddp_timeout
    # see https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
    if cfg.ddp_bucket_cap_mb:
        training_arguments_kwargs["ddp_bucket_cap_mb"] = cfg.ddp_bucket_cap_mb
    if cfg.ddp_broadcast_buffers is not None:
        training_arguments_kwargs["ddp_broadcast_buffers"] = cfg.ddp_broadcast_buffers

    training_args = AxolotlTrainingArguments(  # pylint: disable=unexpected-keyword-arg
        max_steps=total_num_steps if cfg.max_steps else -1,
        max_seq_length=cfg.sequence_len,
        per_device_train_batch_size=cfg.micro_batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        eval_accumulation_steps=cfg.gradient_accumulation_steps,
        num_train_epochs=cfg.num_epochs,
        learning_rate=cfg.learning_rate,
        output_dir=cfg.output_dir,
        save_total_limit=cfg.save_total_limit if cfg.save_total_limit else 4,
        load_best_model_at_end=(
            (cfg.load_best_model_at_end is not False or cfg.early_stopping_patience)
            and cfg.val_set_size > 0
            and cfg.save_steps
            and cfg.eval_steps
            and cfg.save_steps % cfg.eval_steps == 0
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
        eval_sample_packing=cfg.eval_sample_packing,
        sample_packing_seq_len_multiplier=cfg.micro_batch_size,
        relora_steps=cfg.relora_steps,
        relora_warmup_steps=cfg.relora_warmup_steps,
        **training_arguments_kwargs,
    )

    trainer_kwargs = {}

    if cfg.optimizer == "adamw_anyprecision":
        if Path(cfg.torchdistx_path).exists():
            sys.path.append(cfg.torchdistx_path)
            importlib.import_module("torchdistx")

    callbacks = []
    callbacks.append(GPUStatsCallback(cfg))
    callbacks.append(EvalFirstStepCallback)

    if cfg.relora_steps:
        callbacks.append(ReLoRACallback(cfg))

    if hasattr(model, "use_bettertransformer") and model.use_bettertransformer is True:
        callbacks.append(SaveBetterTransformerModelCallback)

    data_collator_kwargs = {
        "padding": True,  # True/"longest" is the default
    }
    if cfg.pad_to_sequence_len:
        data_collator_kwargs["pad_to_multiple_of"] = 64 * math.ceil(
            cfg.sequence_len / 64
        )
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

    trainer_cls = AxolotlTrainer
    if cfg.lr_scheduler == "one_cycle" and (cfg.fsdp or cfg.adapter == "qlora"):
        trainer_cls = OneCycleLRSchedulerTrainer
    elif cfg.relora_steps:
        trainer_cls = ReLoRATrainer
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
        bench_data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer,
            return_tensors="pt",
            **data_collator_kwargs,
        ),
        callbacks=callbacks,
        **trainer_kwargs,
    )

    if cfg.use_wandb and cfg.eval_table_size > 0:
        LogPredictionCallback = log_prediction_callback_factory(trainer, tokenizer)
        trainer.add_callback(LogPredictionCallback(cfg))

    if cfg.use_wandb:
        trainer.add_callback(SaveAxolotlConfigtoWandBCallback(cfg.axolotl_config_path))

    if cfg.do_bench_eval:
        trainer.add_callback(bench_eval_callback_factory(trainer, tokenizer))

    # TODO on_save callback to sync checkpoints to GCP/AWS in background
    if cfg.early_stopping_patience:
        early_stop_cb = EarlyStoppingCallback(
            cfg.early_stopping_patience,
        )
        trainer.add_callback(early_stop_cb)

    return trainer
