# pylint: disable=too-many-lines
"""
Builder for the training args and trainer
"""

import abc
import gc
import importlib
import importlib.util
import inspect
import logging
import math
import os
import sys
from abc import abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Type, Union

import torch
import transformers
from datasets import Dataset
from packaging import version
from peft.optimizers import create_loraplus_optimizer
from torch import nn
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, seed_worker
from transformers.utils import is_sagemaker_mp_enabled
from trl import (
    CPOConfig,
    CPOTrainer,
    DPOConfig,
    DPOTrainer,
    KTOConfig,
    KTOTrainer,
    ORPOConfig,
    ORPOTrainer,
    RewardConfig,
    RewardTrainer,
)
from trl.trainer.utils import RewardDataCollatorWithPadding, pad_to_length

from axolotl.integrations.base import PluginManager
from axolotl.monkeypatch.multipack import SUPPORTED_MULTIPACK_MODEL_TYPES
from axolotl.monkeypatch.relora import ReLoRACallback, ReLoRAScheduler
from axolotl.utils import is_comet_available, is_mlflow_available
from axolotl.utils.callbacks import (
    EvalFirstStepCallback,
    GPUStatsCallback,
    LossWatchDogCallback,
    SaveAxolotlConfigtoWandBCallback,
    SaveBetterTransformerModelCallback,
    SaveModelCallback,
    bench_eval_callback_factory,
    causal_lm_bench_eval_callback_factory,
    log_prediction_callback_factory,
)
from axolotl.utils.callbacks.lisa import lisa_callback_factory
from axolotl.utils.chat_templates import get_chat_template
from axolotl.utils.collators import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    MambaDataCollator,
    V2BatchSamplerDataCollatorForSeq2Seq,
)
from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator
from axolotl.utils.models import ensure_dtype
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.schedulers import (
    get_cosine_schedule_with_min_lr,
    get_cosine_schedule_with_quadratic_warmup,
    get_cosine_schedule_with_warmup_decay_constant,
)

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

try:
    import torch._dynamo  # pylint: disable=ungrouped-imports
except ImportError:
    pass

LOG = logging.getLogger("axolotl.core.trainer_builder")


def _sanitize_kwargs_for_tagging(tag_names, kwargs=None):
    if isinstance(tag_names, str):
        tag_names = [tag_names]

    if kwargs is not None:
        if "tags" not in kwargs:
            kwargs["tags"] = tag_names
        elif "tags" in kwargs and isinstance(kwargs["tags"], list):
            kwargs["tags"].extend(tag_names)
        elif "tags" in kwargs and isinstance(kwargs["tags"], str):
            tag_names.append(kwargs["tags"])
            kwargs["tags"] = tag_names

    return kwargs


def _sanitize_kwargs_for_ds_tagging(dataset_tags, kwargs=None):
    if isinstance(dataset_tags, str):
        dataset_tags = [dataset_tags]

    if (dataset_tags is not None) and (kwargs is not None):
        if "dataset_tags" not in kwargs:
            kwargs["dataset_tags"] = dataset_tags
        elif "dataset_tags" in kwargs and isinstance(kwargs["dataset_tags"], list):
            kwargs["dataset_tags"].extend(dataset_tags)
        elif "dataset_tags" in kwargs and isinstance(kwargs["dataset_tags"], str):
            dataset_tags.append(kwargs["dataset_tags"])
            kwargs["dataset_tags"] = dataset_tags

    return kwargs


@dataclass
class AxolotlTrainingMixins:
    """
    Mixin class for the Axolotl training args.
    """

    # pylint: disable=duplicate-code
    model_type: Optional[str] = field(
        default=None, metadata={"help": "HF model configuration model_type."}
    )
    lr_quadratic_warmup: bool = field(
        default=False,
        metadata={"help": "Use quadratic warmup for cosine scheduling."},
    )
    pretraining: bool = field(
        default=False,
        metadata={
            "help": "Indicates to trainer whether we are doing continued pretraining."
        },
    )
    sample_packing: bool = field(
        default=False,
        metadata={"help": "Use sample packing for efficient training."},
    )
    multipack_real_batches: bool = field(
        default=False,
        metadata={"help": "Use real batches for efficient training."},
    )
    eval_sample_packing: Optional[bool] = field(
        default=None,
        metadata={"help": "Use sample packing for efficient evals."},
    )
    sample_packing_efficiency: float = field(
        default=1.0,
        metadata={"help": "Sample packing efficiency for calculating batch length."},
    )
    sample_packing_bin_size: int = field(
        default=200,
        metadata={
            "help": "The max number of samples that packed sample can contain after packing. Increase for better packing."
        },
    )
    sample_packing_group_size: int = field(
        default=100000,
        metadata={
            "help": "The number of samples to group together for packing. Increase for better packing."
        },
    )
    max_seq_length: int = field(
        default=2048,
        metadata={"help": "The maximum sequence length the model can handle"},
    )
    relora_steps: Optional[int] = field(
        default=None,
        metadata={"help": "how often to reset for ReLoRA"},
    )
    relora_warmup_steps: Optional[int] = field(
        default=None,
        metadata={"help": "how many warmup steps to take after reset for ReLoRA"},
    )
    relora_anneal_steps: Optional[int] = field(
        default=None,
        metadata={"help": "how many warmup steps to take after reset for ReLoRA"},
    )
    relora_prune_ratio: Optional[float] = field(
        default=0.9,
        metadata={"help": "prune ratio for magnitude pruning of the optimizer"},
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
    do_causal_lm_eval: Optional[bool] = field(
        default=False, metadata={"help": "Whether to run the Causal LM evaluation."}
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
    dataloader_prefetch_factor: Optional[int] = field(
        default=None,
        metadata={"help": "prefetch_factor argument to the dataloader"},
    )
    cosine_min_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "Minimum learning rate is min_lr_ratio * learning_rate"},
    )
    cosine_constant_lr_ratio: Optional[float] = field(
        default=None,
        metadata={
            "help": "Starting constant learning rate step is cosine_constant_lr_ratio * max_steps"
        },
    )
    loraplus_lr_ratio: Optional[float] = field(
        default=None, metadata={"help": "loraplus learning rate ratio lr_B / lr_A."}
    )
    loraplus_lr_embedding: Optional[float] = field(
        default=1e-6,
        metadata={"help": "loraplus learning rate for lora embedding layers."},
    )
    embedding_lr_scale: Optional[float] = field(
        default=None,
        metadata={"help": "Scale the learning rate for the embedding layers."},
    )
    embedding_lr: Optional[float] = field(
        default=None,
        metadata={"help": "absolute learning rate for the embedding layers."},
    )
    qlora: bool = field(
        default=False,
        metadata={"help": "whether this is a qlora training"},
    )
    orpo_alpha: Optional[float] = field(
        default=None,
    )
    lisa_n_layers: Optional[int] = field(
        default=None,
        metadata={"help": "the number of activate layers in LISA"},
    )
    lisa_step_interval: Optional[int] = field(
        default=None,
        metadata={"help": "how often to switch layers in LISA"},
    )
    lisa_layers_attribute: Optional[str] = field(
        default=None,
        metadata={"help": "path under the model to access the layers"},
    )
    curriculum_sampling: Optional[bool] = field(
        default=None,
        metadata={"help": "whether to use sequential sampling for curriculum learning"},
    )
    alternate_optimizer: Optional[str] = field(
        default=None,
        metadata={
            "help": "workaround to pass an alternate optimizer to the HF trainer"
        },
    )
    alternate_lr_scheduler_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "workaround to pass an alternate lr scheduler to the HF trainer"
        },
    )
    chat_template: Optional[str] = field(
        default=None,
        metadata={"help": "Chat template converting chat messages to text"},
    )


@dataclass
class AxolotlTrainingArguments(AxolotlTrainingMixins, TrainingArguments):
    """
    Training arguments for Causal trainer

    This code is duplicated due to HF TrainingArguments not setting output_dir with a defaujlt value
    so it can't be used as a mixin.
    """


@dataclass
class AxolotlDPOConfig(AxolotlTrainingMixins, DPOConfig):
    """
    DPO config for DPO training
    """


@dataclass
class AxolotlORPOConfig(AxolotlTrainingMixins, ORPOConfig):
    """
    ORPO config for ORPO training
    """


@dataclass
class AxolotlKTOConfig(AxolotlTrainingMixins, KTOConfig):
    """
    KTO config for KTO training
    """


@dataclass
class AxolotlCPOConfig(AxolotlTrainingMixins, CPOConfig):
    """
    CPO config for CPO training
    """

    simpo_gamma: Optional[float] = field(
        default=None,
        metadata={"help": "simpo gamma parameter"},
    )


@dataclass
class AxolotlRewardConfig(AxolotlTrainingMixins, RewardConfig):
    """
    Reward config for Reward training
    """


class SchedulerMixin(Trainer):
    """
    Mixin class for scheduler setup in CausalTrainer.
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
        use_cosine_quadratic = (
            self.args.lr_scheduler_type == "cosine"
            and self.args.lr_quadratic_warmup is True
        )

        use_cosine_min_lr = (
            self.args.lr_scheduler_type == "cosine"
            and self.args.cosine_min_lr_ratio is not None
        )

        # fmt: off
        if self.lr_scheduler is None:  # type: ignore  # pylint: disable=access-member-before-definition
            # fmt: on
            if self.args.alternate_lr_scheduler_type == "one_cycle":
                num_warmup_steps = self.args.get_warmup_steps(num_training_steps)
                pct_start = num_warmup_steps / num_training_steps
                extra_lr_kwargs = {}
                if "pct_start" not in self.args.lr_scheduler_kwargs:
                    extra_lr_kwargs["pct_start"] = pct_start
                if "anneal_strategy" not in self.args.lr_scheduler_kwargs:
                    extra_lr_kwargs["anneal_strategy"] = "cos"

                self.lr_scheduler = OneCycleLR(
                    optimizer,
                    max_lr=self.args.learning_rate,
                    total_steps=num_training_steps,
                    **extra_lr_kwargs,
                    **self.args.lr_scheduler_kwargs,
                )
            elif use_cosine_quadratic:
                if use_cosine_min_lr:
                    LOG.warning("Both cosine quadratic warmup and min lr detected. Using quadratic warmup.")

                self.lr_scheduler = get_cosine_schedule_with_quadratic_warmup(  # pylint: disable=attribute-defined-outside-init
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                )
            elif self.args.cosine_min_lr_ratio and self.args.cosine_constant_lr_ratio and use_cosine_min_lr:
                assert 0 <= self.args.cosine_min_lr_ratio <= 1.0, "cosine_min_lr_ratio must be between 0.0 and 1.0"
                assert 0 <= self.args.cosine_constant_lr_ratio <= 1.0, "cosine_constant_lr_ratio must be between 0.0 and 1.0"
                self.lr_scheduler = get_cosine_schedule_with_warmup_decay_constant(  # pylint: disable=attribute-defined-outside-init
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=self.args.cosine_min_lr_ratio,
                    constant_lr_ratio=self.args.cosine_constant_lr_ratio,
                )
            elif self.args.cosine_min_lr_ratio and use_cosine_min_lr:
                assert 0 <= self.args.cosine_min_lr_ratio <= 1.0, "cosine_min_lr_ratio must be between 0.0 and 1.0"
                self.lr_scheduler = get_cosine_schedule_with_min_lr(  # pylint: disable=attribute-defined-outside-init
                    optimizer,
                    num_warmup_steps=self.args.get_warmup_steps(num_training_steps),
                    num_training_steps=num_training_steps,
                    min_lr_ratio=self.args.cosine_min_lr_ratio,
                )
            else:
                return super().create_scheduler(num_training_steps, optimizer=optimizer)
        else:
            if use_cosine_quadratic:
                LOG.warning("axolotl's cosine scheduler with quadratic warmup not used (e.g., because of deepspeed).")

            if use_cosine_min_lr:
                LOG.warning("axolotl's cosine scheduler with min lr not used (e.g., because of deepspeed).")

        return self.lr_scheduler


class AxolotlTrainer(SchedulerMixin, Trainer):
    """
    Extend the base Trainer for axolotl helpers
    """

    args = None  # type: AxolotlTrainingArguments
    tag_names = ["axolotl"]

    def __init__(
        self,
        *_args,
        bench_data_collator=None,
        eval_data_collator=None,
        dataset_tags=None,
        **kwargs,
    ):
        self.bench_data_collator = bench_data_collator
        self.eval_data_collator = eval_data_collator
        self.dataset_tags = dataset_tags
        super().__init__(*_args, **kwargs)
        self.train_data_collator = self.data_collator
        self._stored_metrics = defaultdict(lambda: defaultdict(list))
        if self.args.orpo_alpha:
            self.loss_fct = torch.nn.CrossEntropyLoss(reduction="none")

    def _wrap_model(self, model, training=True, dataloader=None):
        if self.args.torch_compile:
            torch._dynamo.config.accumulated_cache_size_limit = (  # pylint: disable=protected-access
                256
            )
            model = torch.compile(
                model,
                backend=self.args.torch_compile_backend,
                mode=self.args.torch_compile_mode,
            )
        return super()._wrap_model(model, training=training, dataloader=dataloader)

    def create_optimizer(self):
        if (
            self.args.loraplus_lr_ratio is None
            and self.args.embedding_lr_scale is None
            and self.args.embedding_lr is None
            and self.args.alternate_optimizer
            not in [
                "optimi_adamw",
                "ao_adamw_8bit",
                "ao_adamw_4bit",
                "ao_adamw_fp8",
                "adopt_adamw",
            ]
        ):
            return super().create_optimizer()

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:  # pylint: disable=access-member-before-definition
            decay_parameters = self.get_decay_parameter_names(opt_model)
            params = {
                "to_weight_decay": {},  # LayerNorm and bias
                "embeddings": {},  # lm_head, embed_tokens,
                "no_weight_decay": {},
            }

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args,
                opt_model,
            )

            for name, param in opt_model.named_parameters():
                if not param.requires_grad:
                    continue
                if name.endswith("modules_to_save.default.weight") or any(
                    embed_name in name for embed_name in ["embed_tokens", "lm_head"]
                ):
                    params["embeddings"][name] = param
                elif name in decay_parameters:
                    params["to_weight_decay"][name] = param
                else:
                    params["no_weight_decay"][name] = param
            optimizer_grouped_parameters = []
            if params["to_weight_decay"]:
                optimizer_grouped_parameters.append(
                    {
                        "params": list(params["to_weight_decay"].values()),
                        "weight_decay": self.args.weight_decay,
                        "lr": optimizer_kwargs["lr"],
                    }
                )
            if params["embeddings"]:
                lr = optimizer_kwargs["lr"]  # pylint: disable=invalid-name
                if self.args.embedding_lr_scale:
                    lr *= self.args.embedding_lr_scale  # pylint: disable=invalid-name
                elif self.args.embedding_lr:
                    lr = self.args.embedding_lr  # pylint: disable=invalid-name
                optimizer_grouped_parameters.append(
                    {
                        "params": list(params["embeddings"].values()),
                        "weight_decay": 0.0,
                        "lr": lr,
                    }
                )
            if params["no_weight_decay"]:
                optimizer_grouped_parameters.append(
                    {
                        "params": list(params["no_weight_decay"].values()),
                        "weight_decay": 0.0,
                        "lr": optimizer_kwargs["lr"],
                    }
                )

            if self.args.loraplus_lr_ratio is not None:
                loraplus_lr_ratio = getattr(self.args, "loraplus_lr_ratio", None)
                loraplus_lr_embedding = getattr(
                    self.args, "loraplus_lr_embedding", 1e-6
                )
                self.optimizer = create_loraplus_optimizer(  # pylint: disable=attribute-defined-outside-init
                    opt_model,
                    optimizer_cls,
                    loraplus_lr_ratio=loraplus_lr_ratio,
                    loraplus_lr_embedding=loraplus_lr_embedding,
                    **optimizer_kwargs,
                )
            elif (
                self.args.embedding_lr_scale is not None
                or self.args.embedding_lr is not None
            ):
                self.optimizer = (  # pylint: disable=attribute-defined-outside-init
                    optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
                )
            elif self.args.alternate_optimizer == "optimi_adamw":
                from optimi import AdamW

                self.optimizer = (  # pylint: disable=attribute-defined-outside-init
                    AdamW(
                        optimizer_grouped_parameters, foreach=False, **optimizer_kwargs
                    )
                )
            elif self.args.alternate_optimizer == "ao_adamw_4bit":
                from torchao.prototype.low_bit_optim import AdamW4bit

                self.optimizer = (  # pylint: disable=attribute-defined-outside-init
                    AdamW4bit(optimizer_grouped_parameters, **optimizer_kwargs)
                )
            elif self.args.alternate_optimizer == "ao_adamw_8bit":
                from torchao.prototype.low_bit_optim import AdamW8bit

                self.optimizer = (  # pylint: disable=attribute-defined-outside-init
                    AdamW8bit(optimizer_grouped_parameters, **optimizer_kwargs)
                )
            elif self.args.alternate_optimizer == "ao_adamw_fp8":
                from torchao.prototype.low_bit_optim import AdamWFp8

                self.optimizer = (  # pylint: disable=attribute-defined-outside-init
                    AdamWFp8(optimizer_grouped_parameters, **optimizer_kwargs)
                )
            elif self.args.alternate_optimizer == "adopt_adamw":
                from axolotl.utils.optimizers.adopt import ADOPT

                self.optimizer = (  # pylint: disable=attribute-defined-outside-init
                    ADOPT(
                        optimizer_grouped_parameters,
                        decouple=True,
                        **optimizer_kwargs,
                    )
                )

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(  # pylint: disable=attribute-defined-outside-init
                self.optimizer
            )

        return self.optimizer

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.args.sample_packing and not self.args.pretraining:
            if self.args.multipack_real_batches:
                batch_size = self.args.per_device_train_batch_size
                batch_max_len = self.args.max_seq_length
            else:
                batch_size = 1
                train_batch_size = (
                    self.state.train_batch_size or self.args.per_device_train_batch_size
                )
                batch_max_len = train_batch_size * self.args.max_seq_length
            return MultipackBatchSampler(
                RandomSampler(self.train_dataset),
                lengths=get_dataset_lengths(self.train_dataset),
                packing_efficiency_estimate=self.args.sample_packing_efficiency,
                batch_max_len=batch_max_len,
                batch_size=batch_size,
                group_size=self.args.sample_packing_group_size,
                bin_size=self.args.sample_packing_bin_size,
                drop_last=True,
            )
        if self.args.curriculum_sampling:
            return SequentialSampler(self.train_dataset)
        return super()._get_train_sampler()

    def _get_eval_sampler(
        self, eval_dataset: Dataset
    ) -> Optional[torch.utils.data.Sampler]:
        if self.args.sample_packing and self.args.eval_sample_packing is not False:
            if self.args.multipack_real_batches:
                batch_size = self.args.per_device_eval_batch_size
                batch_max_len = self.args.max_seq_length
            else:
                batch_size = 1
                batch_max_len = (
                    self.args.per_device_eval_batch_size * self.args.max_seq_length
                )
            return MultipackBatchSampler(
                SequentialSampler(eval_dataset),
                lengths=get_dataset_lengths(self.eval_dataset),
                packing_efficiency_estimate=self.args.sample_packing_efficiency,
                batch_max_len=batch_max_len,
                batch_size=batch_size,
                group_size=self.args.sample_packing_group_size,
                bin_size=self.args.sample_packing_bin_size,
                drop_last=True,
            )
        return super()._get_eval_sampler(eval_dataset)

    def get_train_dataloader(self) -> DataLoader:
        if self.args.sample_packing and not self.args.pretraining:
            train_dataset = self.train_dataset
            if "length" in train_dataset.features.keys():
                train_dataset = train_dataset.remove_columns(["length"])
            data_collator = self.data_collator
            dataloader_params = {
                "batch_size": self._train_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            }
            if self.args.dataloader_prefetch_factor:
                dataloader_params[
                    "prefetch_factor"
                ] = self.args.dataloader_prefetch_factor

            sampler = self._get_train_sampler()
            if isinstance(sampler, BatchSampler):
                dataloader_params["batch_sampler"] = sampler
                del dataloader_params["batch_size"]
            else:
                dataloader_params["sampler"] = sampler
                dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker

            self.accelerator.even_batches = False
            return self.accelerator.prepare_data_loader(
                DataLoader(train_dataset, **dataloader_params)
            )
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if self.args.sample_packing and self.args.eval_sample_packing is False:
            self.data_collator = (  # pylint: disable=attribute-defined-outside-init
                self.eval_data_collator
            )
            if eval_dataset:
                eval_dataset = eval_dataset.remove_columns(["length"])
            dataloader = super().get_eval_dataloader(eval_dataset)
            self.data_collator = (  # pylint: disable=attribute-defined-outside-init
                self.train_data_collator
            )
            return dataloader

        if self.args.sample_packing and self.args.eval_sample_packing is not False:
            eval_dataset = (
                eval_dataset if eval_dataset is not None else self.eval_dataset
            )

            eval_sampler = self._get_eval_sampler(eval_dataset)
            eval_dataset = eval_dataset.remove_columns(["length"])
            data_collator = self.data_collator
            dataloader_params = {
                "batch_size": self.args.eval_batch_size,
                "collate_fn": data_collator,
                "num_workers": self.args.dataloader_num_workers,
                "pin_memory": self.args.dataloader_pin_memory,
            }
            if self.args.dataloader_prefetch_factor:
                dataloader_params[
                    "prefetch_factor"
                ] = self.args.dataloader_prefetch_factor

            if isinstance(eval_sampler, BatchSampler):
                dataloader_params["batch_sampler"] = eval_sampler
                del dataloader_params["batch_size"]
            else:
                dataloader_params["sampler"] = eval_sampler
                dataloader_params["drop_last"] = self.args.dataloader_drop_last

            self.accelerator.even_batches = False
            return self.accelerator.prepare_data_loader(
                DataLoader(eval_dataset, **dataloader_params)
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
    ) -> DataLoader:
        dataloader_params = {
            "batch_size": self.args.eval_batch_size,
            "collate_fn": self.bench_data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }
        if self.args.dataloader_prefetch_factor:
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        if not isinstance(bench_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_bench_sampler(bench_dataset)
            dataloader_params["drop_last"] = self.args.dataloader_drop_last

        return DataLoader(bench_dataset, **dataloader_params)
        # return self.accelerator.prepare(DataLoader(bench_dataset, **dataloader_params))

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # use one's weighted cross entropy loss calc
        # if self.args.sample_packing:
        #     labels = inputs.pop("labels")
        #     outputs = model(**inputs)
        #     loss = trainer_weighted_loss(outputs, labels, shift_labels=True)
        #     return (loss, outputs) if return_outputs else loss
        if self.args.orpo_alpha:
            return self.orpo_compute_loss(
                model,
                inputs,
                return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )
        return super().compute_loss(
            model,
            inputs,
            return_outputs=return_outputs,
            num_items_in_batch=num_items_in_batch,
        )

    @staticmethod
    def orpo_concatenate_inputs(inputs, label_pad_token=-100, pad_token=0, device=None):
        concatenated_batch = {}

        max_length = max(
            inputs["input_ids"].shape[1], inputs["rejected_input_ids"].shape[1]
        )
        # Concatenate positive and negative inputs
        concatenated_batch["input_ids"] = pad_to_length(
            inputs["input_ids"], max_length, pad_token
        )
        concatenated_batch["rejected_input_ids"] = pad_to_length(
            inputs["rejected_input_ids"], max_length, pad_token
        )
        concatenated_batch["labels"] = pad_to_length(
            inputs["labels"], max_length, label_pad_token
        )
        concatenated_batch["rejected_labels"] = pad_to_length(
            inputs["rejected_labels"], max_length, label_pad_token
        )
        concatenated_batch["attention_mask"] = pad_to_length(
            inputs["attention_mask"], max_length, 0
        )
        concatenated_batch["rejected_attention_mask"] = pad_to_length(
            inputs["rejected_attention_mask"], max_length, 0
        )
        concatenated_batch["prompt_attention_mask"] = pad_to_length(
            inputs["prompt_attention_mask"], max_length, 0
        ).to(device=device)

        input_ids = torch.cat(
            [concatenated_batch["input_ids"], concatenated_batch["rejected_input_ids"]],
            dim=0,
        ).to(device=device)
        attention_mask = torch.cat(
            [
                concatenated_batch["attention_mask"],
                concatenated_batch["rejected_attention_mask"],
            ],
            dim=0,
        ).to(device=device)
        labels = torch.cat(
            [concatenated_batch["labels"], concatenated_batch["rejected_labels"]], dim=0
        ).to(device=device)

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "prompt_attention_mask": concatenated_batch["prompt_attention_mask"],
        }

    def orpo_compute_custom_loss(self, logits, labels):
        logits = logits.contiguous()
        loss = 0.0

        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            # Flatten the tokens
            loss = self.loss_fct(shift_logits.transpose(2, 1), shift_labels).mean(
                dim=-1
            )

        return loss

    def orpo_compute_logps(
        self, prompt_attention_mask, chosen_inputs, chosen_attention_mask, logits
    ):
        # Get the shape of chosen_attention_mask[:, :-1]
        chosen_shape = chosen_attention_mask[:, :-1].shape

        # Calculate the padding size
        pad_length = chosen_shape[1] - (prompt_attention_mask.shape[1] - 1)

        # Pad prompt_attention_mask with zeros to match the desired shape
        prompt_attention_mask_padded = torch.nn.functional.pad(
            prompt_attention_mask[:, 1:], (0, pad_length), mode="constant", value=0
        )

        # Perform the subtraction operation
        mask = chosen_attention_mask[:, :-1] > prompt_attention_mask_padded

        per_token_logps = torch.gather(
            logits[:, :-1, :].log_softmax(-1),
            dim=2,
            index=(mask * chosen_inputs[:, 1:]).unsqueeze(2),
        ).squeeze(2)
        return torch.mul(per_token_logps, mask).sum(dim=1) / mask.sum(dim=1)

    def orpo_compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,
        num_items_in_batch=None,  # pylint: disable=unused-argument
    ):
        concat_inputs = AxolotlTrainer.orpo_concatenate_inputs(
            inputs,
            label_pad_token=-100,
            pad_token=self.tokenizer.pad_token_id,
            device=self.accelerator.device,
        )

        # Perform a single forward pass
        outputs = model(
            **{
                "input_ids": concat_inputs["input_ids"],
                "attention_mask": concat_inputs["attention_mask"],
                "labels": concat_inputs["labels"],
            },
            output_hidden_states=True,
        )

        # Split the outputs for positive and negative examples
        outputs_pos, outputs_neg = outputs.logits.chunk(2)

        # Calculate NLL loss
        pos_loss = self.orpo_compute_custom_loss(
            logits=outputs_pos, labels=concat_inputs["input_ids"].chunk(2)[0]
        )

        # Calculate Log Probability
        pos_prob = self.orpo_compute_logps(
            prompt_attention_mask=concat_inputs["prompt_attention_mask"],
            chosen_inputs=concat_inputs["input_ids"].chunk(2)[0],
            chosen_attention_mask=concat_inputs["attention_mask"].chunk(2)[0],
            logits=outputs_pos,
        )
        neg_prob = self.orpo_compute_logps(
            prompt_attention_mask=concat_inputs["prompt_attention_mask"],
            chosen_inputs=concat_inputs["input_ids"].chunk(2)[1],
            chosen_attention_mask=concat_inputs["attention_mask"].chunk(2)[1],
            logits=outputs_neg,
        )

        # Calculate log odds
        log_odds = (pos_prob - neg_prob) - (
            torch.log(1 - torch.exp(pos_prob)) - torch.log(1 - torch.exp(neg_prob))
        )
        sig_ratio = torch.nn.functional.sigmoid(log_odds)
        ratio = torch.log(sig_ratio)

        # Calculate the Final Loss
        loss = torch.mean(pos_loss - self.args.orpo_alpha * ratio).to(
            dtype=torch.bfloat16
        )

        metrics = {}
        metrics["chosen_geometric_mean"] = torch.mean(pos_prob).cpu().item()
        metrics["rejected_geometric_mean"] = torch.mean(neg_prob).cpu().item()
        metrics["log_odds_ratio"] = torch.mean(ratio).cpu().item()
        metrics["log_odds"] = torch.mean(log_odds).cpu().item()
        self.store_metrics(metrics, train_eval="train")

        return (loss, outputs_pos) if return_outputs else loss

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, *args, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tags when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = _sanitize_kwargs_for_ds_tagging(
            dataset_tags=self.dataset_tags, kwargs=kwargs
        )
        kwargs = _sanitize_kwargs_for_tagging(tag_names=self.tag_names, kwargs=kwargs)

        return super().push_to_hub(*args, **kwargs)

    @wraps(Trainer.create_accelerator_and_postprocess)
    def create_accelerator_and_postprocess(self):
        res = super().create_accelerator_and_postprocess()

        if self.is_fsdp_enabled:
            if (
                "limit_all_gathers" in self.args.fsdp_config
                and self.args.fsdp_config["limit_all_gathers"]
            ):
                self.accelerator.state.fsdp_plugin.limit_all_gathers = True

        return res

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Log `logs` on the various objects watching training, including stored metrics.

        Args:
            logs (`Dict[str, float]`):
                The values to log.
            start_time (`Optional[float]`):
                The start of training.
        """
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            try:
                return super().log(logs, start_time)
            except TypeError:
                return super().log(logs)  # transformers<=4.46
        return super().log(logs)  # transformers<=4.46

    def store_metrics(
        self, metrics: Dict[str, float], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def _save_checkpoint(self, model, trial, **kwargs):
        # make sure the checkpoint dir exists, since trainer is flakey
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)
        return super()._save_checkpoint(model, trial, **kwargs)


class AxolotlMambaTrainer(AxolotlTrainer):
    """
    Mamba specific trainer to handle loss calculation
    """

    tag_names = ["axolotl", "mamba"]

    def compute_loss(
        self,
        model,
        inputs,
        return_outputs=False,  # pylint: disable=unused-argument
        num_items_in_batch=None,  # pylint: disable=unused-argument
    ):
        input_ids = inputs.pop("input_ids")
        lm_logits = model(input_ids).logits

        labels = input_ids.to(lm_logits.device)
        shift_logits = lm_logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss()
        lm_loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
        )

        return lm_loss


class ReLoRATrainer(AxolotlTrainer):
    """
    Trainer subclass that uses the OneCycleLR scheduler
    """

    tag_names = ["axolotl", "relora"]

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
            anneal_steps = (
                self.args.relora_anneal_steps if self.args.relora_anneal_steps else 1
            )
            self.lr_scheduler = ReLoRAScheduler(
                optimizer,
                lr_scheduler,
                self.args.relora_steps,
                anneal_steps,
                warmup_steps,
            )
        else:
            self.lr_scheduler = lr_scheduler

        return self.lr_scheduler


class AxolotlDPOTrainer(SchedulerMixin, DPOTrainer):
    """
    Extend the base DPOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "dpo"]

    def __init__(self, *args, dataset_tags=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset_tags = dataset_tags
        self.optimizer = None

    def create_optimizer(self):
        if self.args.loraplus_lr_ratio is None:
            return super().create_optimizer()

        opt_model = self.model_wrapped if is_sagemaker_mp_enabled() else self.model
        if self.optimizer is None:  # pylint: disable=access-member-before-definition
            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args,
                opt_model,
            )

            loraplus_lr_ratio = getattr(self.args, "loraplus_lr_ratio", None)
            if loraplus_lr_ratio:
                print("Using lora+")
            loraplus_lr_embedding = getattr(self.args, "loraplus_lr_embedding", None)
            self.optimizer = create_loraplus_optimizer(  # pylint: disable=attribute-defined-outside-init
                opt_model,
                optimizer_cls,
                loraplus_lr_ratio=loraplus_lr_ratio,
                loraplus_lr_embedding=loraplus_lr_embedding,
                **optimizer_kwargs,
            )

        if is_sagemaker_mp_enabled():
            self.optimizer = smp.DistributedOptimizer(  # pylint: disable=attribute-defined-outside-init
                self.optimizer
            )

        return self.optimizer

    @wraps(DPOTrainer.push_to_hub)
    def push_to_hub(self, *args, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tags when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = _sanitize_kwargs_for_ds_tagging(
            dataset_tags=self.dataset_tags, kwargs=kwargs
        )
        kwargs = _sanitize_kwargs_for_tagging(tag_names=self.tag_names, kwargs=kwargs)

        return super().push_to_hub(*args, **kwargs)

    @staticmethod
    def tokenize_row(
        features,
        processing_class,
        max_prompt_length,
        max_completion_length,
        add_special_tokens,
    ) -> Dict:
        res = DPOTrainer.tokenize_row(
            features,
            processing_class,
            max_prompt_length,
            max_completion_length,
            add_special_tokens,
        )
        # fix when the tokenizer doesn't have a bos_token_id, e.g. Qwen
        if processing_class.bos_token is None and res["prompt_input_ids"][0] is None:
            for key in res.keys():
                res[key] = res[key][1:]

        if processing_class.bos_token and processing_class.bos_token_id is not None:
            # dpo trainer may incorrectly prepend the bos_token_id to the dpo outputs
            if res["chosen_input_ids"][0] == processing_class.bos_token_id:
                res["chosen_input_ids"] = res["chosen_input_ids"][1:]
                res["chosen_labels"] = res["chosen_labels"][1:]
                res["chosen_attention_mask"] = res["chosen_attention_mask"][1:]
            if res["rejected_input_ids"][0] == processing_class.bos_token_id:
                res["rejected_input_ids"] = res["rejected_input_ids"][1:]
                res["rejected_labels"] = res["rejected_labels"][1:]
                res["rejected_attention_mask"] = res["rejected_attention_mask"][1:]

        return res

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch=None,
    ) -> torch.Tensor:
        loss: torch.Tensor = super().training_step(model, inputs, num_items_in_batch)
        gc.collect()
        torch.cuda.empty_cache()
        return loss

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # TODO remove once trl supports the updated to the Trainer.log method
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            return super(DPOTrainer, self).log(  # pylint: disable=bad-super-call
                logs, start_time
            )
        # transformers<=4.46
        return super(DPOTrainer, self).log(logs)  # pylint: disable=bad-super-call


class AxolotlORPOTrainer(SchedulerMixin, ORPOTrainer):
    """
    Extend the base ORPOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "orpo"]

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # TODO remove once trl supports the updated to the Trainer.log method
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            return super(ORPOTrainer, self).log(  # pylint: disable=bad-super-call
                logs, start_time
            )
        # transformers<=4.46
        return super(ORPOTrainer, self).log(logs)  # pylint: disable=bad-super-call


class AxolotlKTOTrainer(SchedulerMixin, KTOTrainer):
    """
    Extend the base KTOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "kto"]

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # TODO remove once trl supports the updated to the Trainer.log method
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # train metrics should have no prefix, eval should have 'eval_'
        prefix = "eval_" if train_eval == "eval" else ""
        # accumulate average metrics from sums and lengths
        for split in ["chosen", "rejected"]:
            if f"count/{split}" in self._stored_metrics[train_eval]:
                count_sum = (
                    torch.Tensor(self._stored_metrics[train_eval][f"count/{split}"])
                    .sum()
                    .item()
                )
                for metric in ["rewards", "logps", "logits"]:
                    logs[f"{prefix}{metric}/{split}"] = (
                        torch.Tensor(
                            self._stored_metrics[train_eval][f"{metric}/{split}_sum"]
                        )
                        .sum()
                        .item()
                        / count_sum
                    )
                    # delete obsolete metric
                    del self._stored_metrics[train_eval][f"{metric}/{split}_sum"]
                del self._stored_metrics[train_eval][f"count/{split}"]
        # calculate reward margin
        if f"{prefix}rewards/chosen" in logs and f"{prefix}rewards/rejected" in logs:
            logs[f"{prefix}rewards/margins"] = (
                logs[f"{prefix}rewards/chosen"] - logs[f"{prefix}rewards/rejected"]
            )
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[f"{prefix}{key}"] = torch.Tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            return super(KTOTrainer, self).log(  # pylint: disable=bad-super-call
                logs, start_time
            )
        # transformers<=4.46
        return super(KTOTrainer, self).log(logs)  # pylint: disable=bad-super-call


class AxolotlCPOTrainer(SchedulerMixin, CPOTrainer):
    """
    Extend the base CPOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "cpo"]

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # TODO remove once trl supports the updated to the Trainer.log method
        # logs either has 'loss' or 'eval_loss'
        train_eval = "train" if "loss" in logs else "eval"
        # Add averaged stored metrics to logs
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]

        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            return super(CPOTrainer, self).log(  # pylint: disable=bad-super-call
                logs, start_time
            )
        # transformers<=4.46
        return super(CPOTrainer, self).log(logs)  # pylint: disable=bad-super-call


class AxolotlRewardTrainer(SchedulerMixin, RewardTrainer):
    """
    Extend the base RewardTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "reward"]

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        # TODO remove once trl supports the updated to the Trainer.log method
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            return super(RewardTrainer, self).log(  # pylint: disable=bad-super-call
                logs, start_time
            )
        # transformers<=4.46
        return super(RewardTrainer, self).log(logs)  # pylint: disable=bad-super-call


class TrainerBuilderBase(abc.ABC):
    """
    Base class for trainer builder
    """

    _train_dataset = None
    _eval_dataset = None
    _model_ref = None
    _peft_config = None

    def __init__(self, cfg, model, tokenizer, processor=None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

        # in case the model supports tagging, add the axolotl tag.
        # This makes sure the tag is correctly pushed even if a user calls
        # model.push_to_hub instad of  trainer.push_to_hub.
        if hasattr(model, "add_model_tags"):
            model.add_model_tags(["axolotl"])

    @property
    def model_ref(self):
        return self._model_ref

    @model_ref.setter
    def model_ref(self, model):
        self._model_ref = model

    @property
    def train_dataset(self):
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, dataset):
        self._train_dataset = dataset

    @property
    def eval_dataset(self):
        return self._eval_dataset

    @eval_dataset.setter
    def eval_dataset(self, dataset):
        self._eval_dataset = dataset

    @property
    def peft_config(self):
        return self._peft_config

    @peft_config.setter
    def peft_config(self, peft_config):
        self._peft_config = peft_config

    @abstractmethod
    def build(self, total_num_steps):
        pass

    def get_callbacks(self) -> List[TrainerCallback]:
        callbacks = []

        plugin_manager = PluginManager.get_instance()
        callbacks.extend(
            plugin_manager.add_callbacks_pre_trainer(cfg=self.cfg, model=self.model)
        )

        if self.cfg.use_wandb:
            callbacks.append(
                SaveAxolotlConfigtoWandBCallback(self.cfg.axolotl_config_path)
            )
        if self.cfg.use_mlflow and is_mlflow_available():
            from axolotl.utils.callbacks.mlflow_ import (
                SaveAxolotlConfigtoMlflowCallback,
            )

            callbacks.extend(
                [
                    SaveAxolotlConfigtoMlflowCallback(self.cfg.axolotl_config_path),
                ]
            )
        if self.cfg.use_comet and is_comet_available():
            from axolotl.utils.callbacks.comet_ import SaveAxolotlConfigtoCometCallback

            callbacks.append(
                SaveAxolotlConfigtoCometCallback(self.cfg.axolotl_config_path)
            )

        return callbacks

    def get_post_trainer_create_callbacks(self, trainer):
        """
        Callbacks added after the trainer is created, usually b/c these need access to the trainer
        """
        callbacks = []
        if self.cfg.plugins:
            plugin_manager = PluginManager.get_instance()
            callbacks.extend(
                [
                    cb
                    for cb in plugin_manager.add_callbacks_post_trainer(
                        self.cfg, trainer
                    )
                    if cb
                ]
            )
        return callbacks

    def hook_pre_create_training_args(self, training_arguments_kwargs):
        # TODO
        return training_arguments_kwargs

    def hook_post_create_training_args(self, training_arguments):
        # TODO
        return training_arguments

    def hook_pre_create_trainer(self, trainer_kwargs, trainer_cls):
        # TODO
        return trainer_kwargs, trainer_cls

    def hook_post_create_trainer(self, trainer):
        # TODO
        return trainer


class HFCausalTrainerBuilder(TrainerBuilderBase):
    """
    Build the HuggingFace training args/trainer for Causal models
    """

    def get_callbacks(self):
        callbacks = super().get_callbacks()
        callbacks.append(GPUStatsCallback(self.cfg))
        callbacks.append(EvalFirstStepCallback())

        if self.cfg.relora_steps:
            callbacks.append(ReLoRACallback(self.cfg))

        if (
            hasattr(self.model, "use_bettertransformer")
            and self.model.use_bettertransformer is True
        ):
            callbacks.append(SaveBetterTransformerModelCallback())

        if self.cfg.loss_watchdog_threshold is not None:
            callbacks.append(LossWatchDogCallback(self.cfg))

        callbacks.append(SaveModelCallback())

        return callbacks

    def get_post_trainer_create_callbacks(self, trainer):
        callbacks = []
        if self.cfg.use_wandb and self.cfg.eval_table_size > 0:
            LogPredictionCallback = log_prediction_callback_factory(
                trainer, self.tokenizer, "wandb"
            )
            callbacks.append(LogPredictionCallback(self.cfg))
        if (
            self.cfg.use_mlflow
            and is_mlflow_available()
            and self.cfg.eval_table_size > 0
        ):
            LogPredictionCallback = log_prediction_callback_factory(
                trainer, self.tokenizer, "mlflow"
            )
            callbacks.append(LogPredictionCallback(self.cfg))
        if self.cfg.use_comet and is_comet_available() and self.cfg.eval_table_size > 0:
            LogPredictionCallback = log_prediction_callback_factory(
                trainer, self.tokenizer, "comet_ml"
            )
            callbacks.append(LogPredictionCallback(self.cfg))

        if self.cfg.do_bench_eval:
            callbacks.append(bench_eval_callback_factory(trainer, self.tokenizer))
        if self.cfg.do_causal_lm_eval:
            CausalLMBenchEvalCallback = causal_lm_bench_eval_callback_factory(
                trainer, self.tokenizer
            )
            callbacks.append(CausalLMBenchEvalCallback(self.cfg))

        if self.cfg.early_stopping_patience:
            early_stop_cb = EarlyStoppingCallback(
                self.cfg.early_stopping_patience,
            )
            callbacks.append(early_stop_cb)

        if self.cfg.lisa_step_interval and self.cfg.lisa_n_layers:
            callbacks.append(lisa_callback_factory(trainer))

        callbacks.extend(super().get_post_trainer_create_callbacks(trainer=trainer))
        return callbacks

    def _get_trainer_cls(self):
        if self.cfg.relora_steps:
            return ReLoRATrainer
        if self.cfg.model_config_type == "mamba":
            return AxolotlMambaTrainer
        if self.cfg.reward_model:
            return AxolotlRewardTrainer
        return AxolotlTrainer

    def build(self, total_num_steps):
        warmup_steps = None
        if self.cfg.warmup_steps is not None:
            warmup_steps = self.cfg.warmup_steps
        elif self.cfg.warmup_ratio is not None:
            warmup_steps = max(int(self.cfg.warmup_ratio * total_num_steps), 0)
        else:
            warmup_steps = min(int(0.03 * total_num_steps), 100)
        if warmup_steps == 1:
            warmup_steps = 2

        logging_steps = (
            self.cfg.logging_steps
            if self.cfg.logging_steps is not None
            else max(min(int(0.005 * total_num_steps), 10), 1)
        )

        training_arguments_kwargs = {}
        if self.cfg.bf16 == "full":
            training_arguments_kwargs["bf16_full_eval"] = True
        else:
            training_arguments_kwargs["bf16"] = self.cfg.bf16
        training_arguments_kwargs["fp16"] = (
            self.cfg.fp16 and not self.cfg.bf16
        ) or False
        training_arguments_kwargs["tf32"] = self.cfg.tf32
        training_arguments_kwargs["warmup_steps"] = warmup_steps
        training_arguments_kwargs["logging_steps"] = logging_steps

        if self.cfg.seed:
            training_arguments_kwargs["seed"] = self.cfg.seed

        if self.cfg.gradient_checkpointing:
            training_arguments_kwargs[
                "gradient_checkpointing"
            ] = self.cfg.gradient_checkpointing
            if self.cfg.gradient_checkpointing_kwargs is not None:
                training_arguments_kwargs[
                    "gradient_checkpointing_kwargs"
                ] = self.cfg.gradient_checkpointing_kwargs
        if self.cfg.fsdp:
            training_arguments_kwargs["fsdp"] = self.cfg.fsdp
            if self.cfg.fsdp_config:
                training_arguments_kwargs["fsdp_config"] = {
                    k.lstrip("fsdp_"): v for k, v in dict(self.cfg.fsdp_config).items()
                }

        if self.cfg.adapter == "qlora":
            training_arguments_kwargs["qlora"] = True

        # deepspeed
        if self.cfg.deepspeed:
            training_arguments_kwargs["deepspeed"] = self.cfg.deepspeed

        if self.cfg.lr_quadratic_warmup is not None:
            training_arguments_kwargs[
                "lr_quadratic_warmup"
            ] = self.cfg.lr_quadratic_warmup

        if self.cfg.adam_beta1:
            training_arguments_kwargs["adam_beta1"] = self.cfg.adam_beta1
        if self.cfg.adam_beta2:
            training_arguments_kwargs["adam_beta2"] = self.cfg.adam_beta2
        if self.cfg.adam_epsilon:
            training_arguments_kwargs["adam_epsilon"] = self.cfg.adam_epsilon
        if self.cfg.max_grad_norm:
            training_arguments_kwargs["max_grad_norm"] = self.cfg.max_grad_norm

        if self.cfg.hub_model_id:
            training_arguments_kwargs["hub_model_id"] = self.cfg.hub_model_id
            training_arguments_kwargs["push_to_hub"] = True
            training_arguments_kwargs["hub_private_repo"] = True
            training_arguments_kwargs["hub_always_push"] = True

            if self.cfg.hub_strategy:
                training_arguments_kwargs["hub_strategy"] = self.cfg.hub_strategy

        if self.cfg.save_safetensors is not None:
            training_arguments_kwargs["save_safetensors"] = self.cfg.save_safetensors

        if self.cfg.dataloader_pin_memory is not None:
            training_arguments_kwargs[
                "dataloader_pin_memory"
            ] = self.cfg.dataloader_pin_memory
        if self.cfg.dataloader_num_workers is not None:
            training_arguments_kwargs[
                "dataloader_num_workers"
            ] = self.cfg.dataloader_num_workers
        if self.cfg.dataloader_prefetch_factor is not None:
            training_arguments_kwargs[
                "dataloader_prefetch_factor"
            ] = self.cfg.dataloader_prefetch_factor
        if self.cfg.dataloader_drop_last is not None:
            training_arguments_kwargs[
                "dataloader_drop_last"
            ] = self.cfg.dataloader_drop_last
        elif self.cfg.sample_packing and self.cfg.eval_sample_packing is False:
            training_arguments_kwargs["dataloader_drop_last"] = True

        if self.cfg.remove_unused_columns is not None:
            training_arguments_kwargs[
                "remove_unused_columns"
            ] = self.cfg.remove_unused_columns

        if not self.cfg.test_datasets and self.cfg.val_set_size == 0:
            # no eval set, so don't eval
            training_arguments_kwargs["eval_strategy"] = "no"
        elif self.cfg.eval_steps:
            training_arguments_kwargs["eval_strategy"] = "steps"
            training_arguments_kwargs["eval_steps"] = self.cfg.eval_steps
        elif self.cfg.eval_strategy:
            training_arguments_kwargs["eval_strategy"] = self.cfg.eval_strategy
        else:
            # we have an eval set, but no steps defined, default to use epoch
            training_arguments_kwargs["eval_strategy"] = "epoch"

        if self.cfg.save_steps:
            training_arguments_kwargs["save_strategy"] = "steps"
            training_arguments_kwargs["save_steps"] = self.cfg.save_steps
        elif self.cfg.save_strategy:
            training_arguments_kwargs["save_strategy"] = self.cfg.save_strategy
        else:
            # default to saving each epoch if not defined
            training_arguments_kwargs["save_strategy"] = "epoch"

        training_arguments_kwargs["save_only_model"] = self.cfg.save_only_model

        if self.cfg.do_bench_eval:
            training_arguments_kwargs["do_bench_eval"] = self.cfg.do_bench_eval
            if self.cfg.bench_dataset:
                training_arguments_kwargs["bench_dataset"] = self.cfg.bench_dataset
        if self.cfg.do_causal_lm_eval:
            training_arguments_kwargs["do_causal_lm_eval"] = self.cfg.do_causal_lm_eval
        if self.cfg.metric_for_best_model:
            training_arguments_kwargs[
                "metric_for_best_model"
            ] = self.cfg.metric_for_best_model
        if self.cfg.greater_is_better:
            training_arguments_kwargs["greater_is_better"] = self.cfg.greater_is_better

        if self.cfg.torch_compile:
            if torch.__version__ < "2.1.0":  # pylint: disable=protected-access
                LOG.warning("torch>=2.1.0 required for torch_compile to work properly")
            elif torch._dynamo:  # pylint: disable=protected-access
                torch._dynamo.config.suppress_errors = (  # pylint: disable=protected-access
                    True
                )
                training_arguments_kwargs["torch_compile"] = self.cfg.torch_compile
                if self.cfg.torch_compile_backend:
                    training_arguments_kwargs[
                        "torch_compile_backend"
                    ] = self.cfg.torch_compile_backend
                if self.cfg.torch_compile_mode:
                    training_arguments_kwargs[
                        "torch_compile_mode"
                    ] = self.cfg.torch_compile_mode

        # DDP Config
        if self.cfg.ddp_timeout:
            training_arguments_kwargs["ddp_timeout"] = self.cfg.ddp_timeout
        # see https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        if self.cfg.ddp_bucket_cap_mb:
            training_arguments_kwargs["ddp_bucket_cap_mb"] = self.cfg.ddp_bucket_cap_mb
        if self.cfg.ddp_broadcast_buffers is not None:
            training_arguments_kwargs[
                "ddp_broadcast_buffers"
            ] = self.cfg.ddp_broadcast_buffers

        # these are all the "standard" kwargs that are def used
        training_arguments_kwargs["max_steps"] = (
            total_num_steps if self.cfg.max_steps else -1
        )
        training_arguments_kwargs["max_seq_length"] = self.cfg.sequence_len
        training_arguments_kwargs[
            "per_device_train_batch_size"
        ] = self.cfg.micro_batch_size
        if self.cfg.eval_batch_size:
            training_arguments_kwargs[
                "per_device_eval_batch_size"
            ] = self.cfg.eval_batch_size
        if self.cfg.auto_find_batch_size is not None:
            training_arguments_kwargs[
                "auto_find_batch_size"
            ] = self.cfg.auto_find_batch_size
        training_arguments_kwargs[
            "gradient_accumulation_steps"
        ] = self.cfg.gradient_accumulation_steps
        training_arguments_kwargs[
            "eval_accumulation_steps"
        ] = self.cfg.gradient_accumulation_steps
        training_arguments_kwargs["num_train_epochs"] = self.cfg.num_epochs
        training_arguments_kwargs["learning_rate"] = self.cfg.learning_rate
        training_arguments_kwargs["output_dir"] = self.cfg.output_dir
        training_arguments_kwargs["save_total_limit"] = (
            self.cfg.save_total_limit if self.cfg.save_total_limit else 4
        )
        training_arguments_kwargs["load_best_model_at_end"] = (
            (
                self.cfg.load_best_model_at_end is not False
                or self.cfg.early_stopping_patience
            )
            and (
                (not self.cfg.test_datasets and self.cfg.val_set_size > 0)
                or (self.cfg.test_datasets and self.cfg.val_set_size == 0)
            )
            and self.cfg.save_steps
            and self.cfg.eval_steps
            and self.cfg.save_steps % self.cfg.eval_steps == 0
        ) or False
        training_arguments_kwargs["ddp_find_unused_parameters"] = (
            False if self.cfg.ddp else None
        )
        training_arguments_kwargs["group_by_length"] = self.cfg.group_by_length
        training_arguments_kwargs["curriculum_sampling"] = self.cfg.curriculum_sampling
        report_to = []
        if self.cfg.use_wandb:
            report_to.append("wandb")
            if self.cfg.wandb_name:
                training_arguments_kwargs["run_name"] = self.cfg.wandb_name
        if self.cfg.use_mlflow:
            report_to.append("mlflow")
        if self.cfg.use_tensorboard:
            report_to.append("tensorboard")
        if self.cfg.use_comet:
            report_to.append("comet_ml")

        training_arguments_kwargs["report_to"] = report_to
        if self.cfg.use_wandb:
            training_arguments_kwargs["run_name"] = self.cfg.wandb_name
        elif self.cfg.use_mlflow:
            training_arguments_kwargs["run_name"] = self.cfg.mlflow_run_name
        else:
            training_arguments_kwargs["run_name"] = None
        training_arguments_kwargs["optim"] = (
            self.cfg.optimizer if self.cfg.optimizer else "adamw_hf"
        )
        if self.cfg.optim_args:
            if isinstance(self.cfg.optim_args, dict):
                optim_args = ",".join(
                    [f"{key}={value}" for key, value in self.cfg.optim_args.items()]
                )
            else:
                optim_args = self.cfg.optim_args
            training_arguments_kwargs["optim_args"] = optim_args
        if self.cfg.optim_target_modules:
            training_arguments_kwargs[
                "optim_target_modules"
            ] = self.cfg.optim_target_modules
        training_arguments_kwargs["loraplus_lr_ratio"] = self.cfg.loraplus_lr_ratio
        training_arguments_kwargs[
            "loraplus_lr_embedding"
        ] = self.cfg.loraplus_lr_embedding
        training_arguments_kwargs["embedding_lr"] = self.cfg.embedding_lr
        training_arguments_kwargs["embedding_lr_scale"] = self.cfg.embedding_lr_scale

        if self.cfg.lr_scheduler in ["one_cycle", "log_sweep"]:
            training_arguments_kwargs["lr_scheduler_type"] = "cosine"
            training_arguments_kwargs[
                "alternate_lr_scheduler_type"
            ] = self.cfg.lr_scheduler
        else:
            training_arguments_kwargs["lr_scheduler_type"] = (
                self.cfg.lr_scheduler if self.cfg.lr_scheduler else "cosine"
            )
        training_arguments_kwargs["lr_scheduler_kwargs"] = (
            self.cfg.lr_scheduler_kwargs if self.cfg.lr_scheduler_kwargs else {}
        )
        training_arguments_kwargs["cosine_min_lr_ratio"] = self.cfg.cosine_min_lr_ratio
        training_arguments_kwargs[
            "cosine_constant_lr_ratio"
        ] = self.cfg.cosine_constant_lr_ratio
        training_arguments_kwargs["weight_decay"] = (
            self.cfg.weight_decay if self.cfg.weight_decay is not None else 0.0
        )

        training_arguments_kwargs["sample_packing"] = bool(self.cfg.sample_packing)
        training_arguments_kwargs["multipack_real_batches"] = (
            not self.cfg.flash_attention or self.cfg.multipack_real_batches
        )
        training_arguments_kwargs["eval_sample_packing"] = bool(
            self.cfg.eval_sample_packing
        )
        if self.cfg.sample_packing_bin_size is not None:
            training_arguments_kwargs[
                "sample_packing_bin_size"
            ] = self.cfg.sample_packing_bin_size
        if self.cfg.sample_packing_group_size is not None:
            training_arguments_kwargs[
                "sample_packing_group_size"
            ] = self.cfg.sample_packing_group_size
        if self.cfg.sample_packing_eff_est:
            training_arguments_kwargs[
                "sample_packing_efficiency"
            ] = self.cfg.sample_packing_eff_est

        if self.cfg.relora_steps:
            training_arguments_kwargs["relora_steps"] = self.cfg.relora_steps
            training_arguments_kwargs[
                "relora_warmup_steps"
            ] = self.cfg.relora_warmup_steps
            if self.cfg.relora_anneal_steps:
                training_arguments_kwargs[
                    "relora_anneal_steps"
                ] = self.cfg.relora_anneal_steps
            if self.cfg.relora_prune_ratio:
                training_arguments_kwargs[
                    "relora_prune_ratio"
                ] = self.cfg.relora_prune_ratio

        if self.cfg.lisa_step_interval and self.cfg.lisa_n_layers:
            training_arguments_kwargs["lisa_n_layers"] = self.cfg.lisa_n_layers
            training_arguments_kwargs[
                "lisa_step_interval"
            ] = self.cfg.lisa_step_interval
            training_arguments_kwargs[
                "lisa_layers_attribute"
            ] = self.cfg.lisa_layers_attribute

        training_arguments_kwargs = self.hook_pre_create_training_args(
            training_arguments_kwargs
        )
        training_arguments_kwargs["model_type"] = self.cfg.model_config_type
        training_arguments_kwargs["pretraining"] = bool(self.cfg.pretraining_dataset)
        if self.cfg.chat_template:
            training_arguments_kwargs["chat_template"] = get_chat_template(
                self.cfg.chat_template,
                tokenizer=self.tokenizer,
            )

        if self.cfg.rl == "orpo":
            training_arguments_kwargs["orpo_alpha"] = self.cfg.orpo_alpha

        if self.cfg.neftune_noise_alpha is not None:
            training_arguments_kwargs[
                "neftune_noise_alpha"
            ] = self.cfg.neftune_noise_alpha

        trainer_kwargs = {}

        if self.cfg.reward_model:
            trainer_kwargs["max_length"] = self.cfg.sequence_len

        # pylint: disable=duplicate-code
        if self.cfg.optimizer in [
            "optimi_adamw",
            "ao_adamw_4bit",
            "ao_adamw_8bit",
            "ao_adamw_fp8",
            "adopt_adamw",
        ]:
            # Set default so transformers doesn't throw
            training_arguments_kwargs["optim"] = "adamw_hf"
            training_arguments_kwargs["alternate_optimizer"] = self.cfg.optimizer

        if self.cfg.optimizer == "lion_pytorch":
            from lion_pytorch import Lion

            lion_kwargs = {"lr": training_arguments_kwargs["learning_rate"]}
            if "weight_decay" in training_arguments_kwargs:
                lion_kwargs["weight_decay"] = training_arguments_kwargs["weight_decay"]

            if (
                "adam_beta1" in training_arguments_kwargs
                and "adam_beta2" in training_arguments_kwargs
            ):
                lion_kwargs["betas"] = (
                    training_arguments_kwargs["adam_beta1"],
                    training_arguments_kwargs["adam_beta2"],
                )

            trainer_kwargs["optimizers"] = (
                Lion(params=self.model.parameters(), **lion_kwargs),
                None,
            )
            # Set default so transformers doesn't throw
            training_arguments_kwargs["optim"] = "adamw_hf"

        if self.cfg.optimizer == "adamw_anyprecision":
            if Path(self.cfg.torchdistx_path).exists():
                sys.path.append(self.cfg.torchdistx_path)
                importlib.import_module("torchdistx")

        if self.cfg.accelerator_config:
            training_arguments_kwargs[
                "accelerator_config"
            ] = self.cfg.accelerator_config

        training_args_cls = (
            AxolotlTrainingArguments
            if not self.cfg.reward_model
            else AxolotlRewardConfig
        )
        training_args = training_args_cls(  # pylint: disable=unexpected-keyword-arg
            **training_arguments_kwargs,
        )
        training_args = self.hook_post_create_training_args(training_args)

        # unset run_name so wandb sets up experiment names
        if self.cfg.use_wandb and training_args.run_name == training_args.output_dir:
            training_args.run_name = (  # pylint: disable=attribute-defined-outside-init
                None
            )

        data_collator_kwargs = {
            "padding": True,  # True/"longest" is the default
        }
        if self.cfg.pad_to_sequence_len:
            data_collator_kwargs["pad_to_multiple_of"] = 64 * math.ceil(
                self.cfg.sequence_len / 64
            )
        else:
            # A100 is best at 64, while others at 8. Let's use the larger so we don't have to check
            # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
            data_collator_kwargs["pad_to_multiple_of"] = 64

        if self.cfg.reward_model:
            data_collator_kwargs["max_length"] = self.cfg.sequence_len

        trainer_cls = self._get_trainer_cls()
        trainer_kwargs, trainer_cls = self.hook_pre_create_trainer(
            trainer_kwargs, trainer_cls
        )
        if eval_data_collator := self.build_collator(
            training_args, is_eval=True, **data_collator_kwargs
        ):
            if not self.cfg.reward_model:
                trainer_kwargs["eval_data_collator"] = eval_data_collator
        if not self.cfg.reward_model:
            trainer_kwargs["bench_data_collator"] = transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                return_tensors="pt",
                **data_collator_kwargs,
            )
        sig = inspect.signature(trainer_cls)
        if "processing_class" in sig.parameters.keys():
            trainer_kwargs["processing_class"] = self.tokenizer
        else:
            trainer_kwargs["tokenizer"] = self.tokenizer

        if (trainer_cls is not AxolotlRewardTrainer) and self.cfg.datasets is not None:
            trainer_kwargs["dataset_tags"] = [
                d["path"] for d in self.cfg.datasets if not Path(d["path"]).is_dir()
            ]
        trainer = trainer_cls(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=training_args,
            data_collator=self.build_collator(training_args, **data_collator_kwargs),
            callbacks=self.get_callbacks(),
            **trainer_kwargs,
        )
        trainer = self.hook_post_create_trainer(trainer)
        for callback in self.get_post_trainer_create_callbacks(trainer):
            trainer.add_callback(callback)

        if self.cfg.deepspeed and self.cfg.sample_packing:
            trainer.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = self.cfg.micro_batch_size

        return trainer

    def build_collator(
        self, training_args: AxolotlTrainingArguments, is_eval=False, **kwargs
    ):
        if training_args.pretraining:
            return None

        if self.cfg.model_config_type == "mamba":
            return MambaDataCollator(tokenizer=self.tokenizer)

        use_batch_sampler_collator = False
        if is_eval is False and training_args.sample_packing:
            use_batch_sampler_collator = True
        if is_eval and training_args.eval_sample_packing:
            use_batch_sampler_collator = True

        collator: Type[
            Union[
                V2BatchSamplerDataCollatorForSeq2Seq,
                BatchSamplerDataCollatorForSeq2Seq,
                DataCollatorForSeq2Seq,
                RewardDataCollatorWithPadding,
            ]
        ]
        if self.cfg.reward_model:
            collator = RewardDataCollatorWithPadding
            if "max_length" in kwargs:
                kwargs.pop("max_length")
        elif use_batch_sampler_collator:
            if self.cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES:
                collator = V2BatchSamplerDataCollatorForSeq2Seq
            elif (
                self.cfg.model_config_type in ["llama"]
                and self.cfg.flash_attention is not True
            ):
                collator = V2BatchSamplerDataCollatorForSeq2Seq
            else:
                collator = BatchSamplerDataCollatorForSeq2Seq
        else:
            if self.cfg.processor_type and self.processor:
                collator = MultiModalChatDataCollator
                kwargs["processor"] = self.processor
                kwargs["chat_template"] = training_args.chat_template
            else:
                collator = DataCollatorForSeq2Seq

        return collator(
            self.tokenizer,
            return_tensors="pt",
            **kwargs,
        )


class HFRLTrainerBuilder(TrainerBuilderBase):
    """
    Trainer factory class for DPO Trainer
    """

    def get_callbacks(self):
        callbacks = super().get_callbacks()
        callbacks.append(SaveModelCallback())

        return callbacks

    def get_post_trainer_create_callbacks(self, trainer):
        callbacks = super().get_post_trainer_create_callbacks(trainer=trainer)
        return callbacks

    def build_training_arguments(self, total_num_steps):
        training_args_kwargs = {}
        for arg in [
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "dataloader_num_workers",
            "dataloader_pin_memory",
        ]:
            if hasattr(self.cfg, arg) and getattr(self.cfg, arg) is not None:
                training_args_kwargs[arg] = getattr(self.cfg, arg)

        if self.cfg.hub_model_id:
            training_args_kwargs["hub_model_id"] = self.cfg.hub_model_id
            training_args_kwargs["push_to_hub"] = True
            training_args_kwargs["hub_private_repo"] = True
            training_args_kwargs["hub_always_push"] = True

            if self.cfg.hub_strategy:
                training_args_kwargs["hub_strategy"] = self.cfg.hub_strategy

        if self.cfg.save_safetensors is not None:
            training_args_kwargs["save_safetensors"] = self.cfg.save_safetensors

        if self.eval_dataset:
            training_args_kwargs["eval_strategy"] = "steps"
            training_args_kwargs["eval_steps"] = self.cfg.eval_steps
        else:
            training_args_kwargs["eval_strategy"] = "no"

        if self.cfg.bf16 or self.cfg.bfloat16:
            training_args_kwargs["bf16"] = True

        training_args_kwargs["loraplus_lr_ratio"] = self.cfg.loraplus_lr_ratio
        training_args_kwargs["loraplus_lr_embedding"] = self.cfg.loraplus_lr_embedding
        training_args_kwargs["lr_scheduler_type"] = (
            self.cfg.lr_scheduler if self.cfg.lr_scheduler else "cosine"
        )
        training_args_kwargs["lr_scheduler_kwargs"] = (
            self.cfg.lr_scheduler_kwargs if self.cfg.lr_scheduler_kwargs else {}
        )
        if self.cfg.remove_unused_columns is not None:
            training_args_kwargs[
                "remove_unused_columns"
            ] = self.cfg.remove_unused_columns
        else:
            training_args_kwargs["remove_unused_columns"] = False

        if self.cfg.dataloader_pin_memory is not None:
            training_args_kwargs[
                "dataloader_pin_memory"
            ] = self.cfg.dataloader_pin_memory
        if self.cfg.dataloader_num_workers is not None:
            training_args_kwargs[
                "dataloader_num_workers"
            ] = self.cfg.dataloader_num_workers
        if self.cfg.dataloader_prefetch_factor is not None:
            training_args_kwargs[
                "dataloader_prefetch_factor"
            ] = self.cfg.dataloader_prefetch_factor
        if self.cfg.gradient_checkpointing:
            training_args_kwargs[
                "gradient_checkpointing"
            ] = self.cfg.gradient_checkpointing
            if self.cfg.gradient_checkpointing_kwargs is not None:
                training_args_kwargs[
                    "gradient_checkpointing_kwargs"
                ] = self.cfg.gradient_checkpointing_kwargs
            else:
                training_args_kwargs["gradient_checkpointing_kwargs"] = {
                    "use_reentrant": False
                }

        # set save_strategy and save_steps
        if self.cfg.save_steps:
            training_args_kwargs["save_strategy"] = "steps"
            training_args_kwargs["save_steps"] = self.cfg.save_steps
        elif self.cfg.save_strategy:
            training_args_kwargs["save_strategy"] = self.cfg.save_strategy
        else:
            # default to saving each epoch if not defined
            training_args_kwargs["save_strategy"] = "epoch"

        training_args_kwargs["dataset_num_proc"] = self.cfg.dataset_processes

        if self.cfg.rl_beta:
            training_args_kwargs["beta"] = self.cfg.rl_beta
        if self.cfg.orpo_alpha:
            # trl does some odd mapping of alpha to beta to reuse the beta parameter ???
            training_args_kwargs["beta"] = self.cfg.orpo_alpha

        if self.cfg.rpo_alpha is not None:
            training_args_kwargs["rpo_alpha"] = self.cfg.rpo_alpha

        training_args_cls = None
        if self.cfg.rl == "simpo":
            training_args_cls = AxolotlCPOConfig
            training_args_kwargs["loss_type"] = "simpo"
            training_args_kwargs["max_length"] = self.cfg.sequence_len
            training_args_kwargs["simpo_gamma"] = self.cfg.simpo_gamma
            if self.cfg.cpo_alpha is not None:
                training_args_kwargs["cpo_alpha"] = self.cfg.cpo_alpha

        elif self.cfg.rl == "orpo":
            training_args_cls = AxolotlORPOConfig
            training_args_kwargs["max_length"] = self.cfg.sequence_len
            if self.cfg.max_prompt_len:
                training_args_kwargs["max_prompt_length"] = self.cfg.max_prompt_len

        elif self.cfg.rl == "kto":
            training_args_cls = AxolotlKTOConfig

            training_args_kwargs["desirable_weight"] = (
                self.cfg.kto_desirable_weight or 1.0
            )
            training_args_kwargs["undesirable_weight"] = (
                self.cfg.kto_undesirable_weight or 1.0
            )

            training_args_kwargs["dataset_num_proc"] = self.cfg.dataset_processes
            training_args_kwargs["max_length"] = self.cfg.sequence_len
            if self.cfg.max_prompt_len:
                training_args_kwargs["max_prompt_length"] = self.cfg.max_prompt_len

        else:
            training_args_cls = AxolotlDPOConfig
            if self.cfg.rl == "ipo":
                training_args_kwargs["loss_type"] = "ipo"
            training_args_kwargs["max_length"] = self.cfg.sequence_len
            training_args_kwargs["max_completion_length"] = None
            training_args_kwargs["max_prompt_length"] = self.cfg.sequence_len
            training_args_kwargs["generate_during_eval"] = self.cfg.use_wandb
            if self.cfg.dpo_use_weighting is not None:
                training_args_kwargs["use_weighting"] = self.cfg.dpo_use_weighting

        training_args = training_args_cls(  # pylint: disable=unexpected-keyword-arg
            output_dir=self.cfg.output_dir,
            per_device_train_batch_size=self.cfg.micro_batch_size,
            max_steps=self.cfg.max_steps or total_num_steps,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            warmup_steps=self.cfg.warmup_steps,
            logging_first_step=True,
            logging_steps=1,
            optim=self.cfg.optimizer,
            save_total_limit=self.cfg.save_total_limit or 5,
            **training_args_kwargs,
        )

        return training_args

    def build(self, total_num_steps):
        training_args = self.build_training_arguments(total_num_steps)
        dpo_trainer_kwargs = {}
        if self.cfg.rl == "ipo":
            if self.cfg.dpo_label_smoothing:
                dpo_trainer_kwargs["label_smoothing"] = self.cfg.dpo_label_smoothing
        if self.eval_dataset:
            dpo_trainer_kwargs["eval_dataset"] = self.eval_dataset
        if self.cfg.adapter and self.peft_config:
            dpo_trainer_kwargs["peft_config"] = self.peft_config
        if self.cfg.precompute_ref_log_probs is not None:
            dpo_trainer_kwargs[
                "precompute_ref_log_probs"
            ] = self.cfg.precompute_ref_log_probs
        if self.cfg.rl in ["dpo", "ipo"]:
            trainer_cls = AxolotlDPOTrainer
            trainer_cls_args = [self.model, self.model_ref]
        elif self.cfg.rl == "orpo":
            trainer_cls = AxolotlORPOTrainer
            trainer_cls_args = [self.model]
        elif self.cfg.rl in ["kto"]:
            trainer_cls = AxolotlKTOTrainer
            trainer_cls_args = [self.model]
        elif self.cfg.rl in ["simpo"]:
            trainer_cls = AxolotlCPOTrainer
            trainer_cls_args = [self.model]
        else:
            raise ValueError(f"Unsupported RL: {self.cfg.rl}")

        sig = inspect.signature(trainer_cls)
        if "processing_class" in sig.parameters.keys():
            dpo_trainer_kwargs["processing_class"] = self.tokenizer
        else:
            dpo_trainer_kwargs["tokenizer"] = self.tokenizer

        if self.cfg.datasets is not None and (trainer_cls is AxolotlDPOTrainer):
            dpo_trainer_kwargs["dataset_tags"] = [
                d["path"] for d in self.cfg.datasets if not Path(d["path"]).is_dir()
            ]
        dpo_trainer = trainer_cls(
            *trainer_cls_args,
            args=training_args,
            train_dataset=self.train_dataset,
            callbacks=self.get_callbacks(),
            **dpo_trainer_kwargs,
        )
        if self.cfg.fsdp:
            ensure_dtype(dpo_trainer.model, dtype=self.cfg.torch_dtype)
            if self.cfg.rl in ["dpo", "ipo"] and dpo_trainer.ref_model:
                ensure_dtype(dpo_trainer.ref_model, dtype=self.cfg.torch_dtype)

        dpo_trainer = self.hook_post_create_trainer(dpo_trainer)
        for callback in self.get_post_trainer_create_callbacks(dpo_trainer):
            dpo_trainer.add_callback(callback)

        return dpo_trainer


class HFPPOTrainerBuilder(TrainerBuilderBase):
    """
    HF Factory class for PPO Trainer
    """

    def get_callbacks(self):
        callbacks = super().get_callbacks()
        return callbacks

    def get_post_trainer_create_callbacks(self, trainer):
        callbacks = super().get_post_trainer_create_callbacks(trainer=trainer)
        return callbacks

    def build(self, total_num_steps):
        # build PPOConfig
        pass
