# pylint: disable=too-many-lines
"""
Builder for the training args and trainer
"""

import abc
import importlib
import importlib.util
import logging
import math
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import List, Optional, Type, Union

import torch
import transformers
from datasets import Dataset
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.trainer_utils import seed_worker
from trl import DPOTrainer

from axolotl.monkeypatch.multipack import SUPPORTED_MULTIPACK_MODEL_TYPES
from axolotl.monkeypatch.relora import ReLoRACallback, ReLoRAScheduler
from axolotl.utils.callbacks import (
    EvalFirstStepCallback,
    GPUStatsCallback,
    LossWatchDogCallback,
    SaveAxolotlConfigtoWandBCallback,
    SaveBetterTransformerModelCallback,
    bench_eval_callback_factory,
    causal_lm_bench_eval_callback_factory,
    log_prediction_callback_factory,
)
from axolotl.utils.collators import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    MambaDataCollator,
    V2BatchSamplerDataCollatorForSeq2Seq,
)
from axolotl.utils.samplers import MultipackBatchSampler, get_dataset_lengths
from axolotl.utils.schedulers import (
    get_cosine_schedule_with_min_lr,
    get_cosine_schedule_with_quadratic_warmup,
    get_cosine_schedule_with_warmup_decay_constant,
)

try:
    import torch._dynamo  # pylint: disable=ungrouped-imports
except ImportError:
    pass

LOG = logging.getLogger("axolotl.core.trainer_builder")


def is_mlflow_available():
    return importlib.util.find_spec("mlflow") is not None


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


@dataclass
class AxolotlTrainingArguments(TrainingArguments):
    """
    Extend the base TrainingArguments for axolotl helpers
    """

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


class AxolotlTrainer(Trainer):
    """
    Extend the base Trainer for axolotl helpers
    """

    args = None  # type: AxolotlTrainingArguments
    tag_names = ["axolotl"]

    def __init__(
        self,
        *_args,
        num_epochs=1,
        bench_data_collator=None,
        eval_data_collator=None,
        **kwargs
    ):
        self.num_epochs = num_epochs
        self.bench_data_collator = bench_data_collator
        self.eval_data_collator = eval_data_collator
        super().__init__(*_args, **kwargs)
        self.train_data_collator = self.data_collator

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
            if use_cosine_quadratic:
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
                return super().create_scheduler(num_training_steps, optimizer)
        else:
            if use_cosine_quadratic:
                LOG.warning("axolotl's cosine scheduler with quadratic warmup not used (e.g., because of deepspeed).")

            if use_cosine_min_lr:
                LOG.warning("axolotl's cosine scheduler with min lr not used (e.g., because of deepspeed).")

        return self.lr_scheduler

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.args.sample_packing and not self.args.pretraining:
            if self.args.multipack_real_batches:
                batch_size = self.args.per_device_train_batch_size
                batch_max_len = self.args.max_seq_length
            else:
                batch_size = 1
                batch_max_len = (
                    self.args.per_device_train_batch_size * self.args.max_seq_length
                )
            return MultipackBatchSampler(
                RandomSampler(self.train_dataset),
                batch_size=batch_size,
                drop_last=True,
                batch_max_len=batch_max_len,
                lengths=get_dataset_lengths(self.train_dataset),
                packing_efficiency_estimate=self.args.sample_packing_efficiency,
            )
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
                batch_size=batch_size,
                drop_last=True,
                batch_max_len=batch_max_len,
                lengths=get_dataset_lengths(eval_dataset),
                packing_efficiency_estimate=self.args.sample_packing_efficiency,
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

    def compute_loss(self, model, inputs, return_outputs=False):
        # use one's weighted cross entropy loss calc
        # if self.args.sample_packing:
        #     labels = inputs.pop("labels")
        #     outputs = model(**inputs)
        #     loss = trainer_weighted_loss(outputs, labels, shift_labels=True)
        #     return (loss, outputs) if return_outputs else loss
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    @wraps(Trainer.push_to_hub)
    def push_to_hub(self, *args, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tags when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = _sanitize_kwargs_for_tagging(tag_names=self.tag_names, kwargs=kwargs)

        return super().push_to_hub(*args, **kwargs)


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


class OneCycleLRSchedulerTrainer(AxolotlTrainer):
    """
    Trainer subclass that uses the OneCycleLR scheduler
    """

    tag_names = ["axolotl", "onecycle"]

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


class AxolotlDPOTrainer(DPOTrainer):
    """
    Extend the base DPOTrainer for axolotl helpers
    """

    tag_names = ["axolotl", "dpo"]

    @wraps(DPOTrainer.push_to_hub)
    def push_to_hub(self, *args, **kwargs) -> str:
        """
        Overwrite the `push_to_hub` method in order to force-add the tags when pushing the
        model on the Hub. Please refer to `~transformers.Trainer.push_to_hub` for more details.
        """
        kwargs = _sanitize_kwargs_for_tagging(tag_names=self.tag_names, kwargs=kwargs)

        return super().push_to_hub(*args, **kwargs)


class TrainerBuilderBase(abc.ABC):
    """
    Base class for trainer builder
    """

    _train_dataset = None
    _eval_dataset = None
    _model_ref = None
    _peft_config = None

    def __init__(self, cfg, model, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

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
        if self.cfg.use_wandb:
            callbacks.append(
                SaveAxolotlConfigtoWandBCallback(self.cfg.axolotl_config_path)
            )

        return callbacks

    @abstractmethod
    def get_post_trainer_create_callbacks(self, trainer):
        """
        Callbacks added after the trainer is created, usually b/c these need access to the trainer
        """

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

        if self.cfg.use_wandb:
            callbacks.append(
                SaveAxolotlConfigtoWandBCallback(self.cfg.axolotl_config_path)
            )
        if self.cfg.use_mlflow and is_mlflow_available():
            from axolotl.utils.callbacks.mlflow_ import (
                SaveAxolotlConfigtoMlflowCallback,
            )

            callbacks.append(
                SaveAxolotlConfigtoMlflowCallback(self.cfg.axolotl_config_path)
            )

        if self.cfg.loss_watchdog_threshold is not None:
            callbacks.append(LossWatchDogCallback(self.cfg))

        return callbacks

    def get_post_trainer_create_callbacks(self, trainer):
        callbacks = []
        if self.cfg.use_wandb and self.cfg.eval_table_size > 0:
            LogPredictionCallback = log_prediction_callback_factory(
                trainer, self.tokenizer
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

        return callbacks

    def _get_trainer_cls(self):
        if self.cfg.lr_scheduler == "one_cycle" and (
            self.cfg.fsdp or self.cfg.adapter == "qlora"
        ):
            return OneCycleLRSchedulerTrainer
        if self.cfg.relora_steps:
            return ReLoRATrainer
        if self.cfg.model_config_type == "mamba":
            return AxolotlMambaTrainer
        return AxolotlTrainer

    def build(self, total_num_steps):
        warmup_steps = None
        if self.cfg.warmup_steps is not None:
            warmup_steps = self.cfg.warmup_steps
        elif self.cfg.warmup_ratio is not None:
            warmup_steps = max(int(self.cfg.warmup_ratio * total_num_steps), 0)
        else:
            warmup_steps = min(int(0.03 * total_num_steps), 100)

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
            else:
                training_arguments_kwargs["gradient_checkpointing_kwargs"] = {
                    "use_reentrant": False
                }
        if self.cfg.fsdp:
            training_arguments_kwargs["fsdp"] = self.cfg.fsdp
            if self.cfg.fsdp_config:
                training_arguments_kwargs["fsdp_config"] = dict(self.cfg.fsdp_config)

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

        if self.cfg.sample_packing_eff_est:
            training_arguments_kwargs[
                "sample_packing_efficiency"
            ] = self.cfg.sample_packing_eff_est

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

        if not self.cfg.test_datasets and self.cfg.val_set_size == 0:
            # no eval set, so don't eval
            training_arguments_kwargs["evaluation_strategy"] = "no"
        elif self.cfg.eval_steps:
            training_arguments_kwargs["evaluation_strategy"] = "steps"
            training_arguments_kwargs["eval_steps"] = self.cfg.eval_steps
        elif self.cfg.evaluation_strategy:
            training_arguments_kwargs[
                "evaluation_strategy"
            ] = self.cfg.evaluation_strategy
        else:
            # we have an eval set, but no steps defined, default to use epoch
            training_arguments_kwargs["evaluation_strategy"] = "epoch"

        if self.cfg.save_steps:
            training_arguments_kwargs["save_strategy"] = "steps"
            training_arguments_kwargs["save_steps"] = self.cfg.save_steps
        elif self.cfg.save_strategy:
            training_arguments_kwargs["save_strategy"] = self.cfg.save_strategy
        else:
            # default to saving each epoch if not defined
            training_arguments_kwargs["save_strategy"] = "epoch"

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
        report_to = None
        if self.cfg.use_wandb:
            report_to = "wandb"
        if self.cfg.use_mlflow:
            report_to = "mlflow"
        training_arguments_kwargs["report_to"] = report_to
        training_arguments_kwargs["run_name"] = (
            self.cfg.wandb_name if self.cfg.use_wandb else None
        )
        training_arguments_kwargs["optim"] = (
            self.cfg.optimizer if self.cfg.optimizer else "adamw_hf"
        )
        training_arguments_kwargs["lr_scheduler_type"] = (
            self.cfg.lr_scheduler
            if self.cfg.lr_scheduler
            and self.cfg.lr_scheduler not in ("one_cycle", "log_sweep")
            else "cosine"
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
        training_arguments_kwargs["sample_packing"] = (
            self.cfg.sample_packing if self.cfg.sample_packing else False
        )
        training_arguments_kwargs["multipack_real_batches"] = (
            self.cfg.flash_attention is not True
        )
        training_arguments_kwargs["eval_sample_packing"] = (
            self.cfg.sample_packing
            if self.cfg.eval_sample_packing is not False
            else False
        )
        training_arguments_kwargs[
            "sample_packing_seq_len_multiplier"
        ] = self.cfg.micro_batch_size
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

        training_arguments_kwargs = self.hook_pre_create_training_args(
            training_arguments_kwargs
        )
        training_arguments_kwargs["model_type"] = self.cfg.model_config_type
        training_arguments_kwargs["pretraining"] = bool(self.cfg.pretraining_dataset)

        if self.cfg.neftune_noise_alpha is not None:
            training_arguments_kwargs[
                "neftune_noise_alpha"
            ] = self.cfg.neftune_noise_alpha

        trainer_kwargs = {}

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

        training_args = (
            AxolotlTrainingArguments(  # pylint: disable=unexpected-keyword-arg
                **training_arguments_kwargs,
            )
        )
        training_args = self.hook_post_create_training_args(training_args)

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

        trainer_cls = self._get_trainer_cls()
        trainer_kwargs, trainer_cls = self.hook_pre_create_trainer(
            trainer_kwargs, trainer_cls
        )
        trainer = trainer_cls(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=training_args,
            data_collator=self.build_collator(training_args, **data_collator_kwargs),
            eval_data_collator=self.build_collator(
                training_args, is_eval=True, **data_collator_kwargs
            ),
            bench_data_collator=transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                return_tensors="pt",
                **data_collator_kwargs,
            ),
            callbacks=self.get_callbacks(),
            num_epochs=self.cfg.num_epochs,
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
            ]
        ]
        if use_batch_sampler_collator:
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
            collator = DataCollatorForSeq2Seq

        return collator(
            self.tokenizer,
            return_tensors="pt",
            **kwargs,
        )


class HFDPOTrainerBuilder(TrainerBuilderBase):
    """
    Trainer factory class for DPO Trainer
    """

    def get_callbacks(self):
        callbacks = super().get_callbacks()
        return callbacks

    def get_post_trainer_create_callbacks(self, trainer):
        callbacks = []
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
            training_args_kwargs["evaluation_strategy"] = "steps"
            training_args_kwargs["eval_steps"] = self.cfg.eval_steps
        else:
            training_args_kwargs["evaluation_strategy"] = "no"
        if self.cfg.bf16 or self.cfg.bfloat16:
            training_args_kwargs["bf16"] = True

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

        training_args = TrainingArguments(
            per_device_train_batch_size=self.cfg.micro_batch_size,
            max_steps=self.cfg.max_steps or total_num_steps,
            gradient_accumulation_steps=self.cfg.gradient_accumulation_steps,
            learning_rate=self.cfg.learning_rate,
            output_dir=self.cfg.output_dir,
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
            dpo_trainer_kwargs["loss_type"] = "ipo"
            if self.cfg.dpo_label_smoothing:
                dpo_trainer_kwargs["label_smoothing"] = self.cfg.dpo_label_smoothing
        elif self.cfg.rl == "kto_pair":
            dpo_trainer_kwargs["loss_type"] = "kto_pair"
        if self.eval_dataset:
            dpo_trainer_kwargs["eval_dataset"] = self.eval_dataset
        if self.cfg.adapter and self.peft_config:
            dpo_trainer_kwargs["peft_config"] = self.peft_config
        if self.cfg.precompute_ref_log_probs is not None:
            dpo_trainer_kwargs[
                "precompute_ref_log_probs"
            ] = self.cfg.precompute_ref_log_probs
        dpo_trainer = AxolotlDPOTrainer(
            self.model,
            self.model_ref,
            args=training_args,
            beta=self.cfg.dpo_beta or 0.1,
            train_dataset=self.train_dataset,
            tokenizer=self.tokenizer,
            max_length=self.cfg.sequence_len,
            max_target_length=None,
            max_prompt_length=self.cfg.sequence_len,
            generate_during_eval=True,
            callbacks=self.get_callbacks(),
            **dpo_trainer_kwargs,
        )
        dpo_trainer = self.hook_post_create_trainer(dpo_trainer)
        for callback in self.get_post_trainer_create_callbacks(dpo_trainer):
            dpo_trainer.add_callback(callback)

        return dpo_trainer


class HFPPOTrainerBuilder(TrainerBuilderBase):
    """
    HF Factory class for PPO Trainer
    """

    def get_callbacks(self):
        callbacks = []
        return callbacks

    def get_post_trainer_create_callbacks(self, trainer):
        callbacks = []
        return callbacks

    def build(self, total_num_steps):
        # build PPOConfig
        pass
