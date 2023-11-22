"""
Builder for the training args and trainer
"""

import abc
import importlib
import logging
import math
import sys
from abc import abstractmethod
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import transformers
from datasets import Dataset
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers.trainer_utils import seed_worker

from axolotl.monkeypatch.relora import ReLoRACallback, ReLoRAScheduler
from axolotl.utils.callbacks import (
    EvalFirstStepCallback,
    GPUStatsCallback,
    SaveAxolotlConfigtoWandBCallback,
    SaveBetterTransformerModelCallback,
    bench_eval_callback_factory,
    log_prediction_callback_factory,
)
from axolotl.utils.collators import BatchSamplerDataCollatorForSeq2Seq
from axolotl.utils.samplers import MultipackBatchSampler
from axolotl.utils.schedulers import get_cosine_schedule_with_quadratic_warmup

try:
    import torch._dynamo  # pylint: disable=ungrouped-imports
except ImportError:
    pass

LOG = logging.getLogger("axolotl.core.trainer_builder")


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
    dataloader_prefetch_factor: Optional[int] = field(
        default=None,
        metadata={"help": "prefetch_factor argument to the dataloader"},
    )


class AxolotlTrainer(Trainer):
    """
    Extend the base Trainer for axolotl helpers
    """

    args = None  # type: AxolotlTrainingArguments

    def __init__(self, *args, num_epochs=1, bench_data_collator=None, **kwargs):
        self.num_epochs = num_epochs
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
        if self.args.sample_packing:
            return MultipackBatchSampler(
                RandomSampler(self.train_dataset),
                self.args.train_batch_size,
                drop_last=True,
                batch_max_len=self._train_batch_size * self.args.max_seq_length,
                lengths=(
                    self.train_dataset.data.column("position_ids")
                    .to_pandas()
                    .apply(lambda x: x[-1] + 1)
                    .values
                ),
                packing_efficiency_estimate=self.args.sample_packing_efficiency,
            )
        return super()._get_train_sampler()

    def _get_eval_sampler(
        self, eval_dataset: Dataset
    ) -> Optional[torch.utils.data.Sampler]:
        if self.args.sample_packing and self.args.eval_sample_packing is not False:
            return MultipackBatchSampler(
                SequentialSampler(eval_dataset),
                self.args.per_device_eval_batch_size,
                drop_last=True,
                batch_max_len=self.args.eval_batch_size * self.args.max_seq_length,
                lengths=(
                    eval_dataset.data.column("position_ids")
                    .to_pandas()
                    .apply(lambda x: x[-1] + 1)
                    .values
                ),
                packing_efficiency_estimate=self.args.sample_packing_efficiency,
            )
        return super()._get_eval_sampler(eval_dataset)

    def get_train_dataloader(self) -> DataLoader:
        if self.args.sample_packing:
            train_dataset = self.train_dataset
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


class TrainerBuilderBase(abc.ABC):
    """
    Base class for trainer builder
    """

    _train_dataset = None
    _eval_dataset = None

    def __init__(self, cfg, model, tokenizer):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer

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

    @abstractmethod
    def build(self, total_num_steps):
        pass

    @abstractmethod
    def get_callbacks(self):
        pass

    @abstractmethod
    def get_post_trainer_create_callbacks(self, trainer):
        """
        Callbacks added after the trainer is created, usually b/c these need access to the trainer
        """


class HFCausalTrainerBuilder(TrainerBuilderBase):
    """
    Build the HuggingFace training args/trainer for Causal models
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

    def get_callbacks(self):
        callbacks = []
        callbacks.append(GPUStatsCallback(self.cfg))
        callbacks.append(EvalFirstStepCallback)

        if self.cfg.relora_steps:
            callbacks.append(ReLoRACallback(self.cfg))

        if (
            hasattr(self.model, "use_bettertransformer")
            and self.model.use_bettertransformer is True
        ):
            callbacks.append(SaveBetterTransformerModelCallback)

        if self.cfg.use_wandb:
            callbacks.append(
                SaveAxolotlConfigtoWandBCallback(self.cfg.axolotl_config_path)
            )

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
        return AxolotlTrainer

    def build(self, total_num_steps):
        warmup_steps = (
            self.cfg.warmup_steps
            if self.cfg.warmup_steps is not None
            else min(int(0.03 * total_num_steps), 100)
        )
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

            if self.cfg.hub_strategy:
                training_arguments_kwargs["hub_strategy"] = self.cfg.hub_strategy

        if self.cfg.save_safetensors:
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

        if self.cfg.val_set_size == 0:
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
            and self.cfg.val_set_size > 0
            and self.cfg.save_steps
            and self.cfg.eval_steps
            and self.cfg.save_steps % self.cfg.eval_steps == 0
        ) or False
        training_arguments_kwargs["ddp_find_unused_parameters"] = (
            False if self.cfg.ddp else None
        )
        training_arguments_kwargs["group_by_length"] = self.cfg.group_by_length
        training_arguments_kwargs["report_to"] = "wandb" if self.cfg.use_wandb else None
        training_arguments_kwargs["run_name"] = (
            self.cfg.wandb_run_id if self.cfg.use_wandb else None
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
        training_arguments_kwargs["weight_decay"] = (
            self.cfg.weight_decay if self.cfg.weight_decay is not None else 0.0
        )
        training_arguments_kwargs["sample_packing"] = (
            self.cfg.sample_packing if self.cfg.sample_packing else False
        )
        training_arguments_kwargs["eval_sample_packing"] = (
            self.cfg.sample_packing
            if self.cfg.eval_sample_packing is not False
            else False
        )
        training_arguments_kwargs[
            "sample_packing_seq_len_multiplier"
        ] = self.cfg.micro_batch_size
        training_arguments_kwargs["relora_steps"] = self.cfg.relora_steps
        training_arguments_kwargs["relora_warmup_steps"] = self.cfg.relora_warmup_steps
        training_arguments_kwargs = self.hook_pre_create_training_args(
            training_arguments_kwargs
        )
        training_args = (
            AxolotlTrainingArguments(  # pylint: disable=unexpected-keyword-arg
                **training_arguments_kwargs,
            )
        )
        training_args = self.hook_post_create_training_args(training_args)
        trainer_kwargs = {}

        if self.cfg.optimizer == "adamw_anyprecision":
            if Path(self.cfg.torchdistx_path).exists():
                sys.path.append(self.cfg.torchdistx_path)
                importlib.import_module("torchdistx")

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

        if self.cfg.is_llama_derived_model and self.cfg.landmark_attention:
            from axolotl.monkeypatch.llama_landmark_attn import (
                add_mem_tokens,
                get_mem_id,
                set_model_mem_id,
            )

            set_model_mem_id(self.model, self.tokenizer)

            LOG.info("Adding landmark attention tokens to dataset")

            for dataset in [self.train_dataset, self.eval_dataset]:
                dataset = dataset.map(
                    partial(
                        add_mem_tokens, mem_freq=50, mem_id=get_mem_id(self.tokenizer)
                    ),
                    batched=False,
                    num_proc=32,
                )

        trainer_cls = self._get_trainer_cls()
        trainer_kwargs, trainer_cls = self.hook_pre_create_trainer(
            trainer_kwargs, trainer_cls
        )
        trainer = trainer_cls(
            model=self.model,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            args=training_args,
            data_collator=BatchSamplerDataCollatorForSeq2Seq(
                self.tokenizer,
                return_tensors="pt",
                **data_collator_kwargs,
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
