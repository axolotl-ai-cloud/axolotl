# Copyright 2024 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=too-many-lines
"""Builder for the training args and trainer"""

import abc
import importlib
import importlib.util
import inspect
import logging
import math
import sys
from abc import abstractmethod
from pathlib import Path
from typing import List, Type, Union

import torch
import transformers
from transformers import (
    DataCollatorWithFlattening,
    EarlyStoppingCallback,
    TrainerCallback,
)
from transformers.training_args import OptimizerNames
from trl.trainer.utils import RewardDataCollatorWithPadding

from axolotl.core.trainers import (
    AxolotlCPOTrainer,
    AxolotlKTOTrainer,
    AxolotlMambaTrainer,
    AxolotlORPOTrainer,
    AxolotlPRMTrainer,
    AxolotlRewardTrainer,
    AxolotlTrainer,
    ReLoRATrainer,
)
from axolotl.core.trainers.dpo import DPOStrategy
from axolotl.core.trainers.dpo.args import AxolotlDPOConfig
from axolotl.core.trainers.grpo import GRPOStrategy
from axolotl.core.training_args import (
    AxolotlCPOConfig,
    AxolotlKTOConfig,
    AxolotlORPOConfig,
    AxolotlPRMConfig,
    AxolotlRewardConfig,
    AxolotlTrainingArguments,
)
from axolotl.integrations.base import PluginManager
from axolotl.monkeypatch.multipack import SUPPORTED_MULTIPACK_MODEL_TYPES
from axolotl.monkeypatch.relora import ReLoRACallback
from axolotl.processing_strategies import get_processing_strategy
from axolotl.utils import is_comet_available, is_mlflow_available
from axolotl.utils.callbacks import (
    EvalFirstStepCallback,
    GCCallback,
    GPUStatsCallback,
    LossWatchDogCallback,
    SaveAxolotlConfigtoWandBCallback,
    SaveBetterTransformerModelCallback,
    bench_eval_callback_factory,
    causal_lm_bench_eval_callback_factory,
    log_prediction_callback_factory,
)
from axolotl.utils.callbacks.lisa import lisa_callback_factory
from axolotl.utils.callbacks.profiler import PytorchProfilerCallback
from axolotl.utils.chat_templates import get_chat_template_from_config
from axolotl.utils.collators import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    MambaDataCollator,
    V2BatchSamplerDataCollatorForSeq2Seq,
)
from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator
from axolotl.utils.models import ensure_dtype
from axolotl.utils.schemas.enums import CustomSupportedOptimizers

try:
    import torch._dynamo  # pylint: disable=ungrouped-imports
except ImportError:
    pass

LOG = logging.getLogger(__name__)


class TrainerBuilderBase(abc.ABC):
    """Base class for trainer builder."""

    _train_dataset = None
    _eval_dataset = None
    _model_ref = None
    _peft_config = None

    def __init__(self, cfg, model, tokenizer, processor=None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

        # If the model supports tagging, add the axolotl tag.
        # This makes sure the tag is correctly pushed even if a user calls
        # model.push_to_hub instead of trainer.push_to_hub.
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

        if self.cfg.profiler_steps:
            callbacks.append(
                PytorchProfilerCallback(
                    steps_to_profile=self.cfg.profiler_steps,
                )
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
    Build the HuggingFace training args/trainer for causal models and reward modeling
    using TRL.
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

        if self.cfg.gc_steps:
            callbacks.append(GCCallback(gc_steps=self.cfg.gc_steps))

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
        if self.cfg.plugins:
            plugin_manager = PluginManager.get_instance()
            trainer_cls = plugin_manager.get_trainer_cls(self.cfg)
            if trainer_cls:
                return trainer_cls
        if self.cfg.relora_steps:
            return ReLoRATrainer
        if self.cfg.model_config_type == "mamba":
            return AxolotlMambaTrainer
        if self.cfg.reward_model:
            return AxolotlRewardTrainer
        if self.cfg.process_reward_model:
            return AxolotlPRMTrainer
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

        if self.cfg.include_tokens_per_second is not None:
            training_arguments_kwargs["include_tokens_per_second"] = (
                self.cfg.include_tokens_per_second
            )

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
            training_arguments_kwargs["gradient_checkpointing"] = (
                self.cfg.gradient_checkpointing
            )
            if self.cfg.gradient_checkpointing_kwargs is not None:
                training_arguments_kwargs["gradient_checkpointing_kwargs"] = (
                    self.cfg.gradient_checkpointing_kwargs
                )
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
            training_arguments_kwargs["lr_quadratic_warmup"] = (
                self.cfg.lr_quadratic_warmup
            )

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
            training_arguments_kwargs["dataloader_pin_memory"] = (
                self.cfg.dataloader_pin_memory
            )
        if self.cfg.dataloader_num_workers is not None:
            training_arguments_kwargs["dataloader_num_workers"] = (
                self.cfg.dataloader_num_workers
            )
        if self.cfg.dataloader_prefetch_factor is not None:
            training_arguments_kwargs["dataloader_prefetch_factor"] = (
                self.cfg.dataloader_prefetch_factor
            )
        if self.cfg.dataloader_drop_last is not None:
            training_arguments_kwargs["dataloader_drop_last"] = (
                self.cfg.dataloader_drop_last
            )
        elif self.cfg.sample_packing and self.cfg.eval_sample_packing is False:
            training_arguments_kwargs["dataloader_drop_last"] = True

        if self.cfg.remove_unused_columns is not None:
            training_arguments_kwargs["remove_unused_columns"] = (
                self.cfg.remove_unused_columns
            )

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
            training_arguments_kwargs["metric_for_best_model"] = (
                self.cfg.metric_for_best_model
            )
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
                    training_arguments_kwargs["torch_compile_backend"] = (
                        self.cfg.torch_compile_backend
                    )
                if self.cfg.torch_compile_mode:
                    training_arguments_kwargs["torch_compile_mode"] = (
                        self.cfg.torch_compile_mode
                    )

        # DDP Config
        if self.cfg.ddp_timeout:
            training_arguments_kwargs["ddp_timeout"] = self.cfg.ddp_timeout
        # see https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
        if self.cfg.ddp_bucket_cap_mb:
            training_arguments_kwargs["ddp_bucket_cap_mb"] = self.cfg.ddp_bucket_cap_mb
        if self.cfg.ddp_broadcast_buffers is not None:
            training_arguments_kwargs["ddp_broadcast_buffers"] = (
                self.cfg.ddp_broadcast_buffers
            )

        # these are all the "standard" kwargs that are def used
        training_arguments_kwargs["max_steps"] = (
            total_num_steps if self.cfg.max_steps else -1
        )
        training_arguments_kwargs["max_seq_length"] = self.cfg.sequence_len
        training_arguments_kwargs["per_device_train_batch_size"] = (
            self.cfg.micro_batch_size
        )
        if self.cfg.eval_batch_size:
            training_arguments_kwargs["per_device_eval_batch_size"] = (
                self.cfg.eval_batch_size
            )
        if self.cfg.auto_find_batch_size is not None:
            training_arguments_kwargs["auto_find_batch_size"] = (
                self.cfg.auto_find_batch_size
            )
        training_arguments_kwargs["gradient_accumulation_steps"] = (
            self.cfg.gradient_accumulation_steps
        )
        training_arguments_kwargs["eval_accumulation_steps"] = (
            self.cfg.gradient_accumulation_steps
        )
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

        # handle ddp
        ddp_find_unused_parameters = None
        if self.cfg.ddp:
            ddp_find_unused_parameters = bool(self.cfg.ddp_find_unused_parameters)
        training_arguments_kwargs["ddp_find_unused_parameters"] = (
            ddp_find_unused_parameters
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

        if self.cfg.lr_scheduler in ["one_cycle", "rex", "log_sweep"]:
            training_arguments_kwargs["lr_scheduler_type"] = "cosine"
            training_arguments_kwargs["alternate_lr_scheduler_type"] = (
                self.cfg.lr_scheduler
            )
        else:
            training_arguments_kwargs["lr_scheduler_type"] = (
                self.cfg.lr_scheduler if self.cfg.lr_scheduler else "cosine"
            )
        training_arguments_kwargs["lr_scheduler_kwargs"] = (
            self.cfg.lr_scheduler_kwargs if self.cfg.lr_scheduler_kwargs else {}
        )
        training_arguments_kwargs["cosine_min_lr_ratio"] = self.cfg.cosine_min_lr_ratio
        training_arguments_kwargs["cosine_constant_lr_ratio"] = (
            self.cfg.cosine_constant_lr_ratio
        )
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
            training_arguments_kwargs["sample_packing_bin_size"] = (
                self.cfg.sample_packing_bin_size
            )
        if self.cfg.sample_packing_group_size is not None:
            training_arguments_kwargs["sample_packing_group_size"] = (
                self.cfg.sample_packing_group_size
            )
        if self.cfg.sample_packing_eff_est:
            training_arguments_kwargs["sample_packing_efficiency"] = (
                self.cfg.sample_packing_eff_est
            )

        if self.cfg.relora_steps:
            training_arguments_kwargs["relora_steps"] = self.cfg.relora_steps
            training_arguments_kwargs["relora_warmup_steps"] = (
                self.cfg.relora_warmup_steps
            )
            if self.cfg.relora_anneal_steps:
                training_arguments_kwargs["relora_anneal_steps"] = (
                    self.cfg.relora_anneal_steps
                )
            if self.cfg.relora_prune_ratio:
                training_arguments_kwargs["relora_prune_ratio"] = (
                    self.cfg.relora_prune_ratio
                )

        if self.cfg.lisa_step_interval and self.cfg.lisa_n_layers:
            training_arguments_kwargs["lisa_n_layers"] = self.cfg.lisa_n_layers
            training_arguments_kwargs["lisa_step_interval"] = (
                self.cfg.lisa_step_interval
            )
            training_arguments_kwargs["lisa_layers_attribute"] = (
                self.cfg.lisa_layers_attribute
            )

        training_arguments_kwargs = self.hook_pre_create_training_args(
            training_arguments_kwargs
        )
        training_arguments_kwargs["model_type"] = self.cfg.model_config_type
        training_arguments_kwargs["pretraining"] = bool(self.cfg.pretraining_dataset)
        if self.cfg.chat_template:
            training_arguments_kwargs["chat_template"] = get_chat_template_from_config(
                cfg=self.cfg,
                tokenizer=self.tokenizer,
            )

        if self.cfg.neftune_noise_alpha is not None:
            training_arguments_kwargs["neftune_noise_alpha"] = (
                self.cfg.neftune_noise_alpha
            )

        trainer_kwargs = {}

        if self.cfg.reward_model:
            training_arguments_kwargs["max_length"] = self.cfg.sequence_len

        # Handle custom optimizer
        custom_supported_optimizers = [opt.value for opt in CustomSupportedOptimizers]
        if self.cfg.optimizer in custom_supported_optimizers:
            # Common optimizer kwargs
            optimizer_kwargs = {
                "lr": training_arguments_kwargs.get("learning_rate"),
                "weight_decay": training_arguments_kwargs.get("weight_decay"),
            }

            # Adam-specific kwargs
            adam_kwargs = {}
            if training_arguments_kwargs.get(
                "adam_beta1"
            ) and training_arguments_kwargs.get("adam_beta2"):
                adam_kwargs["betas"] = (
                    training_arguments_kwargs.get("adam_beta1"),
                    training_arguments_kwargs.get("adam_beta2"),
                )
            if training_arguments_kwargs.get("adam_epsilon"):
                adam_kwargs["eps"] = training_arguments_kwargs.get("adam_epsilon")

            if self.cfg.optimizer == "muon":
                from axolotl.contribs.mit.muon import (  # pylint: disable=no-name-in-module
                    MuonOptimizerFactory,
                )

                optimizer_cls = MuonOptimizerFactory
                optimizer_kwargs.update(adam_kwargs)
            elif self.cfg.optimizer == "optimi_adamw":
                from optimi import AdamW

                optimizer_kwargs["foreach"] = False
                optimizer_cls = AdamW
                optimizer_kwargs.update(adam_kwargs)
            elif self.cfg.optimizer == "ao_adamw_4bit":
                # TODO remove 20250401
                from torchao.prototype.low_bit_optim import AdamW4bit

                optimizer_cls = AdamW4bit
                optimizer_kwargs.update(adam_kwargs)

                LOG.warning(
                    f"`ao_adamw_4bit` will be deprecated soon. Please use `{OptimizerNames.ADAMW_TORCH_4BIT}` instead."
                )
            elif self.cfg.optimizer == "ao_adamw_8bit":
                from torchao.prototype.low_bit_optim import AdamW8bit

                optimizer_cls = AdamW8bit
                optimizer_kwargs.update(adam_kwargs)
            elif self.cfg.optimizer == "ao_adamw_fp8":
                from torchao.prototype.low_bit_optim import AdamWFp8

                optimizer_cls = AdamWFp8
                optimizer_kwargs.update(adam_kwargs)
            elif self.cfg.optimizer == "adopt_adamw":
                from axolotl.utils.optimizers.adopt import ADOPT

                optimizer_cls = ADOPT
                adam_kwargs["decouple"] = True
                optimizer_kwargs.update(adam_kwargs)

            # Parse any additional optimizer args from config
            if self.cfg.optim_args:
                if isinstance(self.cfg.optim_args, dict):
                    optimizer_kwargs.update(self.cfg.optim_args)
                else:
                    # Parse string format "key1=value1,key2=value2"
                    for mapping in self.cfg.optim_args.replace(" ", "").split(","):
                        key, value = mapping.split("=")
                        optimizer_kwargs[key] = value

            trainer_kwargs["optimizer_cls_and_kwargs"] = (
                optimizer_cls,
                optimizer_kwargs,
            )
        else:
            # Use transformers' optimizer
            training_arguments_kwargs["optim"] = self.cfg.optimizer

            # Parse any additional optimizer args from config
            if self.cfg.optim_args:
                if isinstance(self.cfg.optim_args, dict):
                    optim_args = ",".join(
                        [f"{key}={value}" for key, value in self.cfg.optim_args.items()]
                    )
                else:
                    optim_args = self.cfg.optim_args
                training_arguments_kwargs["optim_args"] = optim_args

        if self.cfg.optimizer == "adamw_anyprecision":
            if Path(self.cfg.torchdistx_path).exists():
                sys.path.append(self.cfg.torchdistx_path)
                importlib.import_module("torchdistx")

        if self.cfg.optim_target_modules:
            training_arguments_kwargs["optim_target_modules"] = (
                self.cfg.optim_target_modules
            )

        training_arguments_kwargs["embedding_lr"] = self.cfg.embedding_lr
        training_arguments_kwargs["embedding_lr_scale"] = self.cfg.embedding_lr_scale

        training_arguments_kwargs["loraplus_lr_ratio"] = self.cfg.loraplus_lr_ratio
        training_arguments_kwargs["loraplus_lr_embedding"] = (
            self.cfg.loraplus_lr_embedding
        )
        training_arguments_kwargs["lr_groups"] = self.cfg.lr_groups

        if self.cfg.accelerator_config:
            training_arguments_kwargs["accelerator_config"] = (
                self.cfg.accelerator_config
            )

        if self.cfg.image_size:
            training_arguments_kwargs["image_size"] = self.cfg.image_size
        if self.cfg.image_resize_algorithm:
            training_arguments_kwargs["image_resize_algorithm"] = (
                self.cfg.image_resize_algorithm
            )
        if self.cfg.kd_ce_alpha is not None:
            training_arguments_kwargs["kd_ce_alpha"] = self.cfg.kd_ce_alpha
        if self.cfg.kd_alpha is not None:
            training_arguments_kwargs["kd_alpha"] = self.cfg.kd_alpha
        if self.cfg.kd_temperature is not None:
            training_arguments_kwargs["kd_temperature"] = self.cfg.kd_temperature
        if self.cfg.kd_zscore_base_temp is not None:
            training_arguments_kwargs["kd_zscore_base_temp"] = (
                self.cfg.kd_zscore_base_temp
            )
        if self.cfg.kd_top_k_before_softmax is not None:
            training_arguments_kwargs["kd_top_k_before_softmax"] = (
                self.cfg.kd_top_k_before_softmax
            )

        training_arguments_kwargs["sequence_parallel_degree"] = (
            self.cfg.sequence_parallel_degree
        )

        if self.cfg.reward_model:
            training_args_cls = AxolotlRewardConfig
        elif self.cfg.process_reward_model:
            training_args_cls = AxolotlPRMConfig
        else:
            training_args_cls = AxolotlTrainingArguments
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
            if not (self.cfg.reward_model or self.cfg.process_reward_model):
                trainer_kwargs["eval_data_collator"] = eval_data_collator
        if not (self.cfg.reward_model or self.cfg.process_reward_model):
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
        if (
            not (trainer_cls in [AxolotlRewardTrainer, AxolotlPRMTrainer])
            and self.cfg.datasets is not None
        ):
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
            if (
                self.cfg.pretraining_sample_concatenation is False
                or self.cfg.micro_batch_size > 1
            ):
                return DataCollatorForSeq2Seq(self.tokenizer, **kwargs)
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
                DataCollatorWithFlattening,
                RewardDataCollatorWithPadding,
            ]
        ]
        collator_args = [self.tokenizer]
        if self.cfg.reward_model:
            collator = RewardDataCollatorWithPadding
            if "max_length" in kwargs:
                kwargs.pop("max_length")
        elif use_batch_sampler_collator:
            if self.cfg.flex_attention:
                collator = V2BatchSamplerDataCollatorForSeq2Seq
            elif self.cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES:
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
                kwargs["processing_strategy"] = get_processing_strategy(
                    self.processor,
                    training_args.chat_template,
                    self.cfg.chat_template,
                    image_size=training_args.image_size,
                    image_resize_algorithm=training_args.image_resize_algorithm,
                )
            elif self.cfg.batch_flattening:
                collator = DataCollatorWithFlattening
                collator_args.pop(0)
                kwargs.pop("pad_to_multiple_of", None)
                kwargs.pop("padding", None)
            elif self.cfg.kd_trainer:
                from axolotl.integrations.kd.collator import (
                    DataCollatorForKD,
                    KDBatchSamplerDataCollatorForSeq2Seq,
                )

                if self.cfg.sample_packing:
                    collator = KDBatchSamplerDataCollatorForSeq2Seq
                else:
                    collator = DataCollatorForKD
            else:
                collator = DataCollatorForSeq2Seq

        kwargs["return_tensors"] = "pt"
        if issubclass(collator, DataCollatorForSeq2Seq):
            kwargs["sequence_parallel_degree"] = training_args.sequence_parallel_degree

        return collator(
            *collator_args,
            **kwargs,
        )


class HFRLTrainerBuilder(TrainerBuilderBase):
    """Trainer factory class for TRL-based RLHF trainers (e.g. DPO)"""

    def get_callbacks(self):
        callbacks = super().get_callbacks()

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
            training_args_kwargs["remove_unused_columns"] = (
                self.cfg.remove_unused_columns
            )
        else:
            training_args_kwargs["remove_unused_columns"] = False

        if self.cfg.dataloader_pin_memory is not None:
            training_args_kwargs["dataloader_pin_memory"] = (
                self.cfg.dataloader_pin_memory
            )
        if self.cfg.dataloader_num_workers is not None:
            training_args_kwargs["dataloader_num_workers"] = (
                self.cfg.dataloader_num_workers
            )
        if self.cfg.dataloader_prefetch_factor is not None:
            training_args_kwargs["dataloader_prefetch_factor"] = (
                self.cfg.dataloader_prefetch_factor
            )
        if self.cfg.gradient_checkpointing:
            training_args_kwargs["gradient_checkpointing"] = (
                self.cfg.gradient_checkpointing
            )
            if self.cfg.gradient_checkpointing_kwargs is not None:
                training_args_kwargs["gradient_checkpointing_kwargs"] = (
                    self.cfg.gradient_checkpointing_kwargs
                )
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

        if self.cfg.dataset_processes:
            training_args_kwargs["dataset_num_proc"] = self.cfg.dataset_processes

        if (self.cfg.trl and self.cfg.trl.beta) or self.cfg.rl_beta:
            training_args_kwargs["beta"] = self.cfg.trl.beta or self.cfg.rl_beta
        if self.cfg.orpo_alpha:
            # trl does some odd mapping of alpha to beta to reuse the beta parameter ???
            training_args_kwargs["beta"] = self.cfg.orpo_alpha

        if self.cfg.rpo_alpha is not None:
            training_args_kwargs["rpo_alpha"] = self.cfg.rpo_alpha

        training_args_cls = None
        blocklist_args_kwargs = []
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

            training_args_kwargs["max_length"] = self.cfg.sequence_len
            if self.cfg.max_prompt_len:
                training_args_kwargs["max_prompt_length"] = self.cfg.max_prompt_len

        elif self.cfg.rl == "grpo":
            training_args_cls = GRPOStrategy.get_training_args_class()
            training_args_kwargs.update(GRPOStrategy.set_training_args_kwargs(self.cfg))
            blocklist_args_kwargs = GRPOStrategy.get_blocklist_args_kwargs()

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
            if self.cfg.dpo_use_logits_to_keep is not None:
                training_args_kwargs["use_logits_to_keep"] = (
                    self.cfg.dpo_use_logits_to_keep
                )

        for blocklist_key in blocklist_args_kwargs:
            if blocklist_key in training_args_kwargs:
                del training_args_kwargs[blocklist_key]

        max_steps = self.cfg.max_steps or total_num_steps or -1
        training_args_kwargs["num_train_epochs"] = self.cfg.num_epochs
        training_args = training_args_cls(  # pylint: disable=unexpected-keyword-arg
            self.cfg.output_dir,
            per_device_train_batch_size=self.cfg.micro_batch_size,
            max_steps=max_steps,
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
            dpo_trainer_kwargs["precompute_ref_log_probs"] = (
                self.cfg.precompute_ref_log_probs
            )
        if self.cfg.rl == "grpo":
            trainer_cls = GRPOStrategy.get_trainer_class()
            trainer_cls_args = [self.model]
            trainer_cls_args.extend(GRPOStrategy.set_trainer_args(self.cfg))
            dpo_trainer_kwargs.update(GRPOStrategy.set_trainer_kwargs(self.cfg))
        elif self.cfg.rl in ["dpo", "ipo"]:
            trainer_cls = DPOStrategy.get_trainer_class()
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
        if "tokenizer" in sig.parameters.keys():
            dpo_trainer_kwargs["tokenizer"] = self.tokenizer
        else:
            dpo_trainer_kwargs["processing_class"] = self.tokenizer

        if self.cfg.datasets is not None and (
            trainer_cls is DPOStrategy.get_trainer_class()
        ):
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
