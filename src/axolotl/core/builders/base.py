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

"""Base class for trainer builder"""

import abc
import importlib
import logging
import sys
from abc import abstractmethod
from contextlib import suppress
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback
from transformers.trainer_pt_utils import AcceleratorConfig

from axolotl.integrations.base import PluginManager
from axolotl.monkeypatch.trainer.lr import patch_trainer_get_lr
from axolotl.telemetry.callbacks import TelemetryCallback
from axolotl.telemetry.manager import TelemetryManager
from axolotl.utils import (
    is_comet_available,
    is_mlflow_available,
    is_opentelemetry_available,
    is_trackio_available,
)
from axolotl.utils.callbacks import (
    GCCallback,
    SaveAxolotlConfigtoWandBCallback,
    SaveModelOnFirstStepCallback,
)
from axolotl.utils.callbacks.profiler import PytorchProfilerCallback
from axolotl.utils.distributed import build_parallelism_config
from axolotl.utils.schemas.enums import CustomSupportedOptimizers

LOG = logging.getLogger(__name__)

with suppress(ImportError):
    import torch._dynamo


class TrainerBuilderBase(abc.ABC):
    """Base class for trainer builder."""

    def __init__(self, cfg, model, tokenizer, processor=None):
        self.cfg = cfg
        self.model = model
        self.tokenizer = tokenizer
        self.processor = processor

        self._train_dataset = None
        self._eval_dataset = None
        self._model_ref = None
        self._peft_config = None

        # If the model supports tagging, add the axolotl tag.
        # This makes sure the tag is correctly pushed even if a user calls
        # model.push_to_hub instead of trainer.push_to_hub.
        if hasattr(model, "add_model_tags"):
            model.add_model_tags(["axolotl"])

        patch_trainer_get_lr()

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

    def get_callbacks(self) -> list[TrainerCallback]:
        callbacks = []

        plugin_manager = PluginManager.get_instance()
        callbacks.extend(
            plugin_manager.add_callbacks_pre_trainer(cfg=self.cfg, model=self.model)
        )

        if self.cfg.gc_steps:
            callbacks.append(GCCallback(gc_steps=self.cfg.gc_steps))

        if self.cfg.dynamic_checkpoint and self.cfg.dynamic_checkpoint.enabled:
            from axolotl.utils.callbacks.dynamic_checkpoint import (
                DynamicCheckpointCallback,
            )

            callbacks.append(DynamicCheckpointCallback(self.cfg))

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
        if self.cfg.use_trackio and is_trackio_available():
            from axolotl.utils.callbacks.trackio_ import (
                SaveAxolotlConfigtoTrackioCallback,
            )

            callbacks.append(
                SaveAxolotlConfigtoTrackioCallback(self.cfg.axolotl_config_path)
            )
        if self.cfg.use_otel_metrics and is_opentelemetry_available():
            from axolotl.utils.callbacks.opentelemetry import (
                OpenTelemetryMetricsCallback,
            )

            callbacks.append(OpenTelemetryMetricsCallback(self.cfg))
        if self.cfg.save_first_step:
            callbacks.append(SaveModelOnFirstStepCallback())

        if self.cfg.profiler_steps:
            callbacks.append(
                PytorchProfilerCallback(
                    steps_to_profile=self.cfg.profiler_steps,
                    profiler_steps_start=self.cfg.profiler_steps_start,
                )
            )

        telemetry_manager = TelemetryManager.get_instance()
        if telemetry_manager.enabled:
            callbacks.append(TelemetryCallback())

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

    def _configure_warmup_and_logging(
        self, total_num_steps: int, training_args_kwargs: dict
    ):
        warmup_steps: int | float = 0
        warmup_ratio = 0.0
        if self.cfg.warmup_steps is not None:
            warmup_steps = self.cfg.warmup_steps
        elif self.cfg.warmup_ratio is not None:
            if total_num_steps:
                warmup_steps = max(int(self.cfg.warmup_ratio * total_num_steps), 0)
            else:
                warmup_ratio = self.cfg.warmup_ratio
        elif total_num_steps:
            warmup_steps = min(int(0.03 * total_num_steps), 100)
        else:
            warmup_ratio = 0.03

        # transformers v5
        if warmup_ratio > 0.0 and warmup_steps == 0:
            warmup_steps = warmup_ratio

        if warmup_steps == 1:
            warmup_steps = 2

        if self.cfg.logging_steps is not None:
            training_args_kwargs["logging_steps"] = self.cfg.logging_steps
        else:
            training_args_kwargs["logging_steps"] = (
                500  # transformers defaults to 500
                if not total_num_steps
                else max(min(int(0.005 * total_num_steps), 10), 1)
            )

        training_args_kwargs["warmup_steps"] = warmup_steps

    def _configure_precision_settings(self, training_args_kwargs: dict):
        training_args_kwargs["fp16"] = (self.cfg.fp16 and not self.cfg.bf16) or False
        training_args_kwargs["tf32"] = self.cfg.tf32
        if self.cfg.bf16 == "full":
            training_args_kwargs["bf16_full_eval"] = True
        else:
            bf16 = self.cfg.bf16 or self.cfg.bfloat16
            bf16 = bf16 if bf16 is not None else False
            training_args_kwargs["bf16"] = bf16

    def _configure_scheduler(self, training_args_kwargs: dict):
        if self.cfg.lr_scheduler in ["one_cycle", "rex"]:
            training_args_kwargs["lr_scheduler_type"] = "cosine"
            training_args_kwargs["alternate_lr_scheduler_type"] = self.cfg.lr_scheduler
        else:
            training_args_kwargs["lr_scheduler_type"] = (
                self.cfg.lr_scheduler if self.cfg.lr_scheduler else "cosine"
            )
        training_args_kwargs["lr_scheduler_kwargs"] = (
            self.cfg.lr_scheduler_kwargs if self.cfg.lr_scheduler_kwargs else {}
        )

    def _configure_optimizer(self, training_args_kwargs: dict, trainer_kwargs: dict):
        def _configure_custom_optimizer(
            training_args_kwargs: dict, trainer_kwargs: dict
        ):
            # Common optimizer kwargs
            optimizer_kwargs = {
                "lr": training_args_kwargs["learning_rate"],
                "weight_decay": training_args_kwargs["weight_decay"],
            }

            # Adam-specific kwargs
            adam_kwargs: dict = {}
            if training_args_kwargs.get("adam_beta1") and training_args_kwargs.get(
                "adam_beta2"
            ):
                adam_kwargs["betas"] = (
                    training_args_kwargs.get("adam_beta1"),
                    training_args_kwargs.get("adam_beta2"),
                )
            if training_args_kwargs.get("adam_epsilon"):
                adam_kwargs["eps"] = training_args_kwargs.get("adam_epsilon")

            if self.cfg.optimizer == "muon":
                _, device_mesh = build_parallelism_config(self.cfg)

                if device_mesh is not None:
                    from axolotl.contribs.mit.muon.dist_muon import (
                        DistMuonOptimizerFactory,
                    )

                    optimizer_cls = DistMuonOptimizerFactory
                    optimizer_kwargs["device_mesh"] = device_mesh
                else:
                    from axolotl.contribs.mit.muon import (
                        MuonOptimizerFactory,
                    )

                    optimizer_cls = MuonOptimizerFactory

                optimizer_kwargs.update(adam_kwargs)
            elif self.cfg.optimizer == "dion":
                from axolotl.contribs.mit.dion import (
                    DionOptimizerFactory,
                )

                optimizer_cls = DionOptimizerFactory
                optimizer_kwargs["dion_lr"] = training_args_kwargs["dion_learning_rate"]
                optimizer_kwargs["dion_mu"] = training_args_kwargs["dion_momentum"]
                optimizer_kwargs.update(adam_kwargs)
                _, device_mesh = build_parallelism_config(self.cfg)
                if device_mesh is not None:
                    optimizer_kwargs["device_mesh"] = device_mesh
            elif self.cfg.optimizer == "optimi_adamw":
                from optimi import AdamW

                optimizer_kwargs["foreach"] = False
                optimizer_cls = AdamW
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
            elif self.cfg.optimizer == "came_pytorch":
                from came_pytorch import CAME

                optimizer_cls = CAME

                beta1 = training_args_kwargs.get("adam_beta1", 0.9)
                beta2 = training_args_kwargs.get("adam_beta2", 0.999)
                beta3 = training_args_kwargs.get("adam_beta3", 0.9999)
                eps1 = training_args_kwargs.get("adam_epsilon", 1e-30)
                eps2 = training_args_kwargs.get("adam_epsilon2", 1e-16)
                adam_kwargs["betas"] = (beta1, beta2, beta3)
                adam_kwargs["eps"] = (eps1, eps2)

                optimizer_kwargs.update(adam_kwargs)
            else:
                raise ValueError(
                    f"Unhandled optimizer: {self.cfg.optimizer}. Please raise an Issue."
                )

            # Parse any additional optimizer args from config
            if self.cfg.optim_args:
                if isinstance(self.cfg.optim_args, dict):
                    optimizer_kwargs.update(self.cfg.optim_args)
                else:
                    # Parse string format "key1=value1,key2=value2"
                    for mapping in self.cfg.optim_args.replace(" ", "").split(","):
                        key, value = mapping.split("=")
                        optimizer_kwargs[key] = value

            # Note: This is not used in training_args_kwargs, but in trainer_kwargs
            trainer_kwargs["optimizer_cls_and_kwargs"] = (
                optimizer_cls,
                optimizer_kwargs,
            )

        # Handle custom optimizer
        custom_supported_optimizers = [opt.value for opt in CustomSupportedOptimizers]
        if self.cfg.optimizer in custom_supported_optimizers:
            _configure_custom_optimizer(training_args_kwargs, trainer_kwargs)
        else:
            # Use transformers' optimizer
            training_args_kwargs["optim"] = self.cfg.optimizer

            # Parse any additional optimizer args from config
            if self.cfg.optim_args:
                if isinstance(self.cfg.optim_args, dict):
                    optim_args = ",".join(
                        [f"{key}={value}" for key, value in self.cfg.optim_args.items()]
                    )
                else:
                    optim_args = self.cfg.optim_args
                training_args_kwargs["optim_args"] = optim_args

            if (
                self.cfg.optimizer == "adamw_anyprecision"
                and Path(self.cfg.torchdistx_path).exists()
            ):
                sys.path.append(self.cfg.torchdistx_path)
                importlib.import_module("torchdistx")

    def _configure_hub_parameters(self, training_args_kwargs: dict):
        if self.cfg.hub_model_id:
            training_args_kwargs["hub_model_id"] = self.cfg.hub_model_id
            training_args_kwargs["push_to_hub"] = True
            training_args_kwargs["hub_private_repo"] = True
            training_args_kwargs["hub_always_push"] = True

            if self.cfg.hub_strategy:
                training_args_kwargs["hub_strategy"] = self.cfg.hub_strategy

    def _configure_save_and_eval_strategy(self, training_args_kwargs: dict):
        # save_strategy and save_steps
        if self.cfg.save_steps:
            training_args_kwargs["save_strategy"] = "steps"
            training_args_kwargs["save_steps"] = self.cfg.save_steps
        elif self.cfg.save_strategy:
            training_args_kwargs["save_strategy"] = self.cfg.save_strategy
        else:
            # default to saving each epoch if not defined
            training_args_kwargs["save_strategy"] = "epoch"

        training_args_kwargs["save_total_limit"] = (
            self.cfg.save_total_limit if self.cfg.save_total_limit else 4
        )

        # eval_strategy and eval_steps
        if not self.eval_dataset and self.cfg.val_set_size == 0:
            # do not eval if no eval_dataset and val_set_size=0
            training_args_kwargs["eval_strategy"] = "no"
        elif self.cfg.eval_steps:
            training_args_kwargs["eval_strategy"] = "steps"
            training_args_kwargs["eval_steps"] = self.cfg.eval_steps
            training_args_kwargs["eval_on_start"] = True
        elif self.cfg.eval_strategy:
            training_args_kwargs["eval_strategy"] = self.cfg.eval_strategy
            training_args_kwargs["eval_on_start"] = True

    def _configure_reporting(self, training_args_kwargs: dict):
        report_to = []
        if self.cfg.use_wandb:
            report_to.append("wandb")
        if self.cfg.use_mlflow:
            report_to.append("mlflow")
        if self.cfg.use_tensorboard:
            report_to.append("tensorboard")
        if self.cfg.use_comet:
            report_to.append("comet_ml")
        if self.cfg.use_trackio:
            report_to.append("trackio")

        training_args_kwargs["report_to"] = report_to

        if self.cfg.use_wandb:
            training_args_kwargs["run_name"] = self.cfg.wandb_name
        elif self.cfg.use_mlflow:
            training_args_kwargs["run_name"] = self.cfg.mlflow_run_name
        elif self.cfg.use_trackio:
            training_args_kwargs["run_name"] = self.cfg.trackio_run_name
        else:
            training_args_kwargs["run_name"] = None

    def _configure_torch_compile(self, training_args_kwargs: dict):
        if self.cfg.torch_compile and getattr(torch, "_dynamo", None):
            torch._dynamo.config.suppress_errors = True
            torch._dynamo.config.accumulated_cache_size_limit = 256
            training_args_kwargs["torch_compile"] = self.cfg.torch_compile
            if self.cfg.torch_compile_backend:
                training_args_kwargs["torch_compile_backend"] = (
                    self.cfg.torch_compile_backend
                )
            if self.cfg.torch_compile_mode:
                training_args_kwargs["torch_compile_mode"] = self.cfg.torch_compile_mode

    def _configure_accelerator_config(self, training_args_kwargs: dict):
        if self.cfg.accelerator_config:
            training_args_kwargs["accelerator_config"] = AcceleratorConfig(
                **self.cfg.accelerator_config
            )
        else:
            training_args_kwargs["accelerator_config"] = AcceleratorConfig()

    def _configure_gradient_checkpointing(self, training_args_kwargs: dict):
        if self.cfg.activation_offloading is True:
            # don't use the HF gradient checkpointing, manually wrap
            training_args_kwargs["gradient_checkpointing"] = False
            training_args_kwargs["activation_offloading"] = True
        elif self.cfg.gradient_checkpointing is not None:
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

    def _set_base_training_args(
        self, total_num_steps
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        training_args_kwargs: dict[str, Any] = {}
        trainer_kwargs: dict[str, Any] = {}

        self._configure_warmup_and_logging(total_num_steps, training_args_kwargs)
        self._configure_precision_settings(training_args_kwargs)
        self._configure_save_and_eval_strategy(training_args_kwargs)
        self._configure_gradient_checkpointing(training_args_kwargs)

        # set arg into trainer_args_kwargs with same name if value not None
        for arg in [
            # optim/scheduler
            "adam_beta1",
            "adam_beta2",
            "adam_beta3",
            "adam_epsilon",
            "adam_epsilon2",
            "cosine_min_lr_ratio",
            "cosine_constant_lr_ratio",
            "optim_target_modules",
            # trainer
            "max_grad_norm",
            "dataloader_num_workers",
            "dataloader_pin_memory",
            "dataloader_prefetch_factor",
            "gradient_accumulation_steps",
            "learning_rate",
            "embedding_lr",
            "embedding_lr_scale",
            "lr_groups",
            "loraplus_lr_ratio",
            "loraplus_lr_embedding",
            "output_dir",
            "save_only_model",
            "weight_decay",
            "seed",
            "dion_momentum",
            "dion_rank_fraction",
            "dion_rank_multiple_of",
            "dataset_num_proc",
        ]:
            if hasattr(self.cfg, arg) and getattr(self.cfg, arg) is not None:
                training_args_kwargs[arg] = getattr(self.cfg, arg)

        arg_map = {
            "dion_learning_rate": "dion_lr",
            "include_num_input_tokens_seen": "include_tokens_per_second",
        }
        for kwarg, cfg_arg in arg_map.items():
            if hasattr(self.cfg, cfg_arg) and getattr(self.cfg, cfg_arg) is not None:
                training_args_kwargs[kwarg] = getattr(self.cfg, cfg_arg)

        training_args_kwargs["per_device_train_batch_size"] = self.cfg.micro_batch_size
        training_args_kwargs["average_tokens_across_devices"] = False

        if self.cfg.eval_batch_size:
            training_args_kwargs["per_device_eval_batch_size"] = (
                self.cfg.eval_batch_size
            )

        training_args_kwargs["include_tkps"] = self.cfg.include_tkps
        training_args_kwargs["max_steps"] = self.cfg.max_steps or total_num_steps or -1
        training_args_kwargs["num_train_epochs"] = self.cfg.num_epochs

        # max_length is not used in CausalTrainer
        if self.cfg.reward_model or self.cfg.rl:
            training_args_kwargs["max_length"] = self.cfg.sequence_len

        if self.cfg.fsdp_config or self.cfg.fsdp:
            training_args_kwargs["fsdp_config"] = self.cfg.fsdp_config
            training_args_kwargs["fsdp"] = self.cfg.fsdp if self.cfg.fsdp else True

        self._configure_reporting(training_args_kwargs)
        self._configure_hub_parameters(training_args_kwargs)
        self._configure_scheduler(training_args_kwargs)
        self._configure_optimizer(training_args_kwargs, trainer_kwargs)
        self._configure_torch_compile(training_args_kwargs)
        self._configure_accelerator_config(training_args_kwargs)

        return training_args_kwargs, trainer_kwargs
