"""Base class trainer / training args builder implementation"""

import abc
from typing import Any

from torch import Type
from transformers import TrainerCallback
from transformers.training_args import TrainingArguments

from axolotl.integrations.base import PluginManager
from axolotl.monkeypatch.trainer.lr import patch_trainer_get_lr
from axolotl.utils import is_comet_available, is_mlflow_available
from axolotl.utils.callbacks import GCCallback, SaveAxolotlConfigtoWandBCallback
from axolotl.utils.callbacks.profiler import PytorchProfilerCallback

PLUGIN_MANAGER = PluginManager.get_instance()


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

    @abc.abstractmethod
    def build(self, total_num_steps):
        pass

    def get_common_training_args_kwargs(
        self, total_num_steps: int | None = None
    ) -> dict[str, Any]:
        """Get common training arguments kwargs used across different trainer types."""
        training_args_kwargs = {}

        # Common parameters
        for arg in [
            "adam_beta1",
            "adam_beta2",
            "adam_epsilon",
            "max_grad_norm",
            "dataloader_num_workers",
            "dataloader_pin_memory",
            "dataloader_prefetch_factor",
            "dataloader_drop_last",
            "remove_unused_columns",
        ]:
            if hasattr(self.cfg, arg) and getattr(self.cfg, arg) is not None:
                training_args_kwargs[arg] = getattr(self.cfg, arg)

        # Add Hub integration arguments if needed
        if self.cfg.hub_model_id:
            training_args_kwargs["hub_model_id"] = self.cfg.hub_model_id
            training_args_kwargs["push_to_hub"] = True
            training_args_kwargs["hub_private_repo"] = True
            training_args_kwargs["hub_always_push"] = True

            if self.cfg.hub_strategy:
                training_args_kwargs["hub_strategy"] = self.cfg.hub_strategy

        # BF16/FP16 settings
        if hasattr(self.cfg, "bf16") and self.cfg.bf16:
            if self.cfg.bf16 == "full":
                training_args_kwargs["bf16_full_eval"] = True
            else:
                training_args_kwargs["bf16"] = self.cfg.bf16
        elif hasattr(self.cfg, "bfloat16") and self.cfg.bfloat16:
            training_args_kwargs["bf16"] = True

        if hasattr(self.cfg, "fp16"):
            training_args_kwargs["fp16"] = (
                getattr(self.cfg, "fp16", False)
                and not getattr(self.cfg, "bf16", False)
            ) or False

        # Set save_strategy and save_steps
        if self.cfg.save_steps:
            training_args_kwargs["save_strategy"] = "steps"
            training_args_kwargs["save_steps"] = self.cfg.save_steps
        elif self.cfg.save_strategy:
            training_args_kwargs["save_strategy"] = self.cfg.save_strategy
        else:
            # default to saving each epoch if not defined
            training_args_kwargs["save_strategy"] = "epoch"

        # Handle safetensors
        if self.cfg.save_safetensors is not None:
            training_args_kwargs["save_safetensors"] = self.cfg.save_safetensors

        # Handle gradient checkpointing
        if self.cfg.gradient_checkpointing:
            training_args_kwargs["gradient_checkpointing"] = (
                self.cfg.gradient_checkpointing
            )
            if self.cfg.gradient_checkpointing_kwargs is not None:
                training_args_kwargs["gradient_checkpointing_kwargs"] = (
                    self.cfg.gradient_checkpointing_kwargs
                )

        # Common optimizer and LR scheduler settings
        training_args_kwargs["optim"] = self.cfg.optimizer
        if hasattr(self.cfg, "lr_scheduler") and self.cfg.lr_scheduler:
            training_args_kwargs["lr_scheduler_type"] = self.cfg.lr_scheduler
        else:
            training_args_kwargs["lr_scheduler_type"] = "cosine"

        if hasattr(self.cfg, "lr_scheduler_kwargs") and self.cfg.lr_scheduler_kwargs:
            training_args_kwargs["lr_scheduler_kwargs"] = self.cfg.lr_scheduler_kwargs
        else:
            training_args_kwargs["lr_scheduler_kwargs"] = {}

        # LoRA+ specific settings
        if hasattr(self.cfg, "loraplus_lr_ratio"):
            training_args_kwargs["loraplus_lr_ratio"] = self.cfg.loraplus_lr_ratio
        if hasattr(self.cfg, "loraplus_lr_embedding"):
            training_args_kwargs["loraplus_lr_embedding"] = (
                self.cfg.loraplus_lr_embedding
            )

        # Reporting tools
        report_to = []
        if self.cfg.use_wandb:
            report_to.append("wandb")
            if self.cfg.wandb_name:
                training_args_kwargs["run_name"] = self.cfg.wandb_name
        if self.cfg.use_mlflow:
            report_to.append("mlflow")
        if self.cfg.use_tensorboard:
            report_to.append("tensorboard")
        if self.cfg.use_comet:
            report_to.append("comet_ml")

        if report_to:
            training_args_kwargs["report_to"] = report_to

        # Basic training settings
        if hasattr(self.cfg, "sequence_len"):
            training_args_kwargs["max_length"] = self.cfg.sequence_len

        training_args_kwargs["save_only_model"] = getattr(
            self.cfg, "save_only_model", False
        )
        training_args_kwargs["save_total_limit"] = getattr(
            self.cfg, "save_total_limit", 5
        )

        # Compute warmup steps
        if hasattr(self.cfg, "warmup_steps") and self.cfg.warmup_steps is not None:
            training_args_kwargs["warmup_steps"] = self.cfg.warmup_steps
        elif (
            total_num_steps
            and hasattr(self.cfg, "warmup_ratio")
            and self.cfg.warmup_ratio is not None
        ):
            training_args_kwargs["warmup_steps"] = max(
                int(self.cfg.warmup_ratio * total_num_steps), 0
            )
        elif total_num_steps:
            training_args_kwargs["warmup_steps"] = min(int(0.03 * total_num_steps), 100)

        return training_args_kwargs

    def create_training_args(
        self,
        args_cls: Type[TrainingArguments],
        total_num_steps: int | None = None,
        **additional_kwargs,
    ) -> TrainingArguments:
        """Create training arguments with common logic."""
        # Get common trainings args and update with trainer-specific args
        training_args_kwargs = self.get_common_training_args_kwargs(total_num_steps)
        training_args_kwargs.update(additional_kwargs)

        # Create training args with pre- and post-creation hooks
        training_args_kwargs = self.hook_pre_create_training_args(training_args_kwargs)
        training_args = args_cls(**training_args_kwargs)
        training_args = self.hook_post_create_training_args(training_args)

        # Unset run_name so wandb sets up experiment names properly
        if self.cfg.use_wandb and training_args.run_name == training_args.output_dir:
            training_args.run_name = None

        return training_args

    def create_trainer(
        self, trainer_cls, training_args, trainer_args=None, trainer_kwargs=None
    ):
        """Create trainer with common logic."""
        if trainer_args is None:
            trainer_args = []
        if trainer_kwargs is None:
            trainer_kwargs = {}

        # Create trainer with pre- and post- creation hooks
        trainer_kwargs, trainer_cls = self.hook_pre_create_trainer(
            trainer_kwargs, trainer_cls
        )
        trainer = trainer_cls(
            *trainer_args,
            args=training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            callbacks=self.get_callbacks(),
            **trainer_kwargs,
        )
        trainer = self.hook_post_create_trainer(trainer)

        # Add post-creation callbacks
        for callback in self.get_post_trainer_create_callbacks(trainer):
            trainer.add_callback(callback)

        return trainer

    def get_callbacks(self) -> list[TrainerCallback]:
        callbacks = []
        callbacks.extend(
            PLUGIN_MANAGER.add_callbacks_pre_trainer(cfg=self.cfg, model=self.model)
        )

        if self.cfg.profiler_steps:
            callbacks.append(
                PytorchProfilerCallback(
                    steps_to_profile=self.cfg.profiler_steps,
                )
            )

        if self.cfg.gc_steps:
            callbacks.append(GCCallback(gc_steps=self.cfg.gc_steps))

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
        """Callbacks added after the trainer is created, usually because these need
        access to the trainer.
        """
        callbacks = []
        if self.cfg.plugins:
            callbacks.extend(
                [
                    cb
                    for cb in PLUGIN_MANAGER.add_callbacks_post_trainer(
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
