"""Causal trainer / training args builder implementation"""

import importlib
import inspect
import logging
import math
import os
import sys
from pathlib import Path
from typing import Type

import transformers
from transformers import (
    DataCollatorWithFlattening,
    EarlyStoppingCallback,
)
from transformers.training_args import OptimizerNames
from trl.trainer.utils import RewardDataCollatorWithPadding

from axolotl.core.trainers.base import AxolotlTrainer
from axolotl.core.trainers.builders.base import TrainerBuilderBase
from axolotl.core.trainers.mamba import AxolotlMambaTrainer
from axolotl.core.trainers.relora import ReLoRATrainer
from axolotl.core.trainers.trl import AxolotlPRMTrainer, AxolotlRewardTrainer
from axolotl.core.training_args import (
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
    GPUStatsCallback,
    LossWatchDogCallback,
    SaveBetterTransformerModelCallback,
    bench_eval_callback_factory,
    causal_lm_bench_eval_callback_factory,
    colab_inference_post_train_callback,
    log_prediction_callback_factory,
)
from axolotl.utils.callbacks.lisa import lisa_callback_factory
from axolotl.utils.chat_templates import get_chat_template_from_config
from axolotl.utils.collators.batching import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    V2BatchSamplerDataCollatorForSeq2Seq,
)
from axolotl.utils.collators.mamba import MambaDataCollator
from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator
from axolotl.utils.schemas.enums import CustomSupportedOptimizers

LOG = logging.getLogger(__name__)
PLUGIN_MANAGER = PluginManager.get_instance()


class HFCausalTrainerBuilder(TrainerBuilderBase):
    """Build the HuggingFace training args / trainer for causal models and reward
    modeling using TRL.
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

        if any("COLAB_" in key for key in os.environ):
            ColabCallback = colab_inference_post_train_callback(trainer)
            callbacks.append(ColabCallback(self.cfg))

        callbacks.extend(super().get_post_trainer_create_callbacks(trainer=trainer))
        return callbacks

    def _get_trainer_cls(self):
        if self.cfg.plugins:
            trainer_cls = PLUGIN_MANAGER.get_trainer_cls(self.cfg)
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
        """Build and return a causal trainer instance using the refactored base class."""
        # Get trainer class
        trainer_cls = self._get_trainer_cls()

        # Prepare training arguments
        training_args = self._prepare_training_args(total_num_steps)

        # Prepare data collators
        data_collator_kwargs = self._prepare_data_collator_kwargs()

        # Prepare trainer kwargs
        trainer_kwargs = self._prepare_trainer_kwargs(
            trainer_cls=trainer_cls,
            data_collator_kwargs=data_collator_kwargs,
            training_args=training_args,
        )

        # Create the trainer
        trainer = self.create_trainer(
            trainer_cls=trainer_cls,
            training_args=training_args,
            trainer_kwargs={
                "model": self.model,
                "data_collator": self.build_collator(
                    training_args, **data_collator_kwargs
                ),
                **trainer_kwargs,
            },
        )

        # Handle DeepSpeed config for sample packing if needed
        if self.cfg.deepspeed and self.cfg.sample_packing:
            trainer.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = self.cfg.micro_batch_size

        return trainer

    def _prepare_training_args(self, total_num_steps):
        """Prepare and return training arguments."""
        # Base training arguments
        training_args_kwargs = self._get_base_training_args()

        # Add feature configurations
        self._add_feature_configs(training_args_kwargs)

        # Handle optimizer configuration
        self._configure_optimizer(training_args_kwargs)

        # Create training args using the base class method
        training_args_cls = self._get_training_args_cls()

        return self.create_training_args(
            args_cls=training_args_cls,
            total_num_steps=total_num_steps,
            **training_args_kwargs,
        )

    def _get_base_training_args(self):
        """Return the base training arguments."""
        return {
            "max_steps": self.cfg.max_steps if self.cfg.max_steps else -1,
            "max_seq_length": self.cfg.sequence_len,
            "per_device_train_batch_size": self.cfg.micro_batch_size,
            "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
            "eval_accumulation_steps": self.cfg.gradient_accumulation_steps,
            "num_train_epochs": self.cfg.num_epochs,
            "learning_rate": self.cfg.learning_rate,
            "output_dir": self.cfg.output_dir,
            "weight_decay": (
                self.cfg.weight_decay if self.cfg.weight_decay is not None else 0.0
            ),
            "model_type": self.cfg.model_config_type,
            "pretraining": bool(self.cfg.pretraining_dataset),
            "sequence_parallel_degree": self.cfg.sequence_parallel_degree,
            "ring_attn_func": self.cfg.ring_attn_func,
            "embedding_lr": self.cfg.embedding_lr,
            "embedding_lr_scale": self.cfg.embedding_lr_scale,
            "loraplus_lr_ratio": self.cfg.loraplus_lr_ratio,
            "loraplus_lr_embedding": self.cfg.loraplus_lr_embedding,
            "lr_groups": self.cfg.lr_groups,
        }

    def _add_feature_configs(self, training_args_kwargs):
        """Add various feature configurations."""
        # Sample packing configurations
        self._add_sample_packing_configs(training_args_kwargs)

        # Batch size configurations
        if self.cfg.eval_batch_size:
            training_args_kwargs["per_device_eval_batch_size"] = (
                self.cfg.eval_batch_size
            )
        if self.cfg.auto_find_batch_size is not None:
            training_args_kwargs["auto_find_batch_size"] = self.cfg.auto_find_batch_size

        # Advanced training techniques (ReLoRA & Lisa)
        self._add_advanced_training_configs(training_args_kwargs)

        # Model-specific configurations
        self._add_model_specific_configs(training_args_kwargs)

    def _add_sample_packing_configs(self, training_args_kwargs):
        """Add sample packing configurations if applicable."""
        if hasattr(self.cfg, "sample_packing") and self.cfg.sample_packing:
            training_args_kwargs.update(
                {
                    "sample_packing": bool(self.cfg.sample_packing),
                    "multipack_real_batches": not self.cfg.flash_attention
                    or self.cfg.multipack_real_batches,
                    "eval_sample_packing": bool(self.cfg.eval_sample_packing),
                }
            )

            if self.cfg.sample_packing_bin_size is not None:
                training_args_kwargs["sample_packing_bin_size"] = (
                    self.cfg.sample_packing_bin_size
                )

            if self.cfg.sample_packing_group_size is not None:
                training_args_kwargs["sample_packing_group_size"] = (
                    self.cfg.sample_packing_group_size
                )

            if self.cfg.sample_packing_eff_est:
                training_args_kwargs["sample_packing_efficiency"] = (
                    self.cfg.sample_packing_eff_est
                )

    def _add_advanced_training_configs(self, training_args_kwargs):
        """Add advanced training techniques configurations (ReLoRA & Lisa)."""
        # ReLoRA configurations
        if self.cfg.relora_steps:
            training_args_kwargs.update(
                {
                    "relora_steps": self.cfg.relora_steps,
                    "relora_warmup_steps": self.cfg.relora_warmup_steps,
                }
            )
            if self.cfg.relora_anneal_steps:
                training_args_kwargs["relora_anneal_steps"] = (
                    self.cfg.relora_anneal_steps
                )
            if self.cfg.relora_prune_ratio:
                training_args_kwargs["relora_prune_ratio"] = self.cfg.relora_prune_ratio

        # Lisa configurations
        if self.cfg.lisa_step_interval and self.cfg.lisa_n_layers:
            training_args_kwargs.update(
                {
                    "lisa_n_layers": self.cfg.lisa_n_layers,
                    "lisa_step_interval": self.cfg.lisa_step_interval,
                    "lisa_layers_attribute": self.cfg.lisa_layers_attribute,
                }
            )

    def _add_model_specific_configs(self, training_args_kwargs):
        """Add model-specific configurations."""
        # Chat template
        if self.cfg.chat_template:
            training_args_kwargs["chat_template"] = get_chat_template_from_config(
                cfg=self.cfg,
                tokenizer=self.tokenizer,
            )

        # NEFTune
        if self.cfg.neftune_noise_alpha is not None:
            training_args_kwargs["neftune_noise_alpha"] = self.cfg.neftune_noise_alpha

        # Knowledge distillation configurations
        if self.cfg.kd_ce_alpha is not None:
            training_args_kwargs["kd_ce_alpha"] = self.cfg.kd_ce_alpha
        if self.cfg.kd_alpha is not None:
            training_args_kwargs["kd_alpha"] = self.cfg.kd_alpha
        if self.cfg.kd_temperature is not None:
            training_args_kwargs["kd_temperature"] = self.cfg.kd_temperature
        if self.cfg.kd_zscore_base_temp is not None:
            training_args_kwargs["kd_zscore_base_temp"] = self.cfg.kd_zscore_base_temp
        if self.cfg.kd_top_k_before_softmax is not None:
            training_args_kwargs["kd_top_k_before_softmax"] = (
                self.cfg.kd_top_k_before_softmax
            )

        # Image configurations
        if self.cfg.image_size:
            training_args_kwargs["image_size"] = self.cfg.image_size
        if self.cfg.image_resize_algorithm:
            training_args_kwargs["image_resize_algorithm"] = (
                self.cfg.image_resize_algorithm
            )

        # Accelerator configuration
        if self.cfg.accelerator_config:
            training_args_kwargs["accelerator_config"] = self.cfg.accelerator_config

    def _configure_optimizer(self, training_args_kwargs):
        """Configure optimizer settings."""
        custom_supported_optimizers = [opt.value for opt in CustomSupportedOptimizers]

        if self.cfg.optimizer in custom_supported_optimizers:
            # Use custom optimizer implementation
            self._configure_custom_optimizer(training_args_kwargs)
        else:
            # Use transformers' optimizer
            training_args_kwargs["optim"] = self.cfg.optimizer
            self._add_optimizer_args(training_args_kwargs)

        # Handle optimizer targeting specific modules
        if self.cfg.optim_target_modules:
            training_args_kwargs["optim_target_modules"] = self.cfg.optim_target_modules

        # Special case for anyprecision optimizer
        if self.cfg.optimizer == "adamw_anyprecision":
            if Path(self.cfg.torchdistx_path).exists():
                sys.path.append(self.cfg.torchdistx_path)
                importlib.import_module("torchdistx")

    def _configure_custom_optimizer(self, training_args_kwargs):
        """Configure custom optimizer settings."""
        # Common optimizer kwargs
        optimizer_kwargs = {
            "lr": training_args_kwargs.get("learning_rate"),
            "weight_decay": training_args_kwargs.get("weight_decay"),
        }

        # Add Adam-specific kwargs if available
        adam_kwargs = self._get_adam_kwargs(training_args_kwargs)

        # Get optimizer class and update kwargs based on optimizer type
        optimizer_cls = self._get_optimizer_class(
            training_args_kwargs, optimizer_kwargs, adam_kwargs
        )

        # Add any additional optimizer args from config
        self._update_optimizer_kwargs_from_config(optimizer_kwargs)

        training_args_kwargs["optimizer_cls_and_kwargs"] = (
            optimizer_cls,
            optimizer_kwargs,
        )

    def _get_adam_kwargs(self, training_args_kwargs):
        """Get Adam-specific kwargs if available."""
        adam_kwargs = {}
        if training_args_kwargs.get("adam_beta1") and training_args_kwargs.get(
            "adam_beta2"
        ):
            adam_kwargs["betas"] = (
                training_args_kwargs.get("adam_beta1"),
                training_args_kwargs.get("adam_beta2"),
            )
        if training_args_kwargs.get("adam_epsilon"):
            adam_kwargs["eps"] = training_args_kwargs.get("adam_epsilon")
        return adam_kwargs

    def _get_optimizer_class(self, training_args_kwargs, optimizer_kwargs, adam_kwargs):
        """Get optimizer class based on configuration."""
        if self.cfg.optimizer == "muon":
            from axolotl.contribs.mit.muon import MuonOptimizerFactory   # pylint: disable=no-name-in-module

            optimizer_cls = MuonOptimizerFactory
            optimizer_kwargs.update(adam_kwargs)
        elif self.cfg.optimizer == "optimi_adamw":
            from optimi import AdamW

            optimizer_kwargs["foreach"] = False
            optimizer_cls = AdamW
            optimizer_kwargs.update(adam_kwargs)
        elif self.cfg.optimizer == "ao_adamw_4bit":
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
        elif self.cfg.optimizer == "came_pytorch":
            from came_pytorch import CAME

            optimizer_cls = CAME

            beta1 = training_args_kwargs.get("adam_beta1", 0.9)
            beta2 = training_args_kwargs.get("adam_beta2", 0.999)
            beta3 = training_args_kwargs.get("adam_beta2", 0.9999)
            eps1 = training_args_kwargs.get("adam_epsilon", 1e-30)
            eps2 = training_args_kwargs.get("adam_epsilon2", 1e-16)

            adam_kwargs["betas"] = (beta1, beta2, beta3)
            adam_kwargs["eps"] = (eps1, eps2)
            optimizer_kwargs.update(adam_kwargs)
        else:
            # Default case or unsupported optimizer
            optimizer_cls = None

        return optimizer_cls

    def _update_optimizer_kwargs_from_config(self, optimizer_kwargs):
        """Update optimizer kwargs from config."""
        if self.cfg.optim_args:
            if isinstance(self.cfg.optim_args, dict):
                optimizer_kwargs.update(self.cfg.optim_args)
            else:
                # Parse string format "key1=value1,key2=value2"
                for mapping in self.cfg.optim_args.replace(" ", "").split(","):
                    key, value = mapping.split("=")
                    optimizer_kwargs[key] = value

    def _add_optimizer_args(self, training_args_kwargs):
        """Add optimizer arguments if available."""
        if self.cfg.optim_args:
            if isinstance(self.cfg.optim_args, dict):
                optim_args = ",".join(
                    [f"{key}={value}" for key, value in self.cfg.optim_args.items()]
                )
            else:
                optim_args = self.cfg.optim_args
            training_args_kwargs["optim_args"] = optim_args

    def _get_training_args_cls(self):
        """Get the appropriate training arguments class."""
        if self.cfg.reward_model:
            return AxolotlRewardConfig
        if self.cfg.process_reward_model:
            return AxolotlPRMConfig
        return AxolotlTrainingArguments

    def _prepare_data_collator_kwargs(self):
        """Prepare data collator kwargs."""
        data_collator_kwargs = {"padding": True}  # True/"longest" is the default

        if self.cfg.pad_to_sequence_len:
            data_collator_kwargs["pad_to_multiple_of"] = 64 * math.ceil(
                self.cfg.sequence_len / 64
            )
        else:
            data_collator_kwargs["pad_to_multiple_of"] = 64

        if self.cfg.reward_model:
            data_collator_kwargs["max_length"] = self.cfg.sequence_len

        return data_collator_kwargs

    def _prepare_trainer_kwargs(self, trainer_cls, data_collator_kwargs, training_args):
        """Prepare trainer kwargs."""
        trainer_kwargs = {}

        # Handle special data collators for evaluation
        if eval_data_collator := self.build_collator(
            training_args, is_eval=True, **data_collator_kwargs
        ):
            if not (self.cfg.reward_model or self.cfg.process_reward_model):
                trainer_kwargs["eval_data_collator"] = eval_data_collator

        # Add bench data collator if needed
        if not (self.cfg.reward_model or self.cfg.process_reward_model):
            trainer_kwargs["bench_data_collator"] = transformers.DataCollatorForSeq2Seq(
                self.tokenizer,
                return_tensors="pt",
                **data_collator_kwargs,
            )

        # Add tokenizer or processing class
        sig = inspect.signature(trainer_cls)
        if "processing_class" in sig.parameters.keys():
            trainer_kwargs["processing_class"] = self.tokenizer
        else:
            trainer_kwargs["tokenizer"] = self.tokenizer

        # Add dataset tags if available
        if (
            not (trainer_cls in [AxolotlRewardTrainer, AxolotlPRMTrainer])
            and self.cfg.datasets is not None
        ):
            trainer_kwargs["dataset_tags"] = [
                d["path"] for d in self.cfg.datasets if not Path(d["path"]).is_dir()
            ]

        return trainer_kwargs

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
            V2BatchSamplerDataCollatorForSeq2Seq
            | BatchSamplerDataCollatorForSeq2Seq
            | DataCollatorForSeq2Seq
            | DataCollatorWithFlattening
            | RewardDataCollatorWithPadding
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

        return collator(
            *collator_args,
            **kwargs,
        )
