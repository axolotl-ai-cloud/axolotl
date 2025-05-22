"""Trainer builder for SFT trainers"""

import inspect
import logging
import math
import os
from pathlib import Path
from typing import Type, Union

import torch
import transformers
from transformers import (
    DataCollatorWithFlattening,
    EarlyStoppingCallback,
)
from trl.trainer.utils import RewardDataCollatorWithPadding

from axolotl.core.trainer_builder.base import TrainerBuilderBase
from axolotl.core.trainers import (
    AxolotlMambaTrainer,
    AxolotlPRMTrainer,
    AxolotlRewardTrainer,
    AxolotlTrainer,
    ReLoRATrainer,
)
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
from axolotl.utils.collators import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    MambaDataCollator,
    V2BatchSamplerDataCollatorForSeq2Seq,
)
from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator

try:
    import torch._dynamo  # pylint: disable=ungrouped-imports
except ImportError:
    pass

LOG = logging.getLogger(__name__)


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
        training_arguments_kwargs = self._set_base_training_args(total_num_steps)

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

        if self.cfg.torch_compile and getattr(torch, "_dynamo", None):
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
        training_arguments_kwargs["max_seq_length"] = self.cfg.sequence_len

        if self.cfg.auto_find_batch_size is not None:
            training_arguments_kwargs["auto_find_batch_size"] = (
                self.cfg.auto_find_batch_size
            )

        training_arguments_kwargs["eval_accumulation_steps"] = (
            self.cfg.gradient_accumulation_steps
        )
        training_arguments_kwargs["num_train_epochs"] = self.cfg.num_epochs

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

        trainer_kwargs = {}

        # Pop optimizer_cls_and_kwargs to trainer_kwargs
        if hasattr(training_arguments_kwargs, "optimizer_cls_and_kwargs"):
            trainer_kwargs["optimizer_cls_and_kwargs"] = getattr(
                training_arguments_kwargs, "optimizer_cls_and_kwargs"
            )
            # prevent duplication downstream
            delattr(training_arguments_kwargs, "optimizer_cls_and_kwargs")

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
        multiple = 64
        if self.cfg.pad_to_sequence_len:
            data_collator_kwargs["pad_to_multiple_of"] = multiple * math.ceil(
                self.cfg.sequence_len / multiple
            )
        else:
            # A100 is best at 64, while others at 8. Let's use the larger so we don't have to check
            # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
            data_collator_kwargs["pad_to_multiple_of"] = multiple

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
        if "processing_class" in sig.parameters:
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
