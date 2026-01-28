"""Builder for causal trainers"""

import inspect
import math
import os
from pathlib import Path
from typing import Type, Union

import transformers
from transformers import (
    DataCollatorWithFlattening,
    EarlyStoppingCallback,
    Trainer,
)
from trl.trainer.reward_trainer import DataCollatorForPreference

from axolotl.core.builders.base import TrainerBuilderBase
from axolotl.core.trainers import (
    AxolotlMambaTrainer,
    AxolotlPRMTrainer,
    AxolotlRewardTrainer,
    AxolotlTrainer,
)
from axolotl.integrations.base import PluginManager
from axolotl.monkeypatch.multipack import SUPPORTED_MULTIPACK_MODEL_TYPES
from axolotl.monkeypatch.relora import ReLoRACallback
from axolotl.processing_strategies import get_processing_strategy
from axolotl.utils import is_comet_available, is_mlflow_available
from axolotl.utils.callbacks import (
    LossWatchDogCallback,
    bench_eval_callback_factory,
    causal_lm_bench_eval_callback_factory,
    colab_inference_post_train_callback,
    log_prediction_callback_factory,
)
from axolotl.utils.callbacks.lisa import lisa_callback_factory
from axolotl.utils.callbacks.qat import QATCallback
from axolotl.utils.callbacks.tokens_per_second import TokensPerSecondCallback
from axolotl.utils.chat_templates import get_chat_template_from_config
from axolotl.utils.collators import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    MambaDataCollator,
    V2BatchSamplerDataCollatorForSeq2Seq,
)
from axolotl.utils.collators.mm_chat import MultiModalChatDataCollator
from axolotl.utils.import_helper import get_cls_from_module_str
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class HFCausalTrainerBuilder(TrainerBuilderBase):
    """
    Build the HuggingFace training args/trainer for causal models and reward modeling
    using TRL.
    """

    def get_callbacks(self):
        callbacks = super().get_callbacks()

        if self.cfg.relora:
            callbacks.append(ReLoRACallback(self.cfg))

        # TODO: check if can move to base class
        if self.cfg.loss_watchdog_threshold is not None:
            callbacks.append(LossWatchDogCallback(self.cfg))

        if self.cfg.qat:
            callbacks.append(QATCallback(self.cfg.qat))

        if self.cfg.include_tkps:
            callbacks.append(
                TokensPerSecondCallback(
                    self.cfg.tensor_parallel_size,
                    self.cfg.context_parallel_size,
                    resume_from_checkpoint=self.cfg.resume_from_checkpoint,
                )
            )
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
        """
        Gets the trainer class for the given configuration.
        """
        if self.cfg.plugins:
            plugin_manager = PluginManager.get_instance()
            trainer_cls = plugin_manager.get_trainer_cls(self.cfg)
            if trainer_cls:
                return trainer_cls
        if self.cfg.model_config_type == "mamba":
            return AxolotlMambaTrainer
        if self.cfg.reward_model:
            return AxolotlRewardTrainer
        if self.cfg.process_reward_model:
            return AxolotlPRMTrainer

        if self.cfg.trainer_cls:
            # override the trainer cls
            try:
                trainer_cls = get_cls_from_module_str(self.cfg.trainer_cls)
                LOG.debug(f"Using custom trainer class: {self.cfg.trainer_cls}")
                return trainer_cls
            except (ImportError, AttributeError, ValueError) as e:
                raise ValueError(
                    f"Failed to load custom trainer class '{self.cfg.trainer_cls}': {e}"
                ) from e

        return AxolotlTrainer

    def build(self, total_num_steps):
        from axolotl.core.training_args import (
            AxolotlPRMConfig,
            AxolotlRewardConfig,
            AxolotlTrainingArguments,
        )

        training_arguments_kwargs, trainer_kwargs = self._set_base_training_args(
            total_num_steps
        )
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
        training_arguments_kwargs["sample_packing_drop_attention_mask"] = bool(
            self.cfg.flash_attention
            or self.cfg.xformers_attention
            or self.cfg.flex_attention
        )
        training_arguments_kwargs["multipack_real_batches"] = (
            self.cfg.multipack_real_batches
            if self.cfg.multipack_real_batches is not None
            else not (
                self.cfg.flash_attention
                or self.cfg.flex_attention
                or self.cfg.xformers_attention
            )
        )
        training_arguments_kwargs["eval_sample_packing"] = bool(
            self.cfg.eval_sample_packing
        )
        if self.cfg.sample_packing_sequentially is not None:
            training_arguments_kwargs["sample_packing_sequentially"] = (
                self.cfg.sample_packing_sequentially
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

        if self.cfg.relora and self.cfg.jagged_restart_steps:
            if self.cfg.relora_prune_ratio:
                training_arguments_kwargs["relora_prune_ratio"] = (
                    self.cfg.relora_prune_ratio
                )

        if self.cfg.jagged_restart_steps:
            training_arguments_kwargs["jagged_restart_steps"] = (
                self.cfg.jagged_restart_steps
            )
            if self.cfg.jagged_restart_warmup_steps:
                training_arguments_kwargs["jagged_restart_warmup_steps"] = (
                    self.cfg.jagged_restart_warmup_steps
                )
            if self.cfg.jagged_restart_anneal_steps:
                training_arguments_kwargs["jagged_restart_anneal_steps"] = (
                    self.cfg.jagged_restart_anneal_steps
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

        if self.cfg.image_size:
            training_arguments_kwargs["image_size"] = self.cfg.image_size
        if self.cfg.image_resize_algorithm:
            training_arguments_kwargs["image_resize_algorithm"] = (
                self.cfg.image_resize_algorithm
            )

        if self.cfg.plugins:
            plugin_manager = PluginManager.get_instance()
            plugin_training_args = plugin_manager.get_training_args(self.cfg)
            if plugin_training_args:
                training_arguments_kwargs.update(plugin_training_args)

        if self.cfg.reward_model:
            training_args_cls = AxolotlRewardConfig
            if self.cfg.center_rewards_coefficient is not None:
                training_arguments_kwargs["center_rewards_coefficient"] = (
                    self.cfg.center_rewards_coefficient
                )
        elif self.cfg.process_reward_model:
            training_args_cls = AxolotlPRMConfig
        else:
            training_args_cls = AxolotlTrainingArguments
        training_args = training_args_cls(
            **training_arguments_kwargs,
        )
        training_args = self.hook_post_create_training_args(training_args)

        # unset run_name so wandb sets up experiment names
        if self.cfg.use_wandb and training_args.run_name == training_args.output_dir:
            training_args.run_name = None

        data_collator_kwargs = {
            "padding": True,  # True/"longest" is the default
        }
        multiple = 64
        if self.cfg.pad_to_sequence_len:
            data_collator_kwargs["pad_to_multiple_of"] = multiple * math.ceil(
                self.cfg.sequence_len / multiple
            )
        elif self.cfg.pad_to_sequence_len is None:
            # A100 is best at 64, while others at 8. Let's use the larger so we don't have to check
            # https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html
            data_collator_kwargs["pad_to_multiple_of"] = multiple

        if self.cfg.use_eaft:
            from functools import partial

            from axolotl.monkeypatch.loss.eaft import eaft_loss

            configured_eaft_loss = partial(
                eaft_loss,
                alpha=self.cfg.eaft_alpha if self.cfg.eaft_alpha is not None else 1.0,
                k=self.cfg.eaft_k if self.cfg.eaft_k is not None else 20,
            )
            trainer_kwargs["compute_loss_func"] = configured_eaft_loss

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
        if "processing_class" in sig.parameters or issubclass(trainer_cls, Trainer):
            trainer_kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in sig.parameters:
            trainer_kwargs["tokenizer"] = self.tokenizer

        if (
            trainer_cls not in [AxolotlRewardTrainer, AxolotlPRMTrainer]
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
        # if the trainer has the `axolotl_cfg` property, set it
        if hasattr(trainer, "axolotl_cfg"):
            trainer.axolotl_cfg = self.cfg
        for callback in self.get_post_trainer_create_callbacks(trainer):
            trainer.add_callback(callback)

        if self.cfg.deepspeed and self.cfg.sample_packing:
            trainer.accelerator.state.deepspeed_plugin.deepspeed_config[
                "train_micro_batch_size_per_gpu"
            ] = self.cfg.micro_batch_size

        return trainer

    def build_collator(
        self,
        training_args,  # type: "AxolotlTrainingArguments"  # type: ignore
        is_eval=False,
        **kwargs,
    ):
        if training_args.pretraining:
            if (
                self.cfg.pretraining_sample_concatenation is False
                or self.cfg.micro_batch_size > 1
            ):
                return DataCollatorForSeq2Seq(self.tokenizer, **kwargs)
            if not (self.cfg.sample_packing and self.cfg.pretrain_multipack_attn) or (
                self.cfg.micro_batch_size == 1 and is_eval is False
            ):
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
                DataCollatorForPreference,
            ]
        ]
        collator_args = [self.tokenizer]

        collator_cls_and_kwargs = None
        if self.cfg.plugins:
            plugin_manager = PluginManager.get_instance()
            collator_cls_and_kwargs = plugin_manager.get_collator_cls_and_kwargs(
                self.cfg, is_eval=is_eval
            )

        if collator_cls_and_kwargs:
            collator = collator_cls_and_kwargs[0]
            if kwargs and isinstance(kwargs, dict):
                kwargs.update(collator_cls_and_kwargs[1])
        elif self.cfg.reward_model:
            collator = DataCollatorForPreference
            tokenizer = collator_args.pop(0)
            kwargs["pad_token_id"] = tokenizer.pad_token_id
            kwargs.pop("padding")
        elif use_batch_sampler_collator:
            # Use V2BatchSamplerDataCollatorForSeq2Seq for flex attention,
            # supported multipack models, or non-flash-attention llama
            if (
                self.cfg.flex_attention
                or self.cfg.model_config_type in SUPPORTED_MULTIPACK_MODEL_TYPES
                or (
                    self.cfg.model_config_type in ["llama"]
                    and self.cfg.flash_attention is not True
                )
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
            else:
                collator = DataCollatorForSeq2Seq

        kwargs["return_tensors"] = "pt"

        return collator(
            *collator_args,
            **kwargs,
        )
