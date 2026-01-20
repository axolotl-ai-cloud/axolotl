"""Builder for RLHF trainers"""

import inspect
from pathlib import Path

from axolotl.core.builders.base import TrainerBuilderBase
from axolotl.core.trainers import (
    AxolotlCPOTrainer,
    AxolotlKTOTrainer,
    AxolotlORPOTrainer,
)
from axolotl.core.trainers.dpo import DPOStrategy
from axolotl.core.trainers.dpo.args import AxolotlDPOConfig
from axolotl.core.trainers.grpo import GRPOStrategy
from axolotl.integrations.base import PluginManager
from axolotl.loaders.utils import ensure_dtype
from axolotl.utils.callbacks.qat import QATCallback
from axolotl.utils.import_helper import get_cls_from_module_str
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.enums import RLType

LOG = get_logger(__name__)


class HFRLTrainerBuilder(TrainerBuilderBase):
    """Trainer factory class for TRL-based RLHF trainers (e.g. DPO)"""

    def get_callbacks(self):
        callbacks = super().get_callbacks()

        if self.cfg.qat:
            callbacks.append(QATCallback(self.cfg.qat))

        return callbacks

    def get_post_trainer_create_callbacks(self, trainer):
        callbacks = super().get_post_trainer_create_callbacks(trainer=trainer)
        return callbacks

    def _get_trainer_cls(self, trainer_kwargs: dict):
        """
        Returns trainer_cls and trainer_cls_args
        """
        if self.cfg.plugins:
            plugin_manager = PluginManager.get_instance()
            trainer_cls = plugin_manager.get_trainer_cls(self.cfg)
            trainer_cls_args = []  # type: ignore

            if trainer_cls is not None:
                return trainer_cls, trainer_cls_args

        trainer_cls = None
        trainer_cls_args = [self.model]

        if self.cfg.rl in {RLType.GRPO, RLType.GDPO}:
            trainer_cls = GRPOStrategy.get_trainer_class(
                sequence_parallel=self.cfg.context_parallel_size > 1
            )
            trainer_cls_args.extend(GRPOStrategy.set_trainer_args(self.cfg))
            trainer_kwargs.update(GRPOStrategy.set_trainer_kwargs(self.cfg))

        elif self.cfg.rl in [RLType.DPO, RLType.IPO]:
            trainer_cls = DPOStrategy.get_trainer_class()
            trainer_cls_args.append(self.model_ref)

        elif self.cfg.rl is RLType.ORPO:
            trainer_cls = AxolotlORPOTrainer
        elif self.cfg.rl is RLType.KTO:
            trainer_cls = AxolotlKTOTrainer
        elif self.cfg.rl is RLType.SIMPO:
            trainer_cls = AxolotlCPOTrainer
        else:
            raise ValueError(f"Unsupported RL: {self.cfg.rl}")

        if self.cfg.trainer_cls:
            # override the trainer cls
            try:
                trainer_cls = get_cls_from_module_str(self.cfg.trainer_cls)
                LOG.debug(f"Using custom trainer class: {self.cfg.trainer_cls}")
            except (ImportError, AttributeError, ValueError) as e:
                raise ValueError(
                    f"Failed to load custom trainer class '{self.cfg.trainer_cls}': {e}"
                ) from e

        return trainer_cls, trainer_cls_args

    def _build_training_arguments(self, total_num_steps):
        """
        Returns training_args and trainer_kwargs
        """
        from axolotl.core.training_args import (
            AxolotlCPOConfig,
            AxolotlKTOConfig,
            AxolotlORPOConfig,
        )

        training_args_kwargs, trainer_kwargs = self._set_base_training_args(
            total_num_steps=total_num_steps
        )

        if self.cfg.remove_unused_columns is not None:
            training_args_kwargs["remove_unused_columns"] = (
                self.cfg.remove_unused_columns
            )
        else:
            training_args_kwargs["remove_unused_columns"] = False

        if self.cfg.trl and self.cfg.trl.beta is not None:
            training_args_kwargs["beta"] = self.cfg.trl.beta
        elif self.cfg.rl_beta is not None:
            training_args_kwargs["beta"] = self.cfg.rl_beta
        elif self.cfg.orpo_alpha is not None:
            # trl does some odd mapping of alpha to beta to reuse the beta parameter ???
            training_args_kwargs["beta"] = self.cfg.orpo_alpha

        if self.cfg.rpo_alpha is not None:
            training_args_kwargs["rpo_alpha"] = self.cfg.rpo_alpha

        if self.cfg.use_wandb:
            training_args_kwargs["run_name"] = self.cfg.wandb_name

        if self.cfg.max_prompt_len:
            training_args_kwargs["max_prompt_length"] = self.cfg.max_prompt_len
        else:
            training_args_kwargs["max_prompt_length"] = self.cfg.sequence_len

        training_args_cls = None
        blocklist_args_kwargs = []
        if self.cfg.rl is RLType.SIMPO:
            training_args_cls = AxolotlCPOConfig
            training_args_kwargs["loss_type"] = "simpo"
            training_args_kwargs["simpo_gamma"] = self.cfg.simpo_gamma
            if self.cfg.cpo_alpha is not None:
                training_args_kwargs["cpo_alpha"] = self.cfg.cpo_alpha

            # Handle when max_prompt_length == max_length from defaults
            # CPOTrainer requires strictly less than
            if (
                training_args_kwargs["max_prompt_length"]
                == training_args_kwargs["max_length"]
            ):
                training_args_kwargs["max_prompt_length"] -= 1

        elif self.cfg.rl is RLType.ORPO:
            training_args_cls = AxolotlORPOConfig

        elif self.cfg.rl is RLType.KTO:
            training_args_cls = AxolotlKTOConfig
            # KTOConfig in TRL >= 0.27.0 no longer accepts max_prompt_length
            blocklist_args_kwargs = ["max_prompt_length"]

            training_args_kwargs["desirable_weight"] = (
                self.cfg.kto_desirable_weight or 1.0
            )
            training_args_kwargs["undesirable_weight"] = (
                self.cfg.kto_undesirable_weight or 1.0
            )

        elif self.cfg.rl in {RLType.GRPO, RLType.GDPO}:
            training_args_cls = GRPOStrategy.get_training_args_class()
            training_args_kwargs.update(GRPOStrategy.set_training_args_kwargs(self.cfg))
            blocklist_args_kwargs = GRPOStrategy.get_blocklist_args_kwargs()
            if self.cfg.rl is RLType.GDPO:
                training_args_kwargs.setdefault(
                    "multi_objective_aggregation", "normalize_then_sum"
                )

        elif self.cfg.rl in [RLType.DPO, RLType.IPO]:
            training_args_cls = AxolotlDPOConfig
            training_args_kwargs.update(DPOStrategy.set_training_args_kwargs(self.cfg))
        else:
            raise ValueError(f"Unsupported RL: {self.cfg.rl}")

        for blocklist_key in blocklist_args_kwargs:
            if blocklist_key in training_args_kwargs:
                del training_args_kwargs[blocklist_key]

        if self.cfg.plugins:
            plugin_manager = PluginManager.get_instance()
            plugin_training_args = plugin_manager.get_training_args(self.cfg)
            if plugin_training_args:
                training_args_kwargs.update(plugin_training_args)

        training_args = training_args_cls(
            logging_first_step=True,
            **training_args_kwargs,
        )

        # unset run_name so wandb sets up experiment names
        if self.cfg.use_wandb and training_args.run_name == training_args.output_dir:
            training_args.run_name = None

        return training_args, trainer_kwargs

    def build(self, total_num_steps):
        training_args, trainer_kwargs = self._build_training_arguments(total_num_steps)

        if self.eval_dataset:
            trainer_kwargs["eval_dataset"] = self.eval_dataset
        if self.cfg.adapter and self.peft_config and self.cfg.rl is not RLType.GRPO:
            trainer_kwargs["peft_config"] = self.peft_config
        if self.cfg.precompute_ref_log_probs is not None:
            trainer_kwargs["precompute_ref_log_probs"] = (
                self.cfg.precompute_ref_log_probs
            )

        trainer_cls, trainer_cls_args = self._get_trainer_cls(trainer_kwargs)

        sig = inspect.signature(trainer_cls)
        if "tokenizer" in sig.parameters:
            trainer_kwargs["tokenizer"] = self.tokenizer
        else:
            trainer_kwargs["processing_class"] = self.tokenizer

        if self.cfg.datasets is not None and (
            trainer_cls is DPOStrategy.get_trainer_class()
        ):
            trainer_kwargs["dataset_tags"] = [
                d["path"] for d in self.cfg.datasets if not Path(d["path"]).is_dir()
            ]

        trainer_kwargs, trainer_cls = self.hook_pre_create_trainer(
            trainer_kwargs, trainer_cls
        )

        trainer = trainer_cls(
            *trainer_cls_args,
            args=training_args,
            train_dataset=self.train_dataset,
            callbacks=self.get_callbacks(),
            **trainer_kwargs,
        )
        if self.cfg.fsdp_config or self.cfg.fsdp:
            ensure_dtype(trainer.model, dtype=self.cfg.torch_dtype)
            if self.cfg.rl in [RLType.DPO, RLType.IPO] and trainer.ref_model:
                ensure_dtype(trainer.ref_model, dtype=self.cfg.torch_dtype)

        trainer = self.hook_post_create_trainer(trainer)
        for callback in self.get_post_trainer_create_callbacks(trainer):
            trainer.add_callback(callback)

        return trainer
