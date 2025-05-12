"""RL trainer / training args builder implementation"""

import inspect
from pathlib import Path

from axolotl.core.trainers.builders.base import TrainerBuilderBase
from axolotl.core.trainers.dpo import DPOStrategy
from axolotl.core.trainers.dpo.args import AxolotlDPOConfig
from axolotl.core.trainers.grpo import GRPOStrategy
from axolotl.core.trainers.trl import (
    AxolotlCPOTrainer,
    AxolotlKTOTrainer,
    AxolotlORPOTrainer,
)
from axolotl.core.training_args import (
    AxolotlCPOConfig,
    AxolotlKTOConfig,
    AxolotlORPOConfig,
)
from axolotl.utils.models import ensure_dtype


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

        training_args_kwargs["save_only_model"] = self.cfg.save_only_model

        if self.cfg.dataset_processes:
            training_args_kwargs["dataset_num_proc"] = self.cfg.dataset_processes

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

        # unset run_name so wandb sets up experiment names
        if self.cfg.use_wandb and training_args.run_name == training_args.output_dir:
            training_args.run_name = (  # pylint: disable=attribute-defined-outside-init
                None
            )

        return training_args

    def build(self, total_num_steps):
        """Build and return an RL trainer instance"""
        # Prepare RL-specific training args kwargs
        training_args_kwargs = {
            "per_device_train_batch_size": self.cfg.micro_batch_size,
            "max_steps": self.cfg.max_steps or total_num_steps or -1,
            "gradient_accumulation_steps": self.cfg.gradient_accumulation_steps,
            "learning_rate": self.cfg.learning_rate,
            "warmup_steps": self.cfg.warmup_steps,
            "logging_first_step": True,
            "logging_steps": 1,
            "output_dir": self.cfg.output_dir,
            "num_train_epochs": self.cfg.num_epochs,
        }

        # Handle dataset processes
        if self.cfg.dataset_processes:
            training_args_kwargs["dataset_num_proc"] = self.cfg.dataset_processes

        # Handle beta/alpha parameters for different RL algorithms
        if self.cfg.trl and self.cfg.trl.beta is not None:
            training_args_kwargs["beta"] = self.cfg.trl.beta
        elif self.cfg.rl_beta is not None:
            training_args_kwargs["beta"] = self.cfg.rl_beta
        elif self.cfg.orpo_alpha is not None:
            # trl does some odd mapping of alpha to beta to reuse the beta parameter
            training_args_kwargs["beta"] = self.cfg.orpo_alpha

        if self.cfg.rpo_alpha is not None:
            training_args_kwargs["rpo_alpha"] = self.cfg.rpo_alpha

        # Determine training args class and add RL-specific parameters
        training_args_cls = None
        blocklist_args_kwargs = []

        if self.cfg.rl == "simpo":
            training_args_cls = AxolotlCPOConfig
            training_args_kwargs["loss_type"] = "simpo"
            training_args_kwargs["simpo_gamma"] = self.cfg.simpo_gamma
            if self.cfg.cpo_alpha is not None:
                training_args_kwargs["cpo_alpha"] = self.cfg.cpo_alpha
        elif self.cfg.rl == "orpo":
            training_args_cls = AxolotlORPOConfig
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
            if self.cfg.max_prompt_len:
                training_args_kwargs["max_prompt_length"] = self.cfg.max_prompt_len
        elif self.cfg.rl == "grpo":
            training_args_cls = GRPOStrategy.get_training_args_class()
            training_args_kwargs.update(GRPOStrategy.set_training_args_kwargs(self.cfg))
            blocklist_args_kwargs = GRPOStrategy.get_blocklist_args_kwargs()
        else:  # Default to DPO
            training_args_cls = AxolotlDPOConfig
            if self.cfg.rl == "ipo":
                training_args_kwargs["loss_type"] = "ipo"
            training_args_kwargs["max_prompt_length"] = self.cfg.sequence_len
            training_args_kwargs["max_completion_length"] = None
            training_args_kwargs["generate_during_eval"] = self.cfg.use_wandb
            if self.cfg.dpo_use_weighting is not None:
                training_args_kwargs["use_weighting"] = self.cfg.dpo_use_weighting
            if self.cfg.dpo_use_logits_to_keep is not None:
                training_args_kwargs["use_logits_to_keep"] = (
                    self.cfg.dpo_use_logits_to_keep
                )

        # Remove any blocklisted arguments
        for blocklist_key in blocklist_args_kwargs:
            if blocklist_key in training_args_kwargs:
                del training_args_kwargs[blocklist_key]

        # Create training args using the base class method
        training_args = self.create_training_args(
            args_cls=training_args_cls,
            total_num_steps=total_num_steps,
            **training_args_kwargs,
        )

        # Prepare trainer kwargs
        trainer_kwargs = {}
        if self.cfg.rl == "ipo" and self.cfg.dpo_label_smoothing:
            trainer_kwargs["label_smoothing"] = self.cfg.dpo_label_smoothing
        if self.eval_dataset:
            trainer_kwargs["eval_dataset"] = self.eval_dataset
        if self.cfg.adapter and self.peft_config:
            trainer_kwargs["peft_config"] = self.peft_config
        if self.cfg.precompute_ref_log_probs is not None:
            trainer_kwargs["precompute_ref_log_probs"] = (
                self.cfg.precompute_ref_log_probs
            )

        # Determine trainer class and arguments
        if self.cfg.rl == "grpo":
            trainer_cls = GRPOStrategy.get_trainer_class()
            trainer_args = [self.model]
            trainer_args.extend(GRPOStrategy.set_trainer_args(self.cfg))
            trainer_kwargs.update(GRPOStrategy.set_trainer_kwargs(self.cfg))
        elif self.cfg.rl in ["dpo", "ipo"]:
            trainer_cls = DPOStrategy.get_trainer_class()
            trainer_args = [self.model, self.model_ref]
        elif self.cfg.rl == "orpo":
            trainer_cls = AxolotlORPOTrainer
            trainer_args = [self.model]
        elif self.cfg.rl in ["kto"]:
            trainer_cls = AxolotlKTOTrainer
            trainer_args = [self.model]
        elif self.cfg.rl in ["simpo"]:
            trainer_cls = AxolotlCPOTrainer
            trainer_args = [self.model]
        else:
            raise ValueError(f"Unsupported RL: {self.cfg.rl}")

        # Add tokenizer or processing class
        sig = inspect.signature(trainer_cls)
        if "tokenizer" in sig.parameters.keys():
            trainer_kwargs["tokenizer"] = self.tokenizer
        else:
            trainer_kwargs["processing_class"] = self.tokenizer

        # Add dataset tags if available
        if self.cfg.datasets is not None and (
            trainer_cls is DPOStrategy.get_trainer_class()
        ):
            trainer_kwargs["dataset_tags"] = [
                d["path"] for d in self.cfg.datasets if not Path(d["path"]).is_dir()
            ]

        # Create the trainer
        trainer = self.create_trainer(
            trainer_cls=trainer_cls,
            training_args=training_args,
            trainer_args=trainer_args,
            trainer_kwargs=trainer_kwargs,
        )

        # Handle FSDP specific settings
        if self.cfg.fsdp:
            ensure_dtype(trainer.model, dtype=self.cfg.torch_dtype)
            if (
                self.cfg.rl in ["dpo", "ipo"]
                and hasattr(trainer, "ref_model")
                and trainer.ref_model
            ):
                ensure_dtype(trainer.ref_model, dtype=self.cfg.torch_dtype)

        return trainer
