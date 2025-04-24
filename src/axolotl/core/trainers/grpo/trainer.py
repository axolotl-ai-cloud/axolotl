"""Axolotl GRPO trainer"""

# pylint: disable=too-many-lines,duplicate-code

from contextlib import nullcontext

import datasets
import torch
import torch.distributed as dist
from accelerate.utils import (
    is_peft_model,
)
from datasets import Dataset, IterableDataset
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Sampler,
)
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from transformers.utils import is_peft_available
from trl import GRPOTrainer
from trl.extras.profiling import profiling_decorator
from trl.import_utils import is_deepspeed_available
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import RewardFunc

from axolotl.core.trainers.grpo.sampler import SequenceParallelRepeatRandomSampler
from axolotl.core.trainers.mixins import RngLoaderMixin, SchedulerMixin
from axolotl.monkeypatch.attention.ring_attn.patch import get_ring_attn_group

if is_peft_available():
    # pylint: disable=unused-import
    from peft import PeftConfig

if is_deepspeed_available():
    import deepspeed


class AxolotlGRPOTrainer(RngLoaderMixin, SchedulerMixin, GRPOTrainer):
    """Extend the base GRPOTrainer for axolotl helpers"""

    _tag_names = ["trl", "grpo", "axolotl"]

    def __init__(
        self,
        model: str | PreTrainedModel,
        reward_funcs: RewardFunc | list[RewardFunc],
        args: GRPOConfig | None = None,
        train_dataset: Dataset | IterableDataset | None = None,
        eval_dataset: (
            Dataset | IterableDataset | dict[str, Dataset | IterableDataset] | None
        ) = None,
        processing_class: PreTrainedTokenizerBase | None = None,
        reward_processing_classes: (
            PreTrainedTokenizerBase | list[PreTrainedTokenizerBase] | None
        ) = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple[
            torch.optim.Optimizer | None, torch.optim.lr_scheduler.LambdaLR | None
        ] = (None, None),
        peft_config: "PeftConfig | None" = None,
    ):
        # First call the superclass constructor with all arguments
        super().__init__(
            model=model,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            reward_processing_classes=reward_processing_classes,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
        )

        # Now execute your custom logic
        # Get number of SP groups (number of processes divided by SP degree)
        num_processes = self.accelerator.num_processes
        num_sp_groups = num_processes // self.args.sequence_parallel_degree

        # Calculate batch size per SP group (not per process)
        sp_group_batch_size = self.args.per_device_train_batch_size * num_sp_groups
        possible_values = [
            n_gen
            for n_gen in range(2, sp_group_batch_size + 1)
            if (sp_group_batch_size) % n_gen == 0
        ]

        if self.num_generations not in possible_values:
            raise ValueError(
                f"The batch size per SP group ({num_sp_groups} x "
                f"{self.args.per_device_train_batch_size}) must be evenly divisible by "
                f"the number of generations per prompt ({self.num_generations}). Given "
                "the current configuration, the valid values for the number of "
                f"generations are: {possible_values}."
            )

        if self.args.eval_strategy != "no":
            # If sequence parallelism is enabled, calculate batch size per SP group
            sp_group_eval_batch_size = args.per_device_eval_batch_size * num_sp_groups  # type: ignore[union-attr]
            possible_values = [
                n_gen
                for n_gen in range(2, sp_group_eval_batch_size + 1)
                if (sp_group_eval_batch_size) % n_gen == 0
            ]

            if self.num_generations not in possible_values:
                raise ValueError(
                    f"With sequence parallelism (degree {self.args.sequence_parallel_degree}), "
                    f"the eval batch size per SP group ({num_sp_groups} x {self.args.per_device_eval_batch_size}) "
                    f"must be evenly divisible by the number of generations per prompt "
                    f"({self.num_generations}). Given the current eval batch size, "
                    f"the valid values for the number of generations are: {possible_values}."
                )

        # Initialize the SP group
        self.sp_group = get_ring_attn_group()
        self.local_rank = dist.get_rank(group=self.sp_group)
        self.local_world_size = dist.get_world_size(group=self.sp_group)

    def _get_train_sampler(self) -> Sampler:
        # Get distributed training info
        world_size = dist.get_world_size()
        rank = dist.get_rank()

        effective_batch_size = (
            self.args.per_device_train_batch_size
            * world_size
            * self.args.gradient_accumulation_steps
        )

        return SequenceParallelRepeatRandomSampler(
            dataset=self.train_dataset,
            mini_repeat_count=self.num_generations,
            world_size=world_size,
            rank=rank,
            batch_size=effective_batch_size
            // self.num_generations
            // self.args.sequence_parallel_degree,
            repeat_count=self.num_iterations,
            sequence_parallel_degree=self.args.sequence_parallel_degree,
            shuffle=True,
            seed=self.args.seed,
            drop_last=True,
        )

    def _create_dataloader_params(self, is_eval=False, custom_batch_size=None):
        """Create common dataloader parameters for train or eval."""
        batch_size = custom_batch_size or (
            self.args.eval_batch_size if is_eval else self._train_batch_size
        )

        params = {
            "batch_size": batch_size,
            "collate_fn": self.data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
        }

        # Add persistent workers only for training
        if not is_eval and hasattr(self.args, "dataloader_persistent_workers"):
            params["persistent_workers"] = self.args.dataloader_persistent_workers

        # Add prefetch factor if specified
        if self.args.dataloader_prefetch_factor:
            params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return params

    def _prepare_dataloader(
        self, dataset, sampler, is_eval=False, custom_batch_size=None
    ):
        """Prepare a dataloader with the given dataset and sampler."""
        # Get base parameters
        dataloader_params = self._create_dataloader_params(is_eval, custom_batch_size)

        # Add sampler configuration
        if not isinstance(dataset, torch.utils.data.IterableDataset):
            if isinstance(sampler, BatchSampler):
                # batch_size and batch_sampler are mutually exclusive
                dataloader_params["batch_sampler"] = sampler
                del dataloader_params["batch_size"]
            else:
                dataloader_params["sampler"] = sampler
                dataloader_params["drop_last"] = self.args.dataloader_drop_last

            if not is_eval:
                dataloader_params["worker_init_fn"] = seed_worker

        # Create the dataloader
        dataloader = DataLoader(dataset, **dataloader_params)

        if self.args.sample_packing and (
            (not is_eval and not self.args.pretraining)
            or (is_eval and self.args.eval_sample_packing is not False)
        ):
            self.accelerator.even_batches = False

        # Return unprepared dataloader if using sequence parallelism
        # TODO(djsaunde): We might be able to use `accelerate`'s dataloader preparation
        # if we use `dispatch_batches` and `slice_fn_for_dispatch` properly (i.e.,
        # slice each batch along the sequence dimension).
        if self.args.sequence_parallel_degree > 1:
            return dataloader

        # Otherwise prepare with accelerator
        return self.accelerator.prepare_data_loader(dataloader)

    def get_train_dataloader(self) -> DataLoader:
        """Get dataloader for training"""
        train_dataset = self.train_dataset
        # pylint: disable=access-member-before-definition
        data_collator = self.data_collator  # type: ignore

        # Initialize SP group attributes if sequence parallelism is enabled
        if self.args.sequence_parallel_degree > 1:
            self.sp_group = get_ring_attn_group()
            self.local_rank = dist.get_rank(group=self.sp_group)
            self.local_world_size = dist.get_world_size(group=self.sp_group)

        # Handle dataset preprocessing
        if isinstance(train_dataset, datasets.Dataset):
            # Add debug print before any modifications
            if self.args.sample_packing and not self.args.pretraining:
                train_dataset = train_dataset.remove_columns(["length"])
            if not self.args.sample_packing or self.args.pretraining:
                train_dataset = self._remove_unused_columns(
                    train_dataset, description="training"
                )
        else:
            self.data_collator = self._get_collator_with_removed_columns(  # pylint: disable=attribute-defined-outside-init
                data_collator,
                description="training",
            )

        # Get sampler and create dataloader
        sampler = self._get_train_sampler()
        dataloader = self._prepare_dataloader(train_dataset, sampler, is_eval=False)

        return dataloader

    @profiling_decorator
    def _move_model_to_vllm(self):
        # For DeepSpeed ZeRO-3, we need to gather all parameters before operations
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = (
            deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext
        )

        if is_peft_model(self.model):
            # With PEFT and DeepSpeed ZeRO Stage 3, we must gather the full model at once before merging, as merging
            # adapters in a sharded manner is not supported.
            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                # Update vLLM weights while parameters are gathered
                for name, param in self.model.named_parameters():
                    # When using PEFT, we need to recover the original parameter name and discard some parameters
                    name = (
                        name.removeprefix("base_model.model.")
                        .removeprefix("base_model.model.")
                        .replace(".base_layer", "")
                    )
                    if self.model.prefix in name:
                        continue
                    # When module to save, remove its prefix and discard the original module
                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")

                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

                # Unmerge adapters while parameters are still gathered
                self.model.unmerge_adapter()
                # Parameters will automatically be repartitioned when exiting the context
        else:
            # For non-PEFT models, simply gather and update each parameter individually.
            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.accelerator.is_main_process:
                        self.vllm_client.update_named_param(name, param.data)

        # Reset cache on main process
        if self.accelerator.is_main_process:
            self.vllm_client.reset_prefix_cache()
