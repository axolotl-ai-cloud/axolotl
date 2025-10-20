"""Axolotl GRPO trainers (with and without sequence parallelism handling)"""

import warnings
from functools import partial
from typing import Any

import datasets
import torch
import torch.distributed as dist
import torch.utils.data
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_available,
)
from datasets import Dataset, IterableDataset
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Sampler,
)
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
from transformers.trainer_utils import seed_worker
from trl import GRPOTrainer
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.extras.profiling import profiling_context
from trl.models import unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.grpo_trainer import RewardFunc, nanstd
from trl.trainer.utils import pad

from axolotl.core.trainers.grpo.sampler import SequenceParallelRepeatRandomSampler
from axolotl.core.trainers.mixins import (
    DistributedParallelMixin,
    RngLoaderMixin,
    SchedulerMixin,
)
from axolotl.core.trainers.mixins.optimizer import OptimizerInitMixin, OptimizerMixin
from axolotl.monkeypatch.ring_attn import get_ring_attn_group

if is_peft_available():
    from peft import PeftConfig


class AxolotlGRPOTrainer(
    RngLoaderMixin,
    SchedulerMixin,
    OptimizerMixin,
    OptimizerInitMixin,
    DistributedParallelMixin,
    GRPOTrainer,
):
    """Extend the base GRPOTrainer for axolotl helpers"""

    _tag_names = ["trl", "grpo", "axolotl"]


class AxolotlGRPOSequenceParallelTrainer(AxolotlGRPOTrainer):
    """Extend the base GRPOTrainer for sequence parallelism handling"""

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
        optimizer_cls_and_kwargs: tuple[type, dict] | None = None,
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
            optimizer_cls_and_kwargs=optimizer_cls_and_kwargs,
        )

        # Get number of SP groups (number of processes divided by SP degree)
        num_processes = self.accelerator.num_processes
        num_sp_groups = num_processes // self.args.context_parallel_size

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
                    f"With sequence parallelism (degree {self.args.context_parallel_size}), "
                    f"the eval batch size per SP group ({num_sp_groups} x {self.args.per_device_eval_batch_size}) "
                    f"must be evenly divisible by the number of generations per prompt "
                    f"({self.num_generations}). Given the current eval batch size, "
                    f"the valid values for the number of generations are: {possible_values}."
                )

        self.sp_group = None
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = 0
        self.local_world_size = 1

    def train(self, *args, **kwargs):
        # Initialize the SP group
        self.sp_group = get_ring_attn_group()
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = dist.get_rank(group=self.sp_group)
        self.local_world_size = dist.get_world_size(group=self.sp_group)

        return super().train(*args, **kwargs)

    def _get_train_sampler(self) -> Sampler:
        effective_batch_size = (
            self.args.per_device_train_batch_size
            * self.world_size
            * self.args.gradient_accumulation_steps
        )

        return SequenceParallelRepeatRandomSampler(
            dataset=self.train_dataset,
            mini_repeat_count=self.num_generations,
            world_size=self.world_size,
            rank=self.rank,
            batch_size=effective_batch_size
            // self.num_generations
            // self.args.context_parallel_size,
            repeat_count=self.num_iterations * self.args.gradient_accumulation_steps,
            context_parallel_size=self.args.context_parallel_size,
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
                dataloader_params["worker_init_fn"] = partial(
                    seed_worker,
                    num_workers=self.args.dataloader_num_workers,
                    rank=self.args.process_index,
                )

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
        if self.args.context_parallel_size > 1:
            return dataloader

        # Otherwise prepare with accelerator
        return self.accelerator.prepare_data_loader(dataloader)

    def get_train_dataloader(self) -> DataLoader:
        """Get dataloader for training"""
        train_dataset = self.train_dataset

        data_collator = self.data_collator  # type: ignore

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
            self.data_collator = self._get_collator_with_removed_columns(
                data_collator,
                description="training",
            )

        # Get sampler and create dataloader
        sampler = self._get_train_sampler()
        dataloader = self._prepare_dataloader(train_dataset, sampler, is_eval=False)

        return dataloader

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
        mode = "eval" if self.control.should_evaluate else "train"

        prompts = [x["prompt"] for x in inputs]
        prompts_text = [
            maybe_apply_chat_template(example, self.processing_class)["prompt"]
            for example in inputs
        ]
        prompt_inputs = self.processing_class(
            text=prompts_text,
            return_tensors="pt",
            padding=True,
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        # Generate completions using either vLLM or regular generation
        if self.args.use_vllm:
            # First, have main process load weights if needed

            if self.state.global_step != self._last_loaded_step:  # type: ignore[has-type]
                self._move_model_to_vllm()

                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                if self.args.context_parallel_size > 1:
                    # Calculate sequence parallel group information
                    world_size = self.accelerator.num_processes
                    context_parallel_size = self.args.context_parallel_size
                    num_sp_groups = world_size // context_parallel_size

                    # Since processes in the same SP group have the same prompts, we need to ensure
                    # we only take one copy of each prompt from each SP group
                    ordered_set_of_prompts = []
                    for sp_group_id in range(num_sp_groups):
                        # Get the first process from each SP group (typically the group leader)
                        group_leader_rank = sp_group_id * context_parallel_size

                        # Extract prompts from this SP group, accounting for num_generations duplicates
                        # We only need prompts from one rank in each SP group
                        group_prompts = all_prompts_text[
                            group_leader_rank * len(prompts_text) : (
                                group_leader_rank + 1
                            )
                            * len(prompts_text) : self.num_generations
                        ]

                        ordered_set_of_prompts.extend(group_prompts)
                else:
                    # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                    # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                    # prompt individually.
                    ordered_set_of_prompts = all_prompts_text[
                        :: self.num_generations * self.args.context_parallel_size
                    ]

                with profiling_context(self, "vLLM.generate"):
                    completion_ids = self.vllm_client.generate(
                        prompts=ordered_set_of_prompts,
                        n=self.num_generations,
                        repetition_penalty=self.repetition_penalty,
                        temperature=self.temperature,
                        top_p=self.top_p,
                        top_k=-1 if self.top_k is None else self.top_k,
                        min_p=0.0 if self.min_p is None else self.min_p,
                        max_tokens=self.max_completion_length,
                        guided_decoding_regex=self.guided_decoding_regex,
                    )
            else:
                completion_ids = [None] * (
                    len(all_prompts_text) // self.args.context_parallel_size
                )

            # Broadcast the completions from the main process to all processes
            completion_ids = broadcast_object_list(completion_ids, from_process=0)

            # Determine the appropriate slice based on sequence parallelism
            if self.args.context_parallel_size > 1:
                # Calculate SP group ID (which group of ranks this rank belongs to)
                sp_group_id = self.accelerator.process_index // self.local_world_size

                # Calculate the start index for this SP group
                sp_group_start = sp_group_id * len(prompts) * self.local_world_size

                # All ranks in the same SP group get the same data slice
                process_slice = slice(
                    sp_group_start,
                    sp_group_start + len(prompts),
                )
                completion_ids = completion_ids[process_slice]
            else:
                # Original behavior for non-sequence parallel case
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            # Pad the completions, and concatenate them with the prompts
            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            completion_ids = pad(
                completion_ids, padding_value=self.processing_class.pad_token_id
            )
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
            # Regular generation path
            with unwrap_model_for_generation(
                self.model_wrapped,
                self.accelerator,
                gather_deepspeed3_params=self.args.ds3_gather_for_generation,
            ) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids,
                    attention_mask=prompt_mask,
                    generation_config=self.generation_config,
                )

            # Compute prompt length and extract completion ids
            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # If mask_truncated_completions is enabled, zero out truncated completions in completion_mask
        if self.args.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)

        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text, strict=False):
                bootstrap = (
                    prompt.pop()["content"] if prompt[-1]["role"] == "assistant" else ""
                )
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}]
                )
        else:
            completions = completions_text

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )
        for i, (reward_func, reward_processing_class, reward_func_name) in enumerate(
            zip(
                self.reward_funcs,
                self.reward_processing_classes,
                self.reward_func_names,
                strict=False,
            )
        ):
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + c}
                            for p, c in zip(prompts, completions, strict=False)
                        ]
                        texts = [
                            apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [
                            p + c for p, c in zip(prompts, completions, strict=False)
                        ]
                    reward_inputs = reward_processing_class(
                        text=texts,
                        return_tensors="pt",
                        padding=True,
                        padding_side="right",
                        add_special_tokens=False,
                    )
                    reward_inputs = Trainer._prepare_inputs(self, reward_inputs)
                    with torch.inference_mode():
                        rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
                            :, 0
                        ]  # Shape (B*G,)
                else:
                    # Repeat all input columns (but "prompt" and "completion") to match the number of generations
                    keys = [
                        key for key in inputs[0] if key not in ["prompt", "completion"]
                    ]
                    reward_kwargs = {
                        key: [example[key] for example in inputs] for key in keys
                    }
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, **reward_kwargs
                    )
                    # Convert None values to NaN
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]

                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        # If all reward functions return None for a given row, issue a detailed warning
        if torch.isnan(rewards_per_func).all(dim=1).any():
            nan_row_idx = (
                torch.isnan(rewards_per_func).all(dim=1).nonzero(as_tuple=True)[0][0]
            )
            row_reward_kwargs = {
                key: value[nan_row_idx] for key, value in reward_kwargs.items()
            }
            row_reward_kwargs["prompt"] = prompts[nan_row_idx]
            row_reward_kwargs["completion"] = completions[nan_row_idx]
            warnings.warn(
                f"All reward functions returned None for the following kwargs: {row_reward_kwargs}. "
                "Please ensure that at least one reward function returns a valid reward.",
                stacklevel=2,
            )

        # Gather the reward per function: this part is crucial, because the rewards are normalized per group and the
        # completions may be distributed across processes
        rewards_per_func = gather(rewards_per_func)

        # Apply weights to each reward function's output and sum
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = rewards - mean_grouped_rewards
        if self.args.scale_rewards:
            advantages = advantages / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        if self.args.context_parallel_size > 1:
            # Calculate SP group ID (which group of ranks this rank belongs to)
            sp_group_id = self.accelerator.process_index // self.local_world_size

            # Calculate the start index for this SP group
            sp_group_start = sp_group_id * len(prompts) * self.local_world_size

            # All ranks in the same SP group get the same data slice
            process_slice = slice(
                sp_group_start,
                sp_group_start + len(prompts),
            )
        else:
            # Original behavior for non-sequence parallel case
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
        advantages = advantages[process_slice]

        # Log the metrics
        if mode == "train":
            self._total_train_tokens += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        # log completion lengths, mean, min, max
        agg_completion_mask = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)
        )
        self._metrics[mode]["completions/mean_length"].append(
            agg_completion_mask.float().mean().item()
        )
        self._metrics[mode]["completions/min_length"].append(
            agg_completion_mask.float().min().item()
        )
        self._metrics[mode]["completions/max_length"].append(
            agg_completion_mask.float().max().item()
        )

        # identify sequences that terminated with EOS and log their lengths
        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(
            agg_completion_mask
        )
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )
        if len(term_completion_mask) == 0:
            # edge case where no completed sequences are found
            term_completion_mask = torch.zeros(1, device=device)
        self._metrics[mode]["completions/mean_terminated_length"].append(
            term_completion_mask.float().mean().item()
        )
        self._metrics[mode]["completions/min_terminated_length"].append(
            term_completion_mask.float().min().item()
        )
        self._metrics[mode]["completions/max_terminated_length"].append(
            term_completion_mask.float().max().item()
        )

        # Calculate mean reward per function, but only for samples where the function was applied (non-NaN values)
        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        # Log prompt and completion texts
        self._textual_logs["prompt"].extend(gather_object(prompts_text))
        self._textual_logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._textual_logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
        }
