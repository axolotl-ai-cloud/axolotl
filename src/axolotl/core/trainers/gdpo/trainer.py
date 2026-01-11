"""
Axolotl GDPO trainers (with and without sequence parallelism handling)

GDPO (Group Reward-Decoupled Normalization Policy Optimization) extends GRPO
with decoupled per-reward normalization for multi-reward RL training.

The key difference from GRPO is in advantage calculation:
- GRPO: Combines rewards first, then normalizes
- GDPO: Normalizes each reward independently, then combines

This preserves reward signal resolution when training with multiple reward functions.
"""

import warnings
from typing import Any

import torch
import torch.utils.data
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_available,
)
from datasets import Dataset, IterableDataset
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
)
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

from axolotl.core.trainers.grpo.trainer import (
    AxolotlGRPOSequenceParallelTrainer,
    AxolotlGRPOTrainer,
)

if is_peft_available():
    from peft import PeftConfig


def compute_gdpo_advantages(
    rewards_per_func: torch.Tensor,
    reward_weights: torch.Tensor,
    num_generations: int,
    scale_rewards: bool = True,
    gdpo_epsilon: float = 1e-4,
    gdpo_batch_norm: bool = False,
    gdpo_per_reward_scale: bool = True,
    device: torch.device | None = None,
) -> torch.Tensor:
    """
    Compute advantages using GDPO's decoupled normalization.

    Unlike GRPO which combines rewards first then normalizes, GDPO normalizes
    each reward function independently before combining them with weights.
    This preserves the relative differences between different reward signals.

    Args:
        rewards_per_func: Tensor of shape (batch_size, num_reward_funcs) containing
            rewards from each reward function for each sample.
        reward_weights: Tensor of shape (num_reward_funcs,) containing weights
            for each reward function.
        num_generations: Number of generations per prompt (group size for normalization).
        scale_rewards: Whether to scale rewards by their standard deviation.
        gdpo_epsilon: Small constant for numerical stability in normalization.
        gdpo_batch_norm: Whether to apply batch-wise normalization after combining.
        gdpo_per_reward_scale: Whether to scale each reward by its std before combining.
        device: Device to perform computations on.

    Returns:
        advantages: Tensor of shape (batch_size,) containing computed advantages.
    """
    if device is None:
        device = rewards_per_func.device

    num_samples = rewards_per_func.size(0)
    num_rewards = rewards_per_func.size(1)

    # Ensure reward_weights is on the correct device
    reward_weights = reward_weights.to(device)

    # Initialize advantages accumulator
    combined_advantages = torch.zeros(num_samples, device=device)

    # Process each reward function independently (GDPO's core innovation)
    for i in range(num_rewards):
        reward_i = rewards_per_func[:, i]
        weight_i = reward_weights[i]

        # Reshape to (num_prompts, num_generations) for group-wise statistics
        reward_grouped = reward_i.view(-1, num_generations)

        # Compute group-wise mean using nanmean to handle NaN values
        mean_i = torch.nanmean(reward_grouped, dim=1, keepdim=True)

        # Compute group-wise std handling NaN values
        valid_mask = ~torch.isnan(reward_grouped)
        valid_counts = valid_mask.sum(dim=1, keepdim=True)
        squared_diff = torch.where(
            valid_mask,
            (reward_grouped - mean_i) ** 2,
            torch.zeros_like(reward_grouped),
        )

        # Use Bessel's correction (N-1) for unbiased estimation, matching torch.std()
        # If N <= 1, variance is technically undefined/0 for our purposes
        divisor = (valid_counts - 1).clamp(min=0)
        variance = squared_diff.sum(dim=1, keepdim=True) / divisor.clamp(min=1)

        # If we had 0 or 1 valid samples, variance should be 0
        variance = torch.where(divisor > 0, variance, torch.zeros_like(variance))

        std_i = torch.sqrt(variance)

        # Expand statistics back to original shape
        mean_i = mean_i.repeat(1, num_generations).view(-1)
        std_i = std_i.repeat(1, num_generations).view(-1)

        # Compute normalized advantage for this reward
        # Handle NaNs in reward_i: if reward is NaN, advantage should be 0 (neutral)
        # We use where to avoid propagating NaN
        advantage_i = torch.where(
            torch.isnan(reward_i), torch.zeros_like(reward_i), reward_i - mean_i
        )

        # Apply per-reward scaling if enabled
        if scale_rewards and gdpo_per_reward_scale:
            advantage_i = advantage_i / (std_i + gdpo_epsilon)

        # Apply weight and accumulate
        combined_advantages = combined_advantages + advantage_i * weight_i

    # Optional: Apply batch-wise normalization to combined advantages
    if gdpo_batch_norm:
        batch_mean = torch.nanmean(combined_advantages)
        batch_std = nanstd(combined_advantages)
        combined_advantages = (combined_advantages - batch_mean) / (
            batch_std + gdpo_epsilon
        )

    return combined_advantages


class AxolotlGDPOTrainer(AxolotlGRPOTrainer):
    """
    GDPO Trainer extending GRPO with decoupled reward normalization.

    GDPO (Group Reward-Decoupled Normalization Policy Optimization) addresses
    the reward advantage collapse problem in multi-reward GRPO training by
    normalizing each reward function independently before combining them.

    Key differences from GRPO:
    - Each reward function is normalized within its generation group independently
    - Normalized advantages are then combined using the specified weights
    - Optional batch-wise normalization can be applied after combining

    This preserves the relative differences between reward signals and enables
    more stable training when using multiple reward functions.
    """

    _tag_names = ["trl", "gdpo", "axolotl"]

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """
        Generate completions and score them with reward functions.

        This method overrides the GRPO implementation to use GDPO's
        decoupled normalization for computing advantages when multiple
        reward functions are used.
        """
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
            if self.state.global_step != self._last_loaded_step:  # type: ignore[has-type]
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]

                with profiling_context(self, "vLLM.generate"):
                    output = self.vllm_client.generate(
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
                # vLLM client returns a dict with prompt_ids, completion_ids, logprobs
                completion_ids = output["completion_ids"]
                logprobs_list = output.get("logprobs")
            else:
                completion_ids = [None] * len(all_prompts_text)
                logprobs_list = [None] * len(all_prompts_text)

            completion_ids = broadcast_object_list(completion_ids, from_process=0)
            logprobs_list = broadcast_object_list(logprobs_list, from_process=0)

            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]
            logprobs_list = logprobs_list[process_slice]

            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            completion_ids = pad(
                completion_ids, padding_value=self.processing_class.pad_token_id
            )

            # Convert logprobs to tensor for importance sampling
            if logprobs_list is not None and logprobs_list[0] is not None:
                sampling_per_token_logps = [
                    torch.tensor(logps, device=device) for logps in logprobs_list
                ]
                sampling_per_token_logps = pad(
                    sampling_per_token_logps, padding_value=0.0, padding_side="right"
                )
            else:
                sampling_per_token_logps = None

            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
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

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            sampling_per_token_logps = None

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

        if self.args.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            if self.num_iterations > 1 or (
                self.args.use_vllm
                and getattr(self.args, "vllm_importance_sampling_correction", True)
            ):
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                old_per_token_logps = None

            # Compute importance sampling ratio for vLLM
            if (
                self.args.use_vllm
                and getattr(self.args, "vllm_importance_sampling_correction", True)
                and sampling_per_token_logps is not None
                and old_per_token_logps is not None
            ):
                importance_sampling_ratio = torch.exp(
                    old_per_token_logps - sampling_per_token_logps
                )
                importance_sampling_ratio = torch.clamp(
                    importance_sampling_ratio,
                    max=getattr(self.args, "vllm_importance_sampling_cap", 10.0),
                )
            else:
                importance_sampling_ratio = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
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

        # Compute rewards from each reward function
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
                if isinstance(reward_func, nn.Module):
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
                        ]
                else:
                    keys = [
                        key for key in inputs[0] if key not in ["prompt", "completion"]
                    ]
                    reward_kwargs = {
                        key: [example[key] for example in inputs] for key in keys
                    }
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, **reward_kwargs
                    )
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]

                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

        # Check for all-NaN rewards
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

        rewards_per_func = gather(rewards_per_func)

        # Get GDPO-specific config values
        gdpo_decoupled_norm = getattr(self.args, "gdpo_decoupled_norm", True)
        gdpo_batch_norm = getattr(self.args, "gdpo_batch_norm", False)
        gdpo_epsilon = getattr(self.args, "gdpo_epsilon", 1e-4)
        gdpo_per_reward_scale = getattr(self.args, "gdpo_per_reward_scale", True)

        # Use GDPO's decoupled normalization when we have multiple reward functions
        if gdpo_decoupled_norm and len(self.reward_funcs) > 1:
            advantages = compute_gdpo_advantages(
                rewards_per_func=rewards_per_func,
                reward_weights=self.reward_weights,
                num_generations=self.num_generations,
                scale_rewards=self.args.scale_rewards,
                gdpo_epsilon=gdpo_epsilon,
                gdpo_batch_norm=gdpo_batch_norm,
                gdpo_per_reward_scale=gdpo_per_reward_scale,
                device=device,
            )

            # Also compute combined rewards for logging (same as GRPO)
            rewards = (
                rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
            ).nansum(dim=1)
        else:
            # Fall back to GRPO behavior for single reward or when disabled
            rewards = (
                rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
            ).nansum(dim=1)

            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0
            )
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0
            )
            advantages = rewards - mean_grouped_rewards
            if self.args.scale_rewards:
                advantages = advantages / (std_grouped_rewards + gdpo_epsilon)

        # Compute grouped statistics for logging
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log metrics
        if mode == "train":
            self._total_train_tokens += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

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

        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(
            agg_completion_mask
        )
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )
        if len(term_completion_mask) == 0:
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

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "num_items_in_batch": completion_mask.sum(),
        }

        if importance_sampling_ratio is not None:
            output["importance_sampling_ratio"] = importance_sampling_ratio

        return output


class AxolotlGDPOSequenceParallelTrainer(
    AxolotlGRPOSequenceParallelTrainer, AxolotlGDPOTrainer
):
    """
    GDPO Trainer with sequence parallelism support.

    Extends AxolotlGDPOTrainer with sequence parallelism handling for
    efficient training on long sequences across multiple GPUs.
    """

    _tag_names = ["trl", "gdpo", "axolotl", "sequence-parallel"]

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
        # Call parent constructors
        AxolotlGRPOSequenceParallelTrainer.__init__(
            self,
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

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """
        Generate completions and score them with GDPO advantages.

        This method handles sequence parallelism while using GDPO's
        decoupled normalization for computing advantages.
        """
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

        # Generate completions using vLLM with sequence parallelism handling
        if self.args.use_vllm:
            if self.state.global_step != self._last_loaded_step:  # type: ignore[has-type]
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            all_prompts_text = gather_object(prompts_text)
            if self.accelerator.is_main_process:
                if self.args.context_parallel_size > 1:
                    world_size = self.accelerator.num_processes
                    context_parallel_size = self.args.context_parallel_size
                    num_sp_groups = world_size // context_parallel_size

                    ordered_set_of_prompts = []
                    for sp_group_id in range(num_sp_groups):
                        group_leader_rank = sp_group_id * context_parallel_size
                        group_prompts = all_prompts_text[
                            group_leader_rank * len(prompts_text) : (
                                group_leader_rank + 1
                            )
                            * len(prompts_text) : self.num_generations
                        ]
                        ordered_set_of_prompts.extend(group_prompts)
                else:
                    ordered_set_of_prompts = all_prompts_text[
                        :: self.num_generations * self.args.context_parallel_size
                    ]

                with profiling_context(self, "vLLM.generate"):
                    output = self.vllm_client.generate(
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
                # vLLM client returns a dict with prompt_ids, completion_ids, logprobs
                completion_ids = output["completion_ids"]
            else:
                completion_ids = [None] * (
                    len(all_prompts_text) // self.args.context_parallel_size
                )

            completion_ids = broadcast_object_list(completion_ids, from_process=0)

            if self.args.context_parallel_size > 1:
                sp_group_id = self.accelerator.process_index // self.local_world_size
                sp_group_start = sp_group_id * len(prompts) * self.local_world_size
                process_slice = slice(
                    sp_group_start,
                    sp_group_start + len(prompts),
                )
                completion_ids = completion_ids[process_slice]
            else:
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            completion_ids = [
                torch.tensor(ids, device=device) for ids in completion_ids
            ]
            completion_ids = pad(
                completion_ids, padding_value=self.processing_class.pad_token_id
            )
            prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        else:
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

        if self.args.mask_truncated_completions:
            truncated_completions = ~is_eos.any(dim=1)
            completion_mask = (
                completion_mask * (~truncated_completions).unsqueeze(1).int()
            )

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)
        batch_size = (
            self.args.per_device_train_batch_size
            if mode == "train"
            else self.args.per_device_eval_batch_size
        )

        with torch.no_grad():
            if self.num_iterations > 1:
                old_per_token_logps, _ = self._get_per_token_logps_and_entropies(
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
                ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                    self.ref_model,
                    prompt_completion_ids,
                    attention_mask,
                    logits_to_keep,
                    batch_size,
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps, _ = self._get_per_token_logps_and_entropies(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                        batch_size,
                    )

        # Decode completions
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

        # Compute rewards
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
                if isinstance(reward_func, nn.Module):
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
                        ]
                else:
                    keys = [
                        key for key in inputs[0] if key not in ["prompt", "completion"]
                    ]
                    reward_kwargs = {
                        key: [example[key] for example in inputs] for key in keys
                    }
                    output_reward_func = reward_func(
                        prompts=prompts, completions=completions, **reward_kwargs
                    )
                    output_reward_func = [
                        reward if reward is not None else torch.nan
                        for reward in output_reward_func
                    ]

                    rewards_per_func[:, i] = torch.tensor(
                        output_reward_func, dtype=torch.float32, device=device
                    )

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

        # Gather rewards
        rewards_per_func = gather(rewards_per_func)

        # GDPO advantage calculation
        gdpo_decoupled_norm = getattr(self.args, "gdpo_decoupled_norm", True)
        gdpo_batch_norm = getattr(self.args, "gdpo_batch_norm", False)
        gdpo_epsilon = getattr(self.args, "gdpo_epsilon", 1e-4)
        gdpo_per_reward_scale = getattr(self.args, "gdpo_per_reward_scale", True)

        if gdpo_decoupled_norm and len(self.reward_funcs) > 1:
            advantages = compute_gdpo_advantages(
                rewards_per_func=rewards_per_func,
                reward_weights=self.reward_weights,
                num_generations=self.num_generations,
                scale_rewards=self.args.scale_rewards,
                gdpo_epsilon=gdpo_epsilon,
                gdpo_batch_norm=gdpo_batch_norm,
                gdpo_per_reward_scale=gdpo_per_reward_scale,
                device=device,
            )
            rewards = (
                rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
            ).nansum(dim=1)
        else:
            rewards = (
                rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
            ).nansum(dim=1)

            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0
            )
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(
                self.num_generations, dim=0
            )
            advantages = rewards - mean_grouped_rewards
            if self.args.scale_rewards:
                advantages = advantages / (std_grouped_rewards + gdpo_epsilon)

        # Compute stats for logging
        rewards = (
            rewards_per_func * self.reward_weights.to(device).unsqueeze(0)
        ).nansum(dim=1)
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Slice for sequence parallelism
        if self.args.context_parallel_size > 1:
            sp_group_id = self.accelerator.process_index // self.local_world_size
            sp_group_start = sp_group_id * len(prompts) * self.local_world_size
            process_slice = slice(
                sp_group_start,
                sp_group_start + len(prompts),
            )
        else:
            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
        advantages = advantages[process_slice]

        # Log metrics
        if mode == "train":
            self._total_train_tokens += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

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

        agg_terminated_with_eos = self.accelerator.gather_for_metrics(is_eos.any(dim=1))
        term_completion_mask = agg_completion_mask[agg_terminated_with_eos]
        clipped_completions_ratio = 1 - len(term_completion_mask) / len(
            agg_completion_mask
        )
        self._metrics[mode]["completions/clipped_ratio"].append(
            clipped_completions_ratio
        )
        if len(term_completion_mask) == 0:
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

        for i, reward_func_name in enumerate(self.reward_func_names):
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/mean"].append(mean_rewards)
            std_rewards = nanstd(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}/std"].append(std_rewards)
        self._metrics[mode]["reward"].append(mean_grouped_rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        self._logs["prompt"].extend(gather_object(prompts_text))
        self._logs["completion"].extend(gather_object(completions_text))
        for i, name in enumerate(self.reward_func_names):
            self._logs["rewards"][name].extend(rewards_per_func[:, i].tolist())

        output = {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "advantages": advantages,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "num_items_in_batch": completion_mask.sum(),
        }
        # if importance_sampling_ratio is not None:
        #    output["importance_sampling_ratio"] = importance_sampling_ratio
        return output
