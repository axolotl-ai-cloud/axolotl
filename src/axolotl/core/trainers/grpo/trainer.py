"""Axolotl GRPO trainer"""

import warnings
from contextlib import nullcontext
from typing import Any

import datasets
import torch
import torch.distributed as dist
from accelerate.utils import (
    broadcast_object_list,
    gather,
    gather_object,
    is_peft_model,
)
from torch import nn
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Sampler,
)
from transformers import Trainer, is_wandb_available
from transformers.trainer_utils import seed_worker
from trl import GRPOTrainer
from trl.data_utils import (
    apply_chat_template,
    is_conversational,
    maybe_apply_chat_template,
)
from trl.extras.profiling import profiling_context, profiling_decorator
from trl.import_utils import is_deepspeed_available, is_rich_available
from trl.models import unwrap_model_for_generation
from trl.trainer.utils import (
    pad,
    print_prompt_completions_sample,
    selective_log_softmax,
)

from axolotl.core.trainers.grpo.sampler import SequenceParallelRepeatRandomSampler
from axolotl.core.trainers.mixins import RngLoaderMixin, SchedulerMixin
from axolotl.monkeypatch.attention.ring_attn.patch import get_ring_attn_group

if is_deepspeed_available():
    import deepspeed

if is_wandb_available():
    import wandb


class AxolotlGRPOTrainer(RngLoaderMixin, SchedulerMixin, GRPOTrainer):
    """Extend the base GRPOTrainer for axolotl helpers"""

    _tag_names = ["trl", "grpo", "axolotl"]

    def __init__(self, *args, **kwargs):
        # Call parent constructor with all arguments
        super().__init__(*args, **kwargs)

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
            batch_size=effective_batch_size
            // self.num_generations
            // self.args.sequence_parallel_degree,
            repeat_count=self.num_iterations,
            sequence_parallel_degree=self.args.sequence_parallel_degree,
            world_size=world_size,
            rank=rank,
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
            self.data_collator = self._get_collator_with_removed_columns(
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

    def _generate_and_score_completions(
        self, inputs: dict[str | torch.Tensor | Any]
    ) -> dict[str, torch.Tensor | Any]:
        device = self.accelerator.device
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
            if self.state.global_step != self._last_loaded_step:
                self._move_model_to_vllm()
                self._last_loaded_step = self.state.global_step

            # Generate completions using vLLM: gather all prompts and use them in a single call in the main process
            all_prompts_text = gather_object(prompts_text)

            if self.accelerator.is_main_process:
                # Since 'prompts' contains 'num_generations' duplicates, we first take unique prompts, and generate
                # num_generations outputs for each one. This is faster than generating outputs for each duplicate
                # prompt individually.
                ordered_set_of_prompts = all_prompts_text[:: self.num_generations]
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
                completion_ids = [None] * len(all_prompts_text)

            # Broadcast the completions from the main process to all processes
            completion_ids = broadcast_object_list(completion_ids, from_process=0)

            # Determine the appropriate slice based on sequence parallelism
            if self.args.sequence_parallel_degree > 1:
                # Calculate SP group ID (which group of ranks this rank belongs to)
                sp_group_id = self.accelerator.process_index // self.local_world_size

                # Calculate the start index for this SP group
                sp_group_start = sp_group_id * len(prompts) * self.local_world_size

                # All ranks in the same SP group get the same data slice
                # This ensures identical inputs for sequence-parallel processing
                process_slice = slice(
                    sp_group_start,
                    sp_group_start + len(prompts) * self.local_world_size,
                )

                # Take the full SP group's worth of completions
                completion_ids = completion_ids[process_slice]
            else:
                # Original behavior for non-sequence-parallel case
                process_slice = slice(
                    self.accelerator.process_index * len(prompts),
                    (self.accelerator.process_index + 1) * len(prompts),
                )
                completion_ids = completion_ids[process_slice]

            if dist.get_rank() == 0:
                import ipdb

                ipdb.set_trace()
            dist.barrier()
            if dist.get_rank() == 1:
                import ipdb

                ipdb.set_trace()
            dist.barrier()

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

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B, P+C)
        logits_to_keep = completion_ids.size(
            1
        )  # we only need to compute the logits for the completion tokens

        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
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
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model,
                        prompt_completion_ids,
                        attention_mask,
                        logits_to_keep,
                    )

        # Decode the generated completions
        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
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
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = (
                    f"reward {reward_func.config._name_or_path.split('/')[-1]}"
                )
            else:
                reward_func_name = reward_func.__name__
            with profiling_context(self, reward_func_name):
                if isinstance(
                    reward_func, nn.Module
                ):  # Module instead of PretrainedModel for compat with compiled models
                    if is_conversational(inputs[0]):
                        messages = [
                            {"messages": p + c} for p, c in zip(prompts, completions)
                        ]
                        texts = [
                            apply_chat_template(x, reward_processing_class)["text"]
                            for x in messages
                        ]
                    else:
                        texts = [p + c for p, c in zip(prompts, completions)]
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
                "Please ensure that at least one reward function returns a valid reward."
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
        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "train":
            self._total_train_tokens += (
                self.accelerator.gather_for_metrics(attention_mask.sum()).sum().item()
            )
        self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics[mode]["completion_length"].append(completion_length)

        # Calculate mean reward per function, but only for samples where the function was applied
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(
                reward_func, nn.Module
            ):  # Module instead of PretrainedModel for compat with compiled models
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            # Only calculate mean for samples where this reward function was applied (non-NaN values)
            mean_rewards = torch.nanmean(rewards_per_func[:, i]).item()
            self._metrics[mode][f"rewards/{reward_func_name}"].append(mean_rewards)
        self._metrics[mode]["reward"].append(rewards.mean().item())
        self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if (
            self.log_completions
            and self.state.global_step % self.args.logging_steps == 0
        ):
            prompts_to_log = gather_object(prompts_text)
            completions_to_log = gather_object(completions_text)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        prompts_to_log,
                        completions_to_log,
                        rewards_to_log,
                        self.state.global_step,
                    )
                if (
                    self.args.report_to
                    and "wandb" in self.args.report_to
                    and wandb.run is not None
                ):
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        if self.args.sequence_parallel_degree > 1:
            print(f"{self.local_rank}: input_ids.shape: {input_ids.shape}")
            print(f"{self.local_rank}: input_ids[0, :20]: {input_ids[0, :20]}")
            print(f"{self.local_rank}: input_ids[0, -20:]: {input_ids[0, -20:]}")

            # Pad sequence if needed
            total_seq_len = input_ids.shape[1]
            remainder = total_seq_len % self.local_world_size
            if remainder != 0:
                to_pad = self.local_world_size - remainder
                pad_token_id = self.processing_class.pad_token_id or 0
                padding = torch.full(
                    (input_ids.shape[0], to_pad),
                    pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                input_ids = torch.cat([input_ids, padding], dim=1)

                # Also pad attention mask if it exists
                if attention_mask is not None:
                    attn_padding = torch.zeros(
                        (attention_mask.shape[0], to_pad),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([attention_mask, attn_padding], dim=1)

                # Update total_seq_len after padding
                total_seq_len += to_pad

            # Get local (start, end) for sequence parallelism slicing
            slice_size = total_seq_len // self.local_world_size
            start = self.local_rank * slice_size
            end = start + slice_size

            # Slice data for sequence parallel processing
            input_ids = input_ids[:, start:end]
            attention_mask = attention_mask[:, start:end]

            # Calculate if this rank contains any tokens we need to keep
            tokens_before_our_slice = self.local_rank * slice_size
            print(f"{self.local_rank}: slice_size: {slice_size}")
            print(
                f"{self.local_rank}: tokens_before_our_slice: {tokens_before_our_slice}"
            )
            if tokens_before_our_slice < logits_to_keep:
                # How many tokens from our slice are needed
                tokens_needed_from_slice = logits_to_keep - tokens_before_our_slice
                logits_to_keep = min(slice_size, tokens_needed_from_slice)
            else:
                # This rank doesn't contain any tokens we need to keep
                logits_to_keep = 0

            print(f"{self.local_rank}: logits_to_keep: {logits_to_keep}")

            # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=logits_to_keep + 1,
            ).logits
            logits = logits[
                :, :-1, :
            ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

            print(f"{self.local_rank}: logits.shape: {logits.shape}")

            # First, let all ranks know the shape of each rank's tensor
            local_shape = torch.tensor(
                [logits.shape[0], logits.shape[1], logits.shape[2]],
                device=logits.device,
            )
            all_shapes = [
                torch.zeros_like(local_shape) for _ in range(self.local_world_size)
            ]
            dist.all_gather(all_shapes, local_shape, group=self.sp_group)

            # Use a list-based approach to collect logits of different sizes
            if self.local_rank == 0:
                # Root process allocates space for receiving
                gathered_logits = []
                for shape in all_shapes:
                    b, s, v = shape.tolist()
                    gathered_logits.append(
                        torch.zeros((b, s, v), dtype=logits.dtype, device=logits.device)
                    )
            else:
                gathered_logits = None

            # Gather to rank 0
            dist.gather(logits, gathered_logits, dst=0, group=self.sp_group)

            # On rank 0, concatenate and distribute the result
            if self.local_rank == 0:
                concatenated_logits = torch.cat(gathered_logits, dim=1)
                # Trim to keep only what we need
                if concatenated_logits.shape[1] > logits_to_keep:
                    concatenated_logits = concatenated_logits[:, -logits_to_keep:, :]
            else:
                concatenated_logits = torch.zeros(
                    (logits.shape[0], logits_to_keep, logits.shape[2]),
                    dtype=logits.dtype,
                    device=logits.device,
                )

            # Broadcast the result back to all ranks
            dist.broadcast(concatenated_logits, src=0, group=self.sp_group)
            logits = concatenated_logits

            input_ids = input_ids[:, -logits_to_keep:]
            # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
            # See https://github.com/huggingface/trl/issues/2770
            logits = logits[:, -logits_to_keep:]
            # Divide logits by sampling temperature.
            # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
            logits = logits / self.temperature

            dist.barrier()

            return selective_log_softmax(
                logits, input_ids
            )  # compute logprobs for the input tokens
        else:
            super()._get_per_token_logps(
                model, input_ids, attention_mask, logits_to_keep
            )
