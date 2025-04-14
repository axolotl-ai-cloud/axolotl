"""Axolotl GRPO trainer"""

from contextlib import nullcontext

import torch
import torch.distributed as dist
from accelerate.utils import is_deepspeed_available, is_peft_model
from trl import GRPOTrainer
from trl.extras.profiling import profiling_decorator
from trl.trainer.utils import selective_log_softmax

from axolotl.core.trainers.mixins import RngLoaderMixin, SchedulerMixin
from axolotl.monkeypatch.attention.ring_attn import (
    get_ring_attn_group,
)

if is_deepspeed_available():
    import deepspeed


class AxolotlGRPOTrainer(RngLoaderMixin, SchedulerMixin, GRPOTrainer):
    """Extend the base GRPOTrainer for axolotl helpers"""

    _tag_names = ["trl", "grpo", "axolotl"]

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

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        if dist.get_rank() == 0:
            import ipdb; ipdb.set_trace()
        dist.barrier()

        if dist.get_rank() == 1:
            import ipdb; ipdb.set_trace()
        dist.barrier()

        if self.args.sequence_parallel_degree > 1:
            sp_group = get_ring_attn_group()
            self.local_rank = dist.get_rank(group=sp_group)
            self.local_world_size = dist.get_world_size(group=sp_group)

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
            print(f"{self.local_rank}: tokens_before_our_slice: {tokens_before_our_slice}")
            if tokens_before_our_slice < logits_to_keep:
                # How many tokens from our slice are needed
                tokens_needed_from_slice = logits_to_keep - tokens_before_our_slice
                logits_to_keep = min(slice_size, tokens_needed_from_slice)
            else:
                # This rank doesn't contain any tokens we need to keep
                logits_to_keep = 0

        print(f"{self.local_rank}: logits_to_keep: {logits_to_keep}")

        # We add 1 to `logits_to_keep` because the last logits of the sequence is later excluded
        logits = model(input_ids=input_ids, attention_mask=attention_mask, logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred

        print(f"{self.local_rank}: logits.shape: {logits.shape}")

        # First, let all ranks know the shape of each rank's tensor
        local_shape = torch.tensor([logits.shape[0], logits.shape[1], logits.shape[2]], device=logits.device)
        all_shapes = [torch.zeros_like(local_shape) for _ in range(self.local_world_size)]
        dist.all_gather(all_shapes, local_shape, group=sp_group)

        # Use a list-based approach to collect logits of different sizes
        if self.local_rank == 0:
            # Root process allocates space for receiving
            gathered_logits = []
            for shape in all_shapes:
                b, s, v = shape.tolist()
                gathered_logits.append(torch.zeros((b, s, v), dtype=logits.dtype, device=logits.device))
        else:
            gathered_logits = None
            
        # Gather to rank 0
        dist.gather(logits, gathered_logits, dst=0, group=sp_group)
        
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
                device=logits.device
            )
        
        # Broadcast the result back to all ranks
        dist.broadcast(concatenated_logits, src=0, group=sp_group)
        logits = concatenated_logits

        input_ids = input_ids[:, -logits_to_keep:]
        # For transformers<=4.48, logits_to_keep argument isn't supported, so here we drop logits ourselves.
        # See https://github.com/huggingface/trl/issues/2770
        logits = logits[:, -logits_to_keep:]
        # Divide logits by sampling temperature.
        # See https://huggingface.co/blog/the_n_implementation_details_of_rlhf_with_ppo#policy-training-implementation-details
        logits = logits / self.temperature

        dist.barrier()

        return selective_log_softmax(logits, input_ids)  # compute logprobs for the input tokens

        # super()._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
