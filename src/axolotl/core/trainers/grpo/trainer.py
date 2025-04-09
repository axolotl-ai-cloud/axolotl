"""Axolotl GRPO trainer"""

from contextlib import nullcontext

import torch
import torch.distributed as dist
from accelerate.utils import is_deepspeed_available, is_peft_model
from trl import GRPOTrainer
from trl.extras.profiling import profiling_decorator

from axolotl.core.trainers.mixins import RngLoaderMixin, SchedulerMixin
from axolotl.monkeypatch.attention.ring_attn import (
    get_ring_attn_group,
    update_ring_attn_params,
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
        if self.args.sequence_parallel_degree > 1:
            sp_group = get_ring_attn_group()
            self.local_rank = dist.get_rank(group=sp_group)
            self.local_world_size = dist.get_world_size(group=sp_group)

            # Pad sequence if needed
            total_seq_len = input_ids.shape[1]
            remainder = total_seq_len % self.local_world_size
            if remainder != 0:
                padding = self.local_world_size - remainder

                if dist.get_rank() == 0:
                    import ipdb

                    ipdb.set_trace()
                dist.barrier()

                pad_token_id = self.processing_class.pad_token_id or 0
                padding = torch.full(
                    (input_ids.shape[0], padding),
                    pad_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
                input_ids = torch.cat([input_ids, padding], dim=1)

                # Also pad attention mask if it exists
                if attention_mask is not None:
                    attn_padding = torch.zeros(
                        (attention_mask.shape[0], padding),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([attention_mask, attn_padding], dim=1)

                # Update total_seq_len after padding
                total_seq_len += padding

            # Get local (start, end) for sequence parallelism slicing
            slice_size = total_seq_len // self.local_world_size
            start = self.local_rank * slice_size
            end = start + slice_size

            # Slice data for sequence parallel processing
            input_ids = input_ids[:, start:end]
            attention_mask = attention_mask[:, start:end]

        super()._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)
