"""Module for Axolotl trainer sequence parallelism manager and utilities"""

import functools
from collections import defaultdict
from typing import DefaultDict

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.hooks import RemovableHandle
from transformers.modeling_outputs import CausalLMOutputWithPast

from axolotl.monkeypatch.attention.ring_attn.patch import (
    get_ring_attn_group,
    update_ring_attn_params,
)
from axolotl.utils.schemas.enums import RingAttnFunc


# TODO(djsaunde): implement zigzag, stripe patterns here (and elsewhere) in this
# module. Currently, we just focus on batch ring and varlen llama3 for simplicity.
def apply_sequence_parallelism(
    batch: dict[str, torch.Tensor],
    local_rank: int,
    local_world_size: int,
) -> tuple[dict[str, torch.Tensor], int, int]:
    """
    Apply sequence parallelism slicing to a batch.

    Special handling is implemented for integer logits_to_keep, which indicates
    to only keep the last N tokens in the sequence during generation.

    Args:
        batch: Batch dictionary (e.g., input_ids, attention_mask, etc.)
        local_rank: Local rank in the sequence parallel group
        local_world_size: World size of the sequence parallel group

    Returns:
        tuple: (sliced_batch, original_seq_len, pad_len)
            - sliced_batch: Batch dictionary with sliced tensors
            - original_seq_len: The original sequence length before padding
            - pad_len: The number of padding tokens added
    """
    batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

    # Update ring attention params if needed
    if batch.get("position_ids") is not None:
        update_ring_attn_params(position_ids=batch["position_ids"])

    # Add padding to make sequence length divisible by local_world_size
    original_seq_len = batch["input_ids"].size(1)
    total_seq_len = original_seq_len
    pad_len = 0

    # Handle logits_to_keep
    if "logits_to_keep" in batch and isinstance(batch["logits_to_keep"], int):
        logits_to_keep = batch["logits_to_keep"]

        # Calculate which positions in the full sequence contain the last N tokens
        start_position = max(0, total_seq_len - logits_to_keep)
        chunk_size = total_seq_len // local_world_size
        rank_start = local_rank * chunk_size
        rank_end = rank_start + chunk_size

        # Create a boolean mask tensor for this rank's chunk
        mask = torch.zeros(
            chunk_size,
            dtype=torch.bool,
            device=batch["input_ids"].device,
        )

        if rank_end > start_position:
            # Calculate how many of the last N tokens fall within this rank's range
            tokens_in_rank = min(rank_end, total_seq_len) - max(
                rank_start, start_position
            )

            # Calculate where these tokens start in the local chunk
            local_start_idx = max(0, start_position - rank_start)

            # Set the appropriate positions in the mask to True
            mask[local_start_idx : local_start_idx + tokens_in_rank] = True

        # Replace the integer with the boolean mask
        batch["logits_to_keep"] = mask

    # Calculate padding needed to make the sequence length divisible by the divisor
    divisor = local_world_size
    if total_seq_len % divisor != 0:
        pad_len = divisor - (total_seq_len % divisor)

        # Apply padding to all relevant tensors
        for key in batch:
            if (
                isinstance(batch[key], torch.Tensor)
                and batch[key].dim() > 1
                and batch[key].size(1) == total_seq_len
            ):
                # Create padding tensor
                pad_value = -100 if key == "labels" else 0
                padding = torch.full(
                    (batch[key].size(0), pad_len, *batch[key].shape[2:]),
                    pad_value,
                    dtype=batch[key].dtype,
                    device=batch[key].device,
                )

                # Concatenate padding to the right side of the tensor
                batch[key] = torch.cat([batch[key], padding], dim=1)
            if key == "logits_to_keep":
                # Create padding tensor
                padding = torch.ones(
                    1,
                    dtype=batch[key].dtype,
                    device=batch[key].device,
                )

                # Concatenate padding to the right side of the tensor
                batch[key] = torch.cat([batch[key], padding], dim=0)

        # Update the total sequence length after padding
        total_seq_len = batch["input_ids"].size(1)

    # Slice batch for sequence parallel processing
    for key in batch:
        if not isinstance(batch[key], torch.Tensor) or batch[key].dim() <= 1:
            continue

        if batch[key].size(1) == total_seq_len:
            # Split in sequential fashion and grab this rank's chunk
            batch[key] = (
                batch[key].chunk(local_world_size, dim=1)[local_rank].contiguous()
            )

    # Calculate the chunk size for each rank
    original_chunk_size = original_seq_len // local_world_size
    padded_chunk_size = total_seq_len // local_world_size

    # If position_ids aren't already in the batch, create them with the proper offset
    if "position_ids" not in batch:
        position_offset = local_rank * original_chunk_size
        batch["position_ids"] = torch.arange(
            position_offset,
            position_offset + padded_chunk_size,
            dtype=torch.long,
            device=batch["input_ids"].device,
        ).expand(batch["input_ids"].size(0), -1)

    return batch, original_seq_len, pad_len


class SequenceParallelContextManager:
    """
    Context manager for sequence parallelism operations.

    This class provides a context that will automatically apply sequence parallelism
    during model forward passes using a pre-forward hook, and gather outputs from
    across the sequence parallelism group using a post-forward hook.
    """

    def __init__(
        self,
        models: list[nn.Module],
        sequence_parallel_degree: int,
        ring_attn_func: RingAttnFunc,
    ):
        self.models = models
        self.sequence_parallel_degree = sequence_parallel_degree
        self.ring_attn_func = ring_attn_func
        self.process_group = get_ring_attn_group()

        # Initialize sequence parallel group details
        self.local_rank = dist.get_rank(self.process_group)
        self.local_world_size = dist.get_world_size(self.process_group)

        # Will store hook handles for removal
        self.hook_handles: DefaultDict[list[RemovableHandle]] = defaultdict(list)

        # Store original sequence length and padding information
        self.original_seq_len = None
        self.pad_len = 0

        # Create a partially applied version of the apply_sequence_parallelism function
        self.apply_sequence_parallelism = functools.partial(
            apply_sequence_parallelism,
            local_rank=self.local_rank,
            local_world_size=self.local_world_size,
        )

    def __enter__(self):
        # Forward pre-hook to apply sequence parallelism
        def sequence_parallel_pre_hook(_, args, kwargs):
            # Apply sequence parallelism to kwargs and get original sequence length and padding info
            kwargs, self.original_seq_len, self.pad_len = (
                self.apply_sequence_parallelism(batch=kwargs)
            )
            return args, kwargs

        # Forward post-hook to gather outputs
        def sequence_parallel_post_hook(_, __, output):
            # Gather the sharded outputs
            output = self.gather_outputs(output)

            # Remove padding if it was added
            if self.pad_len > 0:
                for key, value in output.items():
                    if isinstance(value, torch.Tensor) and value.dim() > 1:
                        if value.size(1) == self.original_seq_len + self.pad_len:
                            # Slice to remove padding
                            output[key] = value[:, : self.original_seq_len].contiguous()

            return output

        # Register both hooks
        for i, model in enumerate(self.models):
            self.hook_handles[i].append(
                model.register_forward_pre_hook(
                    sequence_parallel_pre_hook, with_kwargs=True
                )
            )
            self.hook_handles[i].append(
                model.register_forward_hook(sequence_parallel_post_hook)
            )

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks
        for key in self.hook_handles:
            for handle in self.hook_handles[key]:
                handle.remove()
            self.hook_handles[key] = []

    def gather_outputs(self, output: CausalLMOutputWithPast) -> CausalLMOutputWithPast:
        """Gather sharded outputs from all ranks and reconstruct the full tensor."""
        for key, value in output.items():
            if isinstance(value, torch.Tensor) and value.dim() > 1:
                # Gather sequence-sharded tensors
                gathered_value = self.gather_tensor(value)
                output[key] = gathered_value
            else:
                gathered_value = value.clone()
                dist.all_reduce(
                    gathered_value, op=dist.ReduceOp.SUM, group=self.process_group
                )
                output[key] = gathered_value

        return output

    def gather_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """Gather a sharded tensor from all ranks while preserving gradients."""
        world_size = self.local_world_size
        group = self.process_group

        if not tensor.requires_grad:
            # Non-grad version — simple all_gather under no_grad
            with torch.no_grad():
                local_shape = torch.tensor(list(tensor.shape), device=tensor.device)
                all_shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
                dist.all_gather(all_shapes, local_shape, group=group)
                gathered_tensors = [
                    torch.zeros(tuple(s.tolist()), dtype=tensor.dtype, device=tensor.device)
                    for s in all_shapes
                ]
                dist.all_gather(gathered_tensors, tensor, group=group)
                return torch.cat(gathered_tensors, dim=1)

        # Gradient-preserving gather
        return AllGatherWithGrad.apply(tensor, group)

class AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, group):
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.input_shape = input_tensor.shape

        # Gather shape metadata
        local_shape = torch.tensor(list(input_tensor.shape), device=input_tensor.device)
        world_size = dist.get_world_size(group)
        all_shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
        ctx.all_seq_lens = [int(s[1].item()) for s in all_shapes]

        # Gradient-tracking all_gather
        gathered = [
            torch.zeros(tuple(s.tolist()), dtype=input_tensor.dtype, device=input_tensor.device)
            for s in all_shapes
        ]
        dist.all_gather(gathered, input_tensor, group=group)
        return torch.cat(gathered, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        rank = ctx.rank
        all_seq_lens = ctx.all_seq_lens
        input_shape = ctx.input_shape

        # Slice the portion of grad_output for this rank
        start_idx = sum(all_seq_lens[:rank])
        end_idx = start_idx + all_seq_lens[rank]
        grad_input = grad_output[:, start_idx:end_idx].contiguous()

        if grad_input.shape != input_shape:
            raise ValueError(
                f"Gradient shape mismatch: expected {input_shape}, got {grad_input.shape}"
            )

        return grad_input, None  # second return is for 'group'
