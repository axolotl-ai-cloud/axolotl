"""Module for Axolotl trainer sequence parallelism manager and utilities"""

from collections import defaultdict
import functools
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

PADDING_MAP = {
    "input_ids": 0,
    "attention_mask": 0,
    "labels": -100,
}


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
            tokens_in_rank = min(rank_end, total_seq_len) - max(rank_start, start_position)
            
            # Calculate where these tokens start in the local chunk
            local_start_idx = max(0, start_position - rank_start)
            
            # Set the appropriate positions in the mask to True
            mask[local_start_idx:local_start_idx + tokens_in_rank] = True
        
        # Replace the integer with the boolean mask
        batch["logits_to_keep"] = mask

    divisor = local_world_size

    # Calculate padding needed to make the sequence length divisible by the divisor
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
                pad_value = PADDING_MAP.get(key, 0)
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

    # Calculate the chunk size for each rank
    chunk_size = total_seq_len // local_world_size

    # If position_ids aren't already in the batch, create them with the proper offset
    if "position_ids" not in batch:
        # Calculate position offset for this rank
        position_offset = local_rank * chunk_size

        # Create position IDs tensor with the correct global positions
        batch["position_ids"] = torch.arange(
            position_offset, 
            position_offset + chunk_size,
            dtype=torch.long,
            device=batch["input_ids"].device
        ).expand(batch["input_ids"].size(0), -1)

    # Slice batch for sequence parallel processing
    for key in batch:
        # Skip non-tensor values or tensors without sequence dimension
        if not isinstance(batch[key], torch.Tensor) or batch[key].dim() <= 1:
            continue

        if batch[key].size(1) == total_seq_len:
            # Split in sequential fashion and grab this rank's chunk
            batch[key] = (
                batch[key].chunk(local_world_size, dim=1)[local_rank].contiguous()
            )

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
            kwargs, self.original_seq_len, self.pad_len = self.apply_sequence_parallelism(batch=kwargs)
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
                            output[key] = value[:, :self.original_seq_len].contiguous()
            
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
        for i in range(len(self.hook_handles)):
            for handle in self.hook_handles[i]:
                handle.remove()
            self.hook_handles[i] = []

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
        # Prepare tensors for all_gather
        world_size = self.local_world_size
        requires_grad = tensor.requires_grad

        # Create list to store tensors from all ranks
        with torch.no_grad():
            # Each rank might have different sequence lengths based on logits_to_keep
            # So we need to gather shape information first
            local_shape = torch.tensor(list(tensor.shape), device=tensor.device)
            all_shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
            dist.all_gather(all_shapes, local_shape, group=self.process_group)
            all_seq_lens = [shape[1] for shape in all_shapes]
            
            # Create properly sized tensors for gathering
            gathered_tensors = []
            for i in range(world_size):
                # Create tensor with the same shape as the corresponding rank's tensor
                shape = [int(dim) for dim in all_shapes[i].tolist()]
                gathered_tensors.append(
                    torch.zeros(shape, dtype=tensor.dtype, device=tensor.device)
                )
            dist.all_gather(gathered_tensors, tensor.detach(), group=self.process_group)

        # If doesn't require gradient, just process normally
        if not requires_grad:
            return torch.cat(gathered_tensors, dim=1)

        # For tensors requiring gradients, use a custom autograd function
        rank = dist.get_rank(group=self.process_group)
        result = AllGatherWithGrad.apply(tensor, gathered_tensors, rank, all_seq_lens)

        return result


class AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, gathered_tensors, rank, all_seq_lens):
        ctx.rank = rank
        ctx.input_shape = input_tensor.shape
        ctx.all_seq_lens = all_seq_lens
        ctx.save_for_backward(input_tensor)

        return torch.cat(gathered_tensors, dim=1)

    @staticmethod
    def backward(ctx, grad_output):
        rank = ctx.rank
        input_shape = ctx.input_shape
        all_seq_lens = ctx.all_seq_lens

        # Simple sharding - each rank gets its corresponding section
        start_idx = sum(all_seq_lens[:rank])
        end_idx = start_idx + all_seq_lens[rank]
        grad_input = grad_output[:, start_idx:end_idx].contiguous()

        # Verify shape
        if grad_input.shape != input_shape:
            raise ValueError(
                f"Gradient shape mismatch in AllGatherWithGrad.backward: "
                f"got {grad_input.shape} but expected {input_shape}"
            )

        return grad_input, None, None, None, None
