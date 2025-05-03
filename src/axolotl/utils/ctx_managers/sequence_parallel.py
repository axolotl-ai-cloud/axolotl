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
    # batch = {k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    original_seq_len = batch["input_ids"].size(1)

    # Update ring attention params if needed
    if batch.get("position_ids") is not None:
        update_ring_attn_params(position_ids=batch["position_ids"])
    else:
        # If position_ids aren't already in the batch, create them
        batch["position_ids"] = torch.arange(
            0,
            original_seq_len,
            dtype=torch.long,
            device=batch["input_ids"].device,
        ).expand(batch["input_ids"].size(0), -1)

    # Handle logits_to_keep
    if "logits_to_keep" in batch and isinstance(batch["logits_to_keep"], int):
        logits_to_keep = batch["logits_to_keep"]

        # Calculate which positions in the full sequence contain the last N tokens
        start_position = max(0, original_seq_len - logits_to_keep)
        chunk_size = original_seq_len // local_world_size
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
            tokens_in_rank = min(rank_end, original_seq_len) - max(
                rank_start, start_position
            )

            # Calculate where these tokens start in the local chunk
            local_start_idx = max(0, start_position - rank_start)

            # Set the appropriate positions in the mask to True
            mask[local_start_idx : local_start_idx + tokens_in_rank] = True

        # Replace the integer with the boolean mask
        batch["logits_to_keep"] = mask

    # Add padding to make sequence length divisible by local_world_size
    total_seq_len = original_seq_len
    pad_len = 0
    divisor = min(local_world_size, 64)
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

        # Split in sequential fashion and grab this rank's chunk
        if batch[key].size(1) == total_seq_len:
            batch[key] = (
                batch[key].chunk(local_world_size, dim=1)[local_rank].contiguous()
            )
        elif key == "logits_to_keep":
            batch[key] = (
                batch[key].chunk(local_world_size, dim=0)[local_rank].contiguous()
            )

    # Handle num_items_in_batch
    if "num_items_in_batch" in batch:
        batch["num_items_in_batch"] = (batch["labels"] != -100).sum()

    shapes = {}
    for k, v in batch.items():
        if len(v.shape) > 1:
            shapes[k] = v.shape
        else:
            shapes[k] = v.item()
    
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
            # shapes = {k: v.shape for k, v in kwargs.items()}
            # print(f"{dist.get_rank()}: before {shapes}")
            # print(f"{dist.get_rank()}: before {kwargs['attention_mask'].sum(1)}")
            # dist.barrier()
            kwargs, self.original_seq_len, self.pad_len = (
                self.apply_sequence_parallelism(batch=kwargs)
            )
            # shapes = {k: v.shape for k, v in kwargs.items()}
            # print(f"{dist.get_rank()}: after {shapes}")
            # print(f"{dist.get_rank()}: after {kwargs['attention_mask'].sum(1)}")
            # dist.barrier()
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
                output[key] = AllGatherWithGrad.apply(value, self.process_group)
            else:
                output[key] = AllReduceWithGrad.apply(value, self.process_group)

        return output


class AllGatherWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, group):
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        
        # Gather shape metadata
        local_shape = torch.tensor(list(input_tensor.shape), device=input_tensor.device)
        world_size = ctx.world_size
        all_shapes = [torch.zeros_like(local_shape) for _ in range(world_size)]
        dist.all_gather(all_shapes, local_shape, group=group)
        
        # Store sequence lengths for backward pass
        seq_lens = [int(shape[1].item()) for shape in all_shapes]
        ctx.seq_lens = seq_lens
        
        # Perform all_gather operation
        gathered = [
            torch.zeros(tuple(shape.tolist()), dtype=input_tensor.dtype, device=input_tensor.device)
            for shape in all_shapes
        ]
        dist.all_gather(gathered, input_tensor, group=group)
        
        # Concatenate tensors along sequence dimension
        result = torch.cat(gathered, dim=1)
        
        return result
    
    @staticmethod
    def backward(ctx, grad_output):
        # Unpack saved variables
        rank = ctx.rank
        seq_lens = ctx.seq_lens

        # Calculate offset for this rank's chunk
        offset = sum(seq_lens[:rank])

        # Extract gradient for this rank's chunk
        grad_slice = grad_output[:, offset:offset + seq_lens[rank]].contiguous()

        # No additional modifications or scaling necessary
        return grad_slice, None


class AllReduceWithGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_tensor, group):
        ctx.group = group
        ctx.had_nan = torch.isnan(input_tensor).any().item()

        safe_input = torch.where(
            torch.isnan(input_tensor), 
            torch.zeros_like(input_tensor), 
            input_tensor,
        )
        output = safe_input.clone()
        dist.all_reduce(output, op=dist.ReduceOp.AVG, group=group)

        ctx.save_for_backward(input_tensor)

        return output
        
    @staticmethod
    def backward(ctx, grad_output):
        input_tensor, = ctx.saved_tensors
        group = ctx.group
        world_size = dist.get_world_size(group)
        grad_input = grad_output / world_size

        if ctx.had_nan:
            grad_input = torch.where(
                torch.isnan(input_tensor),
                torch.zeros_like(grad_input),
                grad_input,
            )

        return grad_input, None
