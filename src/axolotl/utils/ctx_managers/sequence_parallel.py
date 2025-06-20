"""Module for Axolotl trainer sequence parallelism manager and utilities"""

import functools
import inspect

import torch
import torch.distributed as dist
from torch import nn
from torch.utils.hooks import RemovableHandle

from axolotl.monkeypatch.ring_attn import (
    get_ring_attn_group,
    patch_prepare_data_loader,
    patch_prepare_device_mesh,
    register_ring_attn,
    update_ring_attn_params,
)
from axolotl.utils.schemas.enums import RingAttnFunc


# TODO(djsaunde): implement zigzag, stripe patterns here (and elsewhere) in this
# module. Currently, we just focus on batch ring and varlen llama3 for simplicity.
def apply_sequence_parallelism(
    batch: dict[str, torch.Tensor],
    local_rank: int,
    local_world_size: int,
    gradient_accumulation_steps: int,
    ring_attn_func: RingAttnFunc,  # pylint: disable=unused-argument
) -> tuple[dict[str, torch.Tensor], int, int]:
    """
    Apply sequence parallelism slicing to a batch.

    Special handling is implemented for integer logits_to_keep, which indicates
    to only keep the last N tokens in the sequence during generation.

    Args:
        batch: Batch dictionary (e.g., input_ids, attention_mask, etc.).
        local_rank: Local rank in the sequence parallel group.
        local_world_size: World size of the sequence parallel group.
        gradient_accumulation_steps: Number of steps to accumulate gradients over.
        ring_attn_func: Which ring attention function to use. Currently unused, but
            related to above TODO.

    Returns:
        tuple of:
            - Batch dictionary with sliced tensors.
            - The original sequence length before padding.
            - The number of padding tokens added.
    """
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

    # Slice batch for sequence parallel
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
            # Approximation; this needed since num_items_in_batch may be counted across
            # all samples in a gradient accumulated batch, not on a per-step basis.
            batch["num_items_in_batch"] = (
                batch["labels"] != -100
            ).sum() * gradient_accumulation_steps

    return batch, original_seq_len, pad_len


class SequenceParallelContextManager:
    """Context manager for sequence parallelism operations.

    This class provides a context that will automatically apply sequence parallelism
    during model forward passes using a pre-forward hook, and gather outputs from
    across the sequence parallelism group using a post-forward hook.

    Args:
        models: List of models to apply sequence parallelism to pre- and post- forward
            hooks.
        sequence_parallel_degree: Number of processes to split sequences over.
        gradient_accumulation_steps: Number of steps to accumulate gradients over.
        ring_attn_func: Which ring attention function to use. Currently unused.
        heads_k_stride: Sequence parallelism K head stride size. Passed through to
            `varlen_llama3` `ring_flash_attn` implementation.
    """

    def __init__(
        self,
        models: list[nn.Module],
        sequence_parallel_degree: int,
        gradient_accumulation_steps: int,
        ring_attn_func: RingAttnFunc,
        heads_k_stride: int | None,
    ):
        self.models = models
        self.sequence_parallel_degree = sequence_parallel_degree
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.ring_attn_func = ring_attn_func
        self.heads_k_stride = heads_k_stride
        self._register_ring_attn()

        # Set distributed info for local rank
        self.process_group = get_ring_attn_group()
        self.local_rank = dist.get_rank(self.process_group)
        self.local_world_size = dist.get_world_size(self.process_group)

        # Will store hook handles for removal
        self.hook_handles: list[RemovableHandle] = []

        # Store original sequence length and padding information
        self.original_seq_len = 0
        self.pad_len = 0

        # Create a partially applied version of the apply_sequence_parallelism function
        self.apply_sequence_parallelism = functools.partial(
            apply_sequence_parallelism,
            local_rank=self.local_rank,
            local_world_size=self.local_world_size,
            gradient_accumulation_steps=self.gradient_accumulation_steps,
            ring_attn_func=self.ring_attn_func,
        )

    def __enter__(self):
        self._register_model_hooks()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []

        # TODO(djsaunde): Un-patch attention and accelerate functions (low priority)

    def _register_ring_attn(self):
        # Initialize ring attn for sequence parallelism
        register_ring_attn(
            sequence_parallel_degree=self.sequence_parallel_degree,
            heads_k_stride=self.heads_k_stride,
            ring_attn_func=self.ring_attn_func,
        )

        # Patches for accelerate functionality
        patch_prepare_data_loader()
        patch_prepare_device_mesh(
            sequence_parallel_degree=self.sequence_parallel_degree
        )

    def _register_model_hooks(self):
        # Forward pre-hook to apply sequence parallelism
        def sequence_parallel_pre_hook(_, args, kwargs):
            # Get parameter names from the model's forward function
            forward_params = list(
                inspect.signature(self.models[0].forward).parameters.keys()
            )

            updated_kwargs = kwargs.copy()
            for i, arg in enumerate(args):
                if i < len(forward_params):
                    updated_kwargs[forward_params[i]] = arg

            # Any excess positional arguments are kept as-is
            remaining_args = args[len(forward_params) :]

            # Apply sequence parallelism to updated kwargs
            updated_kwargs, self.original_seq_len, self.pad_len = (
                self.apply_sequence_parallelism(updated_kwargs)
            )

            return remaining_args, updated_kwargs

        # Register hooks
        for model in self.models:
            self.hook_handles.append(
                model.register_forward_pre_hook(
                    sequence_parallel_pre_hook, with_kwargs=True
                )
            )
