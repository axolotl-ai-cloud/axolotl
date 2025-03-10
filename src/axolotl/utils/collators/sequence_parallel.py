"""Module for sequence parallelism data collators."""

import logging
from dataclasses import dataclass

import torch
import torch.distributed as dist

from axolotl.monkeypatch.attention.ring_attn import get_ring_attn_group
from axolotl.utils.collators.batching import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    V2BatchSamplerDataCollatorForSeq2Seq,
)

logger = logging.getLogger(__name__)


def adjust_position_ids_for_slice(
    position_ids: list | torch.Tensor, start_idx: int
) -> torch.Tensor:
    """
    Adjust position IDs for a sliced sequence to maintain proper relative positions.
    This handles the case where position IDs might not be contiguous due to sample packing.
    """
    # Convert to tensor if not already
    if not isinstance(position_ids, torch.Tensor):
        position_ids = torch.tensor(
            position_ids,
            device=position_ids.device if hasattr(position_ids, "device") else None,
        )

    # Find the boundaries between samples (where position_ids reset)
    adjusted_pos_ids = position_ids.clone()

    # Process each sequence in the batch
    for i in range(position_ids.shape[0]):
        seq = position_ids[i]

        # Find sample boundaries
        boundaries = []
        for j in range(1, len(seq)):
            if seq[j] < seq[j - 1]:
                boundaries.append(j)

        # No need to adjust if there are no boundaries or this is a single sample
        if not boundaries:
            adjusted_pos_ids[i] = seq - start_idx
            continue

        # Adjust each segment separately
        prev_boundary = 0
        for boundary in boundaries:
            adjusted_pos_ids[i, prev_boundary:boundary] -= start_idx
            prev_boundary = boundary

        # Last segment
        adjusted_pos_ids[i, prev_boundary:] -= start_idx

    return adjusted_pos_ids


class SequenceParallelMixin:
    """
    Mixin to add sequence parallelism slicing to data collators.
    """

    def __post_init__(self):
        # Get information about our position in the SP group
        sp_group = get_ring_attn_group()
        self.rank = dist.get_rank(group=sp_group)
        self.world_size = dist.get_world_size(group=sp_group)

    def apply_sequence_parallelism(
        self, batch: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Apply sequence parallelism slicing to a batch.

        Args:
            batch: Batch dictionary from parent collator.

        Returns:
            Sliced batch dictionary.
        """
        # Process keys that need to be sliced
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in batch:
                seq_len = batch[key].shape[1]
                slice_size = seq_len // self.world_size
                start_idx = self.rank * slice_size
                end_idx = (
                    start_idx + slice_size
                    if self.rank < self.world_size - 1
                    else seq_len
                )

                if key == "input_ids":
                    # Before slicing
                    non_pad_tokens_total = (batch["input_ids"] != 128001).sum().item()
                    logger.info(
                        f"GPU {self.rank}: Total sequence length: {seq_len}, "
                        f"Non-padding tokens: {non_pad_tokens_total}"
                    )
                    logger.info(f"GPU {self.rank} token IDs: {batch['input_ids']}")

                    # After slicing
                    non_pad_tokens_slice = (batch["input_ids"] != 128001).sum().item()
                    logger.info(
                        f"GPU {self.rank}: Slice {start_idx}-{end_idx}, "
                        f"Non-padding tokens in slice: {non_pad_tokens_slice}"
                    )

                dist.barrier()

                batch[key] = batch[key][:, start_idx:end_idx]

        # Handle position_ids if present
        if "position_ids" in batch:
            pos_ids = batch["position_ids"]
            seq_len = pos_ids.shape[1]
            slice_size = seq_len // self.world_size
            start_idx = self.rank * slice_size
            end_idx = (
                start_idx + slice_size if self.rank < self.world_size - 1 else seq_len
            )

            batch["position_ids"] = pos_ids[:, start_idx:end_idx]

            # Adjust position_ids to be relative to the slice start
            if self.rank > 0:
                batch["position_ids"] = adjust_position_ids_for_slice(
                    batch["position_ids"], start_idx
                )

        return batch


@dataclass
class SequenceParallelPackedDataCollator(
    SequenceParallelMixin, BatchSamplerDataCollatorForSeq2Seq
):
    """
    Data collator for sequence parallelism with sample packing. Combines multiple
    samples into a packed sequence, then slices it for each GPU.
    """

    def __call__(self, features, return_tensors=None):
        # Use the parent collator to handle sample packing and padding
        batch = super().__call__(features, return_tensors=return_tensors)
        return self.apply_sequence_parallelism(batch)


@dataclass
class V2SequenceParallelPackedDataCollator(
    SequenceParallelMixin, V2BatchSamplerDataCollatorForSeq2Seq
):
    """
    Data collator for sequence parallelism with V2 sample packing.
    """

    def __call__(self, features, return_tensors=None):
        # Use the parent collator to handle sample packing and padding
        batch = super().__call__(features, return_tensors=return_tensors)
        return self.apply_sequence_parallelism(batch)


@dataclass
class SequenceParallelDataCollator(SequenceParallelMixin, DataCollatorForSeq2Seq):
    """
    Data collator for sequence parallelism without sample packing.
    """

    def __call__(self, features, return_tensors=None):
        # Use the parent collator to pad everything correctly
        batch = super().__call__(features, return_tensors=return_tensors)
        return self.apply_sequence_parallelism(batch)
