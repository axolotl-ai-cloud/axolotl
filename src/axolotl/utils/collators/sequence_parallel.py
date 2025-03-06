"""Module for sequence parallelism data collators."""

from dataclasses import dataclass

import torch
import torch.distributed as dist
from accelerate.logging import get_logger

from axolotl.logging_config import configure_logging
from axolotl.monkeypatch.attention.ring_attn import get_ring_attn_group
from axolotl.utils.collators.batching import (
    BatchSamplerDataCollatorForSeq2Seq,
    DataCollatorForSeq2Seq,
    V2BatchSamplerDataCollatorForSeq2Seq,
)

configure_logging()
LOG = get_logger(__name__)


def find_sample_boundaries(position_ids):
    """
    Find the boundaries between packed samples in a sequence by looking for
    where position_ids decrease.

    Returns:
        List of boundary indices for each sequence in the batch
    """
    batch_boundaries = []

    for i in range(position_ids.shape[0]):
        seq = position_ids[i]
        boundaries = []
        for j in range(1, len(seq)):
            if seq[j] < seq[j - 1]:
                boundaries.append(j)
        batch_boundaries.append(boundaries)

    return batch_boundaries


def adjust_position_ids_for_slice(position_ids, start_idx):
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

        # Debug: log the found boundaries
        LOG.debug(f"Sequence {i}: Found sample boundaries at positions {boundaries}")

        # No need to adjust if there are no boundaries or this is a single sample
        if not boundaries:
            old_values = seq[0:5].tolist()  # Sample of original values
            adjusted_pos_ids[i] = seq - start_idx
            new_values = adjusted_pos_ids[i, 0:5].tolist()  # Sample of new values
            LOG.debug(
                f"Sequence {i}: No boundaries, subtracting {start_idx} uniformly. Example values before: {old_values}, after: {new_values}"
            )
            continue

        # Adjust each segment separately
        prev_boundary = 0
        for boundary_idx, boundary in enumerate(boundaries):
            segment = seq[prev_boundary:boundary]
            old_values = segment[
                0 : min(5, len(segment))
            ].tolist()  # Sample of original values
            adjusted_pos_ids[i, prev_boundary:boundary] -= start_idx
            new_values = adjusted_pos_ids[
                i, prev_boundary : min(prev_boundary + 5, boundary)
            ].tolist()  # Sample of new values
            LOG.debug(
                f"Sequence {i}, Segment {boundary_idx}: Adjusting positions {prev_boundary}-{boundary-1}. Example values before: {old_values}, after: {new_values}"
            )
            prev_boundary = boundary

        # Last segment
        segment = seq[prev_boundary:]
        old_values = segment[
            0 : min(5, len(segment))
        ].tolist()  # Sample of original values
        adjusted_pos_ids[i, prev_boundary:] -= start_idx
        new_values = adjusted_pos_ids[
            i, prev_boundary : min(prev_boundary + 5, len(seq))
        ].tolist()  # Sample of new values
        LOG.debug(
            f"Sequence {i}, Last segment: Adjusting positions {prev_boundary}-end. Example values before: {old_values}, after: {new_values}"
        )

    return adjusted_pos_ids


def check_for_boundary_splits(boundaries, slice_start, slice_end):
    """
    Check if any sample boundaries fall near the edge of a sequence slice.
    These edge cases could cause issues with gradient computation.

    Args:
        boundaries: List of indices where sample boundaries occur
        slice_start: Start index of this GPU's slice
        slice_end: End index of this GPU's slice

    Returns:
        List of potentially problematic boundaries
    """
    # Consider a boundary "near" an edge if it's within 5 tokens
    buffer_size = 5
    problem_boundaries = []

    for boundary in boundaries:
        # Check if boundary is near the start of the slice
        if slice_start <= boundary < slice_start + buffer_size:
            problem_boundaries.append((boundary, "start", boundary - slice_start))
        # Check if boundary is near the end of the slice
        elif slice_end - buffer_size <= boundary < slice_end:
            problem_boundaries.append((boundary, "end", slice_end - boundary))

    return problem_boundaries


@dataclass
class SequenceParallelPackedDataCollator(BatchSamplerDataCollatorForSeq2Seq):
    """
    Data collator for sequence parallelism with sample packing.
    Combines multiple samples into a packed sequence, then slices it for each GPU.
    """

    debug_level: str = "debug"  # Can be "debug" for more verbose output

    def __call__(self, features, return_tensors=None):
        # First, use the parent collator to handle sample packing and padding
        batch = super().__call__(features, return_tensors=return_tensors)

        sp_group = get_ring_attn_group()
        if sp_group is None:
            return batch  # Not using sequence parallelism

        # Get information about our position in the SP group
        rank = dist.get_rank(group=sp_group)
        world_size = dist.get_world_size(group=sp_group)

        # Enable debug level if requested
        if self.debug_level == "debug":
            original_shapes = {
                k: v.shape if hasattr(v, "shape") else None for k, v in batch.items()
            }
            LOG.info(f"GPU {rank}: Original batch shapes: {original_shapes}")

            if "position_ids" in batch:
                # Find and log sample boundaries before slicing
                boundaries = find_sample_boundaries(batch["position_ids"])
                for i, seq_boundaries in enumerate(boundaries):
                    LOG.info(
                        f"GPU {rank}: Sequence {i} has {len(seq_boundaries)} packed samples with boundaries at {seq_boundaries}"
                    )

        # Process keys that need to be sliced
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in batch:
                seq_len = batch[key].shape[1]
                slice_size = seq_len // world_size
                start_idx = rank * slice_size
                end_idx = start_idx + slice_size if rank < world_size - 1 else seq_len

                LOG.info(
                    f"GPU {rank}: Slicing {key} from {start_idx} to {end_idx} (total len: {seq_len})"
                )

                if self.debug_level == "debug" and key == "input_ids":
                    # Log portions of the input to verify correct slicing
                    for i in range(
                        min(2, batch[key].shape[0])
                    ):  # Look at up to 2 sequences
                        # Sample the beginning, middle and end of the sequence before slicing
                        start_sample = batch[key][i, 0:5].tolist()
                        mid_sample = batch[key][
                            i, seq_len // 2 : seq_len // 2 + 5
                        ].tolist()
                        end_sample = batch[key][i, -5:].tolist()
                        LOG.info(
                            f"GPU {rank}, Seq {i} before slicing: start={start_sample}, mid={mid_sample}, end={end_sample}"
                        )

                batch[key] = batch[key][:, start_idx:end_idx]

                if self.debug_level == "debug" and key == "input_ids":
                    # Log after slicing to verify
                    for i in range(min(2, batch[key].shape[0])):
                        sliced_sample = batch[key][i, 0:5].tolist()
                        sliced_end = batch[key][i, -5:].tolist()
                        LOG.info(
                            f"GPU {rank}, Seq {i} after slicing: start={sliced_sample}, end={sliced_end}"
                        )

        # Handle position_ids specially if present (important for packed sequences)
        if "position_ids" in batch:
            # For position_ids, we need to adjust them after slicing
            # Each position_id should be relative to its slice
            pos_ids = batch["position_ids"]
            seq_len = pos_ids.shape[1]
            slice_size = seq_len // world_size
            start_idx = rank * slice_size
            end_idx = start_idx + slice_size if rank < world_size - 1 else seq_len

            # Find boundaries before slicing
            if self.debug_level == "debug":
                full_boundaries = find_sample_boundaries(pos_ids)

                # Check for boundaries that fall near slice edges
                for i, boundaries in enumerate(full_boundaries):
                    problem_boundaries = check_for_boundary_splits(
                        boundaries, start_idx, end_idx
                    )
                    if problem_boundaries:
                        LOG.warning(
                            f"GPU {rank}: Sequence {i} has sample boundaries near slice edges: {problem_boundaries}"
                        )

            batch["position_ids"] = pos_ids[:, start_idx:end_idx]

            # Find boundaries after slicing to verify correct transfer
            if self.debug_level == "debug":
                sliced_boundaries = find_sample_boundaries(batch["position_ids"])
                for i, boundaries in enumerate(sliced_boundaries):
                    LOG.info(
                        f"GPU {rank}: After slicing, sequence {i} has boundaries at {boundaries}"
                    )

            # Adjust position_ids to be relative to the start of this slice
            # Only subtract if not the first GPU in the group
            if rank > 0:
                # Find boundaries between samples in the position_ids
                # This preserves the sample packing structure
                old_pos_ids = batch["position_ids"].clone()
                batch["position_ids"] = adjust_position_ids_for_slice(
                    batch["position_ids"], start_idx
                )

                if self.debug_level == "debug":
                    # Compare before and after adjustment
                    for i in range(min(2, old_pos_ids.shape[0])):
                        before = old_pos_ids[i, 0:10].tolist()
                        after = batch["position_ids"][i, 0:10].tolist()
                        LOG.info(
                            f"GPU {rank}, Seq {i} position_ids adjustment: before={before}, after={after}"
                        )

        # Add gradient norm tracking for debugging
        if self.debug_level == "debug":
            # Attach hook to track gradient norms during backward pass
            def hook_fn(grad):
                norm = grad.norm().item()
                LOG.info(f"GPU {rank}: Gradient norm = {norm:.4f}")
                # Record any abnormally high gradients
                if norm > 10.0:
                    LOG.warning(f"GPU {rank}: High gradient norm detected: {norm:.4f}")
                return grad

            # Apply hook to input_ids embeddings if it goes through backward pass
            if "input_ids" in batch and batch["input_ids"].requires_grad:
                batch["input_ids"].register_hook(hook_fn)

        return batch


@dataclass
class V2SequenceParallelPackedDataCollator(V2BatchSamplerDataCollatorForSeq2Seq):
    """
    Data collator for sequence parallelism with V2 sample packing.
    """

    debug_level: str = "debug"  # Can be "debug" for more verbose output

    def __call__(self, features, return_tensors=None):
        # Implementation similar to SequenceParallelPackedDataCollator with V2 base
        # First, use the parent collator to handle sample packing and padding
        batch = super().__call__(features, return_tensors=return_tensors)

        sp_group = get_ring_attn_group()
        if sp_group is None:
            return batch  # Not using sequence parallelism

        # Get information about our position in the SP group
        rank = dist.get_rank(group=sp_group)
        world_size = dist.get_world_size(group=sp_group)

        # Enable debug level if requested
        if self.debug_level == "debug":
            original_shapes = {
                k: v.shape if hasattr(v, "shape") else None for k, v in batch.items()
            }
            LOG.info(f"GPU {rank}: Original batch shapes: {original_shapes}")

            if "position_ids" in batch:
                # Find and log sample boundaries before slicing
                boundaries = find_sample_boundaries(batch["position_ids"])
                for i, seq_boundaries in enumerate(boundaries):
                    LOG.info(
                        f"GPU {rank}: Sequence {i} has {len(seq_boundaries)} packed samples with boundaries at {seq_boundaries}"
                    )

        # Process keys that need to be sliced
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in batch:
                seq_len = batch[key].shape[1]
                slice_size = seq_len // world_size
                start_idx = rank * slice_size
                end_idx = start_idx + slice_size if rank < world_size - 1 else seq_len

                if self.debug_level == "debug" and key == "input_ids":
                    # Log portions of the input to verify correct slicing
                    for i in range(
                        min(2, batch[key].shape[0])
                    ):  # Look at up to 2 sequences
                        # Sample the beginning, middle and end of the sequence before slicing
                        start_sample = batch[key][i, 0:5].tolist()
                        mid_sample = batch[key][
                            i, seq_len // 2 : seq_len // 2 + 5
                        ].tolist()
                        end_sample = batch[key][i, -5:].tolist()
                        LOG.info(
                            f"GPU {rank}, Seq {i} before slicing: start={start_sample}, mid={mid_sample}, end={end_sample}"
                        )

                batch[key] = batch[key][:, start_idx:end_idx]

        # Handle position_ids specially (same as in SequenceParallelPackedDataCollator)
        if "position_ids" in batch:
            # Implementation identical to the one in SequenceParallelPackedDataCollator
            pos_ids = batch["position_ids"]
            seq_len = pos_ids.shape[1]
            slice_size = seq_len // world_size
            start_idx = rank * slice_size
            end_idx = start_idx + slice_size if rank < world_size - 1 else seq_len

            # Find boundaries before slicing
            if self.debug_level == "debug":
                full_boundaries = find_sample_boundaries(pos_ids)

                # Check for boundaries that fall near slice edges
                for i, boundaries in enumerate(full_boundaries):
                    problem_boundaries = check_for_boundary_splits(
                        boundaries, start_idx, end_idx
                    )
                    if problem_boundaries:
                        LOG.warning(
                            f"GPU {rank}: Sequence {i} has sample boundaries near slice edges: {problem_boundaries}"
                        )

            batch["position_ids"] = pos_ids[:, start_idx:end_idx]

            # Adjust position_ids to be relative to the start of this slice
            if rank > 0:
                batch["position_ids"] = adjust_position_ids_for_slice(
                    batch["position_ids"], start_idx
                )

        return batch


@dataclass
class SequenceParallelDataCollator(DataCollatorForSeq2Seq):
    """
    Data collator for sequence parallelism without sample packing.
    """

    debug_level: str = "debug"  # Can be "debug" for more verbose output

    def __call__(self, features, return_tensors=None):
        # First, use the parent collator to pad everything correctly
        batch = super().__call__(features, return_tensors=return_tensors)

        sp_group = get_ring_attn_group()
        if sp_group is None:
            return batch  # Not using sequence parallelism

        # Get information about our position in the SP group
        rank = dist.get_rank(group=sp_group)
        world_size = dist.get_world_size(group=sp_group)

        if self.debug_level == "debug":
            original_shapes = {
                k: v.shape if hasattr(v, "shape") else None for k, v in batch.items()
            }
            LOG.info(f"GPU {rank}: Original batch shapes: {original_shapes}")

        # Process keys that need to be sliced
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in batch:
                seq_len = batch[key].shape[1]
                slice_size = seq_len // world_size
                start_idx = rank * slice_size
                end_idx = start_idx + slice_size if rank < world_size - 1 else seq_len

                LOG.info(
                    f"GPU {rank}: Slicing {key} from {start_idx} to {end_idx} (total len: {seq_len})"
                )
                batch[key] = batch[key][:, start_idx:end_idx]

        # Handle position_ids if present
        if "position_ids" in batch:
            pos_ids = batch["position_ids"]
            seq_len = pos_ids.shape[1]
            slice_size = seq_len // world_size
            start_idx = rank * slice_size
            end_idx = start_idx + slice_size if rank < world_size - 1 else seq_len

            batch["position_ids"] = pos_ids[:, start_idx:end_idx]

            # For non-packed sequences, we can simply subtract start_idx from all position_ids
            if rank > 0:
                batch["position_ids"] -= start_idx

        return batch
