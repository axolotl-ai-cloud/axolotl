"""DFT (Dynamic Fine-Tuning) loss utilities."""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from axolotl.utils.logging import get_logger

from .chunked_ce import chunked_cross_entropy

LOG = get_logger(__name__)


def _get_context_parallel_group(trainer):
    """Get the context parallel group from the trainer."""
    try:
        if hasattr(trainer, "accelerator") and hasattr(
            trainer.accelerator, "context_parallel_group"
        ):
            return trainer.accelerator.context_parallel_group
    except AttributeError:
        pass
    return None


def compute_per_token_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    shift_labels: bool = True,
    chunk_size: Optional[int] = None,
    trainer=None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute per-token CE loss and the corresponding valid mask.

    This function is CP-aware: when Context Parallelism is enabled and logits
    are CP-local (not gathered), it correctly aligns labels with the local
    logits shard to compute boundary-correct losses.

    Args:
        logits: Model output logits, shape [batch, logits_seq_len, vocab_size].
               May be CP-local (logits_seq_len = full_seq_len / cp_size) or full.
        labels: Target labels, shape [batch, full_seq_len]. Always full sequence.
        ignore_index: Label value to ignore in loss computation.
        shift_labels: Whether to shift labels for causal language modeling.
        chunk_size: If provided and > 0, use chunked cross-entropy for memory efficiency.
        trainer: Optional trainer instance for CP group detection.

    Returns:
        per_token_loss: 1D tensor of shape [B * local_token_count] with grad.
        valid_mask: 1D bool tensor with True where label != ignore_index.

    CP Behavior:
        - Non-CP or CP-gathered: Standard causal shift [batch, T-1, vocab] → labels[:,1:]
        - CP-local: Each rank computes losses for its token shard's targets,
          with labels correctly sliced from the full label tensor.
    """
    batch_size, label_seq_len = labels.shape
    logits_seq_len = logits.size(1)

    # Detect CP environment
    cp_group = _get_context_parallel_group(trainer) if trainer is not None else None
    cp_enabled = cp_group is not None and dist.is_initialized()
    cp_size = dist.get_world_size(cp_group) if cp_enabled else 1
    cp_rank = dist.get_rank(cp_group) if cp_enabled else 0

    # Detect whether logits are CP-local (post-hook gather disabled) or already gathered
    is_cp_local_logits = False
    if cp_size > 1:
        # CP pads to divisor (min(cp_size, 64)) for Ring-Flash-Attention
        divisor = min(cp_size, 64)
        pad_len = (divisor - (label_seq_len % divisor)) % divisor
        expected_chunk_len = (label_seq_len + pad_len) // cp_size
        is_cp_local_logits = logits_seq_len == expected_chunk_len

        if dist.is_initialized():
            LOG.info(
                f"[DFT CP] Rank {dist.get_rank()}: logits_seq_len={logits_seq_len}, "
                f"label_seq_len={label_seq_len}, expected_chunk_len={expected_chunk_len}, "
                f"is_cp_local={is_cp_local_logits}, cp_rank={cp_rank}/{cp_size}"
            )

    if shift_labels:
        if cp_size > 1 and is_cp_local_logits:
            # CP-local path: compute boundary-correct losses for this shard
            #
            # For CP rank r with local token chunk [s, s+L), we compute losses for targets:
            #   tokens [s+1, s+L] (non-last ranks)
            #   tokens [s+1, s+L-1] (last rank; global last token has no target)
            #
            # This uses the full `labels` tensor and pads out-of-range tokens with -100.
            token_chunk_len = logits_seq_len
            local_token_start = cp_rank * token_chunk_len
            is_last_cp_rank = cp_rank == (cp_size - 1)

            token_count = token_chunk_len - (1 if is_last_cp_rank else 0)
            if token_count <= 0:
                # Edge case: empty shard
                shift_logits = logits[:, :0, :].contiguous()
                shift_labels = labels[:, :0].contiguous()
            else:
                # Prepare shift_logits
                shift_logits = (
                    logits[:, :-1, :].contiguous() if is_last_cp_rank else logits
                )

                # Prepare shift_labels by slicing from full labels
                label_start = local_token_start + 1
                label_end = label_start + token_count

                if label_start >= label_seq_len:
                    # All out-of-range: pad with ignore_index
                    shift_labels = torch.full(
                        (batch_size, token_count),
                        ignore_index,
                        dtype=labels.dtype,
                        device=labels.device,
                    )
                else:
                    slice_end = min(label_end, label_seq_len)
                    shift_labels = labels[:, label_start:slice_end]

                    # Pad if needed (partial overlap)
                    if shift_labels.size(1) < token_count:
                        pad = torch.full(
                            (batch_size, token_count - shift_labels.size(1)),
                            ignore_index,
                            dtype=labels.dtype,
                            device=labels.device,
                        )
                        shift_labels = torch.cat([shift_labels, pad], dim=1)

            # Verify alignment
            if shift_logits.size(1) != shift_labels.size(1):
                LOG.warning(
                    f"[DFT CP] Misalignment: shift_logits.shape={tuple(shift_logits.shape)}, "
                    f"shift_labels.shape={tuple(shift_labels.shape)} (cp_rank={cp_rank})"
                )
                min_len = min(shift_logits.size(1), shift_labels.size(1))
                shift_logits = shift_logits[:, :min_len, :].contiguous()
                shift_labels = shift_labels[:, :min_len].contiguous()
        else:
            # Non-CP or CP-gathered path: standard causal shift
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
    else:
        shift_logits = logits
        shift_labels = labels

    shift_logits = shift_logits.float()
    shift_labels = shift_labels.to(shift_logits.device)

    # Flatten tensors for loss computation
    logits_flat = shift_logits.view(-1, shift_logits.size(-1))
    labels_flat = shift_labels.reshape(-1)

    # Use chunked cross-entropy if chunk_size is specified
    if chunk_size is not None and chunk_size > 0:
        per_token_loss = chunked_cross_entropy(
            logits_flat,
            labels_flat,
            chunk_size=chunk_size,
            ignore_index=ignore_index,
        )
    else:
        per_token_loss = F.cross_entropy(
            logits_flat,
            labels_flat,
            ignore_index=ignore_index,
            reduction="none",
        )

    valid_mask = labels_flat != ignore_index
    return per_token_loss, valid_mask


def apply_dft_weighting(per_token_loss: torch.Tensor) -> torch.Tensor:
    """Apply DFT weighting without allowing gradients through the weight."""
    with torch.no_grad():
        weights = torch.exp(-per_token_loss)
    return per_token_loss * weights


def reduce_token_loss(
    per_token_loss: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    num_items_in_batch: int | None = None,
) -> torch.Tensor:
    """Reduce masked per-token loss to a scalar."""
    if num_items_in_batch is None:
        denom = valid_mask.sum()
    else:
        denom = per_token_loss.new_tensor(int(num_items_in_batch))

    if int(denom.item()) == 0:
        return per_token_loss.sum() * 0.0

    return per_token_loss[valid_mask].sum() / denom


def compute_dft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    shift_labels: bool = True,
    num_items_in_batch: int | None = None,
    chunk_size: Optional[int] = None,
    trainer=None,
) -> torch.Tensor:
    """Compute scalar DFT loss from logits + labels.

    This function is CP-aware when trainer is provided.

    Args:
        logits: Model output logits, shape [batch, seq_len, vocab_size].
        labels: Target labels, shape [batch, seq_len].
        ignore_index: Label value to ignore in loss computation.
        shift_labels: Whether to shift labels for causal language modeling.
        num_items_in_batch: Number of items for loss normalization.
        chunk_size: If provided and > 0, use chunked cross-entropy for memory efficiency.
        trainer: Optional trainer instance for CP detection.

    Returns:
        Scalar loss tensor.
    """
    per_token_loss, valid_mask = compute_per_token_cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
        shift_labels=shift_labels,
        chunk_size=chunk_size,
        trainer=trainer,
    )
    per_token_loss = apply_dft_weighting(per_token_loss)
    return reduce_token_loss(
        per_token_loss,
        valid_mask,
        num_items_in_batch=num_items_in_batch,
    )


def compute_dft_loss_with_intermediate(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    ignore_index: int = -100,
    shift_labels: bool = True,
    num_items_in_batch: int | None = None,
    chunk_size: Optional[int] = None,
    trainer=None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute DFT loss and return intermediate values for Channel Loss integration.

    This function is designed to support feature composition, particularly with
    Channel Loss tracking. It returns both the scalar loss and the intermediate
    per-token loss tensor, allowing downstream code to access per-token statistics.

    This function is CP-aware when trainer is provided.

    Args:
        logits: Model output logits, shape [batch, seq_len, vocab_size].
        labels: Target labels, shape [batch, seq_len].
        ignore_index: Label value to ignore in loss computation.
        shift_labels: Whether to shift labels for causal language modeling.
        num_items_in_batch: Number of items for loss normalization.
        chunk_size: If provided and > 0, use chunked cross-entropy for memory efficiency.
        trainer: Optional trainer instance for CP detection.

    Returns:
        Tuple of (scalar_loss, per_token_loss, valid_mask):
        - scalar_loss: Reduced scalar loss for backpropagation.
        - per_token_loss: DFT-weighted per-token loss, shape [N] where N = batch * (seq - 1).
          Useful for Channel Loss statistics or other per-token analysis.
        - valid_mask: Boolean mask indicating which tokens are valid (not ignore_index),
          shape [N] matching per_token_loss.

    Example:
        >>> scalar_loss, per_token_loss, mask = compute_dft_loss_with_intermediate(
        ...     logits, labels, chunk_size=2048, trainer=trainer
        ... )
        >>> # Use scalar_loss for backprop
        >>> scalar_loss.backward()
        >>> # Use per_token_loss for channel statistics
        >>> channel_losses = per_token_loss[channel_mask & mask].mean()
    """
    per_token_loss, valid_mask = compute_per_token_cross_entropy(
        logits,
        labels,
        ignore_index=ignore_index,
        shift_labels=shift_labels,
        chunk_size=chunk_size,
        trainer=trainer,
    )
    per_token_loss = apply_dft_weighting(per_token_loss)
    scalar_loss = reduce_token_loss(
        per_token_loss,
        valid_mask,
        num_items_in_batch=num_items_in_batch,
    )
    return scalar_loss, per_token_loss, valid_mask

