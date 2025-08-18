"""Diffusion LM loss function for integration with transformers LOSS_MAPPING."""

from typing import Optional

import torch
import torch.nn.functional as F


def ForDiffusionLMLoss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    vocab_size: int,
    config: Optional[dict] = None,
    inputs: Optional[dict] = None,
    model: Optional[torch.nn.Module] = None,
    **kwargs,
) -> torch.Tensor:
    """
    Diffusion Language Modeling loss function.

    This function computes cross-entropy loss only on masked tokens using
    diffusion info stored by the model patch during forward pass.

    Args:
        logits: Model predictions [batch_size, seq_len, vocab_size]
        labels: Ground truth tokens [batch_size, seq_len]
        vocab_size: Size of vocabulary
        config: Model configuration (contains diffusion parameters)
        inputs: Input batch dictionary (contains input_ids, attention_mask)
        model: The model instance (to access stored diffusion info)
        **kwargs: Additional arguments

    Returns:
        loss: Computed diffusion loss
    """
    # Get diffusion info stored by model patch
    if model is None or not hasattr(model, "_diffusion_info"):
        # Fallback to regular causal LM loss if no diffusion info
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = torch.nn.CrossEntropyLoss()
        return loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

    diffusion_info = model._diffusion_info
    original_input_ids = diffusion_info["original_input_ids"]
    masked_indices = diffusion_info["masked_indices"]
    p_mask = diffusion_info["p_mask"]

    # Get diffusion config parameters
    diffusion_config = getattr(config, "diffusion_config", {})
    importance_weighting = diffusion_config.get("importance_weighting", True)

    # Check if we have any masked tokens
    if not masked_indices.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Get predictions and targets for masked positions only
    masked_logits = logits[masked_indices]
    masked_targets = original_input_ids[masked_indices]  # Original unmasked tokens

    # Compute cross-entropy loss without reduction
    token_loss = F.cross_entropy(
        masked_logits.float(), masked_targets, reduction="none"
    )

    if importance_weighting:
        # Apply importance weighting: 1 / p_mask
        masked_p_mask = p_mask.expand_as(masked_indices)[masked_indices]
        weighted_loss = token_loss / masked_p_mask

        if labels is not None:
            # For SFT data: normalize by answer length per sample
            answer_mask = labels != -100
            answer_lengths = answer_mask.sum(dim=1).float()

            # Group losses by batch sample
            batch_indices = torch.arange(
                original_input_ids.shape[0], device=original_input_ids.device
            )
            batch_indices = batch_indices.unsqueeze(1).expand_as(masked_indices)
            masked_batch_indices = batch_indices[masked_indices]

            # Sum losses per sample and normalize by answer length
            loss_per_sample = torch.zeros(
                original_input_ids.shape[0], device=original_input_ids.device
            )
            for i in range(original_input_ids.shape[0]):
                sample_mask = masked_batch_indices == i
                if sample_mask.any():
                    sample_loss = weighted_loss[sample_mask].sum()
                    loss_per_sample[i] = sample_loss / max(answer_lengths[i], 1)

            loss = loss_per_sample.mean()
        else:
            # For completion data: simple average
            loss = weighted_loss.mean()
    else:
        # No importance weighting
        loss = token_loss.mean()

    return loss


def register_diffusion_loss():
    """Register the diffusion loss function in transformers LOSS_MAPPING."""
    try:
        from transformers.loss.loss_utils import LOSS_MAPPING

        LOSS_MAPPING["ForDiffusionLM"] = ForDiffusionLMLoss
        return True
    except ImportError:
        # Fallback for older transformers versions
        return False
