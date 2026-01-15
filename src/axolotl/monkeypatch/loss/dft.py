"""Dynamic Fine-Tuning (DFT) loss implementation"""

from typing import Optional

import torch
import torch.nn.functional as F


def selective_log_softmax(logits, index):
    """Memory-efficient log_softmax -> gather"""
    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)
        ).squeeze(-1)
        logsumexp_values = torch.stack([torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values
    else:
        per_token_logps = []
        for row_logits, row_labels in zip(logits, index, strict=True):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)
            ).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


def get_dft_loss(ignore_index: int = -100):
    """Creates DFT loss function"""

    def for_causal_lm_dft_loss(
        logits,
        labels,
        vocab_size: int = None,
        num_items_in_batch: Optional[int] = None,
        ignore_index: int = -100,
        shift_labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """DFT loss: -exp(logprobs).detach() * logprobs"""
        if shift_labels is None:
            # Shift so that tokens < n predict n
            labels = F.pad(labels, (0, 1), value=ignore_index)
            shift_labels = labels[..., 1:].contiguous()

        shift_labels = shift_labels.to(logits.device)

        # Create loss mask
        loss_mask = shift_labels != ignore_index
        shift_labels_masked = shift_labels.clone()
        shift_labels_masked[~loss_mask] = 0

        # Compute log probabilities
        logprobs = selective_log_softmax(logits, shift_labels_masked)

        # DFT loss: -exp(logprobs).detach() * logprobs
        per_token_loss = -logprobs.exp().detach() * logprobs

        # Sum over valid tokens and normalize
        if num_items_in_batch is None:
            num_items_in_batch = loss_mask.sum()

        loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
        return loss

    return for_causal_lm_dft_loss


def patch_dft_loss_fn(ignore_index: int = -100):
    """Patch transformers to use DFT loss for causal LM"""
    import transformers.loss.loss_utils

    for_causal_lm_dft_loss = get_dft_loss(ignore_index)
    transformers.loss.loss_utils.ForCausalLMLoss = for_causal_lm_dft_loss
    transformers.loss.loss_utils.LOSS_MAPPING["ForCausalLM"] = for_causal_lm_dft_loss
