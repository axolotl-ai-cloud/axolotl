"""
dft (dynamic fine-tuning) loss implementation
weights each token's cross entropy by its own detached probability

Reference: https://arxiv.org/abs/2508.05629
"""

import torch.nn.functional as F


def dft_loss(outputs, labels, num_items_in_batch=None):
    """
    compute dft loss: -sg[p(y_t)] * log p(y_t)

    args:
        outputs: model outputs containing logits
        labels: target labels for computing loss
        num_items_in_batch: unmasked token count across gradient accumulation steps
    """
    logits = outputs.logits
    labels = labels.to(logits.device)

    vocab_size = logits.size(-1)
    shift_logits = logits[..., :-1, :].float().reshape(-1, vocab_size)
    shift_labels = labels[..., 1:].reshape(-1)

    # ignore_index instead of boolean-mask indexing, whose nonzero syncs with the host;
    # ignored positions get logp 0 and therefore zero loss and zero gradient.
    per_token_logps = -F.cross_entropy(
        shift_logits, shift_labels, ignore_index=-100, reduction="none"
    )

    per_token_loss = -per_token_logps.exp().detach() * per_token_logps

    if num_items_in_batch is None:
        num_items_in_batch = (shift_labels != -100).sum().clamp(min=1)

    return per_token_loss.sum() / num_items_in_batch
