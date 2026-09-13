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
        num_items_in_batch: for sample packing support
    """
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    shift_logits_view = shift_logits.view(-1, vocab_size)
    shift_labels_view = shift_labels.view(-1)

    mask = shift_labels_view != -100
    if not mask.any():
        return shift_logits_view.sum() * 0.0

    logprobs = F.log_softmax(shift_logits_view[mask].float(), dim=-1)
    per_token_logps = logprobs.gather(
        dim=-1, index=shift_labels_view[mask].unsqueeze(-1)
    ).squeeze(-1)

    per_token_loss = -per_token_logps.exp().detach() * per_token_logps

    if num_items_in_batch is not None:
        loss = per_token_loss.sum() / num_items_in_batch
    else:
        loss = per_token_loss.mean()

    return loss
