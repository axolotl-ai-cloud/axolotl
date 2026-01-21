"""
eaft (entropy-aware focal training) loss implementation
weights examples by entropy approximation from top-k logits
"""

import torch
import torch.nn.functional as F


def eaft_loss(
    model, inputs, return_outputs=False, num_items_in_batch=None, alpha=1.0, k=20
):
    """
    compute eaft loss with entropy weighting

    args:
        model: the model being trained
        inputs: input batch
        return_outputs: whether to return model outputs
        num_items_in_batch: for sample packing support
        alpha: exponent for entropy weighting (default 1.0)
        k: number of top logits for entropy approximation (default 20)
    """
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    vocab_size = shift_logits.size(-1)
    shift_logits_view = shift_logits.view(-1, vocab_size)
    shift_labels_view = shift_labels.view(-1)

    mask = shift_labels_view != -100

    with torch.no_grad():
        top_k_logits, _ = torch.topk(
            shift_logits_view[mask].float(), k=min(k, vocab_size), dim=-1
        )
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        entropy = -(top_k_probs * torch.log(top_k_probs + 1e-10)).sum(dim=-1)
        weights = torch.pow(entropy, alpha)

    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    per_token_loss = loss_fct(shift_logits_view[mask], shift_labels_view[mask])
    weighted_loss = per_token_loss * weights

    if num_items_in_batch is not None:
        loss = weighted_loss.sum() / num_items_in_batch
    else:
        loss = weighted_loss.mean()

    return (loss, outputs) if return_outputs else loss
