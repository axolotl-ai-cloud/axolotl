"""
eaft (entropy-aware focal training) loss implementation
weights examples by entropy approximation from top-k logits

Reference: https://github.com/ymxyll/LlamaFactory-EAFT/blob/e2ce19e8efcc226450ee8f2b81dfe4e69f1f945d/src/llamafactory/train/trainer_utils.py
"""

import torch
import torch.nn.functional as F


def eaft_loss(outputs, labels, num_items_in_batch=None, alpha=1.0, k=20):
    """
    compute eaft loss with entropy weighting

    args:
        outputs: model outputs containing logits
        labels: target labels for computing loss
        num_items_in_batch: for sample packing support
        alpha: exponent for entropy weighting (default 1.0)
        k: number of top logits for entropy approximation (default 20)
    """
    logits = outputs.logits
    labels = labels.to(logits.device)

    vocab_size = logits.size(-1)
    shift_logits = logits[..., :-1, :].reshape(-1, vocab_size)
    shift_labels = labels[..., 1:].reshape(-1)
    # ignore_index instead of boolean-mask indexing, whose nonzero syncs with the host
    valid = shift_labels != -100

    with torch.no_grad():
        # upcasting is monotonic, so topk before .float() selects the same logits
        top_k_logits, _ = torch.topk(shift_logits, k=min(k, vocab_size), dim=-1)
        top_k_probs = F.softmax(top_k_logits.float(), dim=-1)
        entropy = -(top_k_probs * torch.log(top_k_probs + 1e-10)).sum(dim=-1)
        # zero on ignored rows so a non-finite weight there cannot poison the sum
        weights = torch.pow(entropy, alpha).masked_fill(~valid, 0)

    per_token_loss = F.cross_entropy(
        shift_logits, shift_labels, ignore_index=-100, reduction="none"
    )
    weighted_loss = per_token_loss * weights

    if num_items_in_batch is not None:
        return weighted_loss.sum() / num_items_in_batch
    return weighted_loss.sum() / valid.sum().clamp(min=1)
