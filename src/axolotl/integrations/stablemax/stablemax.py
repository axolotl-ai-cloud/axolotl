import torch
import torch.nn.functional as F

def stablemax_fn(x):
    """
    Numerically stable alternative to softmax.
    s(x) = x + 1 if x >= 0, else 1 / (1 - x)
    StableMax(x_i) = s(x_i) / sum_j s(x_j)
    """
    s = torch.where(x >= 0, x + 1, 1 / (1 - x))
    return s / s.sum(dim=-1, keepdim=True)

def stablemax_cross_entropy(input, target, reduction="mean"):
    """
    Cross-entropy loss using StableMax instead of softmax.
    Args:
        input: logits (batch_size, num_classes)
        target: target indices (batch_size,) or one-hot (batch_size, num_classes)
        reduction: 'mean' or 'sum'
    Returns:
        loss: scalar
    """
    probs = stablemax_fn(input)
    if target.dim() == input.dim():
        # one-hot targets
        log_probs = torch.log(probs + 1e-12)
        loss = -(target * log_probs).sum(dim=-1)
    else:
        # class indices
        log_probs = torch.log(probs + 1e-12)
        loss = -log_probs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
