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

def stablemax_cross_entropy(input, target, weight=None, ignore_index=-100, 
                           size_average=None, reduce=None, reduction="mean", 
                           label_smoothing=0.0):
    """
    Cross-entropy loss using StableMax instead of softmax.
    Args:
        input: logits (batch_size, num_classes)
        target: target indices (batch_size,) or one-hot (batch_size, num_classes)
        weight: manual rescaling weight given to each class (num_classes,)
        ignore_index: specifies a target value that is ignored and does not contribute to the input gradient
        size_average: deprecated (kept for compatibility)
        reduce: deprecated (kept for compatibility)
        reduction: 'none' | 'mean' | 'sum'
        label_smoothing: label smoothing factor (0.0 to 1.0)
    Returns:
        loss: scalar or tensor depending on reduction
    """
    probs = stablemax_fn(input)
    log_probs = torch.log(probs + 1e-12)
    
    # Handle target format and create mask for ignore_index
    if target.dim() == input.dim():
        # one-hot targets
        targets_one_hot = target.float()
        # For one-hot targets, ignore_index doesn't apply directly
        valid_mask = torch.ones(target.shape[0], dtype=torch.bool, device=target.device)
    else:
        # class indices
        valid_mask = target != ignore_index
        # Convert to one-hot
        num_classes = input.shape[-1]
        targets_one_hot = torch.zeros_like(input)
        # Only set one-hot for valid targets
        valid_targets = target[valid_mask]
        if valid_targets.numel() > 0:
            targets_one_hot[valid_mask] = F.one_hot(valid_targets, num_classes).float()
    
    # Apply label smoothing
    if label_smoothing > 0.0:
        num_classes = input.shape[-1]
        uniform_dist = torch.ones_like(targets_one_hot) / num_classes
        targets_one_hot = (1.0 - label_smoothing) * targets_one_hot + label_smoothing * uniform_dist
    
    # Compute loss
    loss = -(targets_one_hot * log_probs).sum(dim=-1)
    
    # Apply class weights
    if weight is not None:
        if target.dim() == input.dim():
            # For one-hot targets, weight each class contribution
            class_weights = (targets_one_hot * weight.unsqueeze(0)).sum(dim=-1)
        else:
            # For class indices, use weight for each target class
            class_weights = torch.ones_like(loss)
            class_weights[valid_mask] = weight[target[valid_mask]]
        loss = loss * class_weights
    
    # Apply ignore_index mask
    if ignore_index != -100 or not valid_mask.all():
        loss = loss[valid_mask]
    
    # Apply reduction
    if reduction == "none":
        # For "none" reduction with ignored indices, we need to return full-size tensor
        if not valid_mask.all():
            full_loss = torch.zeros(valid_mask.shape[0], dtype=loss.dtype, device=loss.device)
            full_loss[valid_mask] = loss
            return full_loss
        return loss
    elif reduction == "mean":
        return loss.mean() if loss.numel() > 0 else torch.tensor(0.0, device=input.device)
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")
