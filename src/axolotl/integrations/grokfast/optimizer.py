# Copyright: MIT License (c) 2024 Jaerin Lee, Bong Gyun Kang, Kihoon Kim, Kyoung Mu Lee
# Reference: https://github.com/ironjr/grokfast

# pylint: skip-file
from collections import deque
from typing import Dict, Literal, Optional

import torch
import torch.nn as nn


def gradfilter_ma(
    m: nn.Module,
    grads: Optional[Dict[str, deque]] = None,
    window_size: int = 100,
    lamb: float = 5.0,
    filter_type: Literal["mean", "sum"] = "mean",
    warmup: bool = True,
    trigger: bool = False,  # For ablation study.
) -> Dict[str, deque]:
    if grads is None:
        grads = {
            n: deque(maxlen=window_size)
            for n, p in m.named_parameters()
            if p.requires_grad and p.grad is not None
        }

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n].append(p.grad.data.detach())  # .cpu())

            # Modify the gradients.
            if not warmup or len(grads[n]) == window_size and not trigger:
                if filter_type == "mean":
                    avg = sum(grads[n]) / len(grads[n])
                elif filter_type == "sum":
                    avg = sum(grads[n])
                else:
                    raise ValueError(f"Unrecognized filter_type {filter_type}")
                p.grad.data = p.grad.data + avg * lamb

    return grads


def gradfilter_ema(
    m: nn.Module,
    grads: Optional[Dict[str, torch.Tensor]] = None,
    alpha: float = 0.98,
    lamb: float = 2.0,
) -> Dict[str, torch.Tensor]:
    if grads is None:
        grads = {
            n: p.grad.data.detach()
            for n, p in m.named_parameters()
            if p.requires_grad and p.grad is not None
        }

    for n, p in m.named_parameters():
        if p.requires_grad and p.grad is not None:
            grads[n] = grads[n] * alpha + p.grad.data.detach() * (1 - alpha)
            p.grad.data = p.grad.data + grads[n] * lamb

    return grads
