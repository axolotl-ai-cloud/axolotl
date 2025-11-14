"""Attention instrumentation utilities for MuonClip."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:  # pragma: no cover
    from axolotl.muonclip.controller import MuonClipController

BUFFER_NAME = "muonclip_max_logits"
TRACKER_ATTR = "_muonclip_tracker"
SUPPORTED_ATTENTION_CLASSES = {
    "LlamaSdpaAttention",
    "LlamaAttention",
    "Qwen3Attention",
}


@dataclass
class AttentionTracker:
    name: str
    num_heads: int
    buffer_name: str = BUFFER_NAME
    active: bool = True


def register_attention_module(
    module: nn.Module,
    *,
    name: str,
    num_heads: int,
    device: Optional[torch.device] = None,
    buffer_name: str = BUFFER_NAME,
) -> AttentionTracker:
    """
    Attach a per-head max-logit buffer to an attention module.

    The module must call `record_attention_logits` with attention logits shaped
    like (batch, num_heads, seq_len, seq_len).
    """

    if hasattr(module, TRACKER_ATTR):
        tracker: AttentionTracker = getattr(module, TRACKER_ATTR)
        return tracker

    if device is None:
        try:
            device = next(module.parameters()).device
        except StopIteration:  # pragma: no cover - edge case for headless modules
            device = torch.device("cpu")

    buffer = torch.zeros(num_heads, device=device)
    module.register_buffer(buffer_name, buffer)
    tracker = AttentionTracker(name=name, num_heads=num_heads, buffer_name=buffer_name)
    setattr(module, TRACKER_ATTR, tracker)
    return tracker


def record_attention_logits(
    module: nn.Module,
    logits: torch.Tensor,
    *,
    buffer_name: str = BUFFER_NAME,
) -> None:
    """
    Update the per-head max logit buffer using the provided attention logits tensor.

    Args:
        module: attention module previously instrumented via `register_attention_module`.
        logits: tensor shaped (batch, num_heads, query_len, key_len) or (num_heads, seq, seq).
    """

    tracker: AttentionTracker | None = getattr(module, TRACKER_ATTR, None)
    if tracker is None or not tracker.active:
        return

    buffer: torch.Tensor = getattr(module, buffer_name)
    vals = logits.detach().float()
    if vals.dim() == 4:
        batch, heads, q_len, k_len = vals.shape
        vals = vals.reshape(batch, heads, q_len * k_len)
        max_per_head = vals.amax(dim=(0, 2))
    elif vals.dim() == 3:
        heads, q_len, k_len = vals.shape
        vals = vals.reshape(1, heads, q_len * k_len)
        max_per_head = vals.amax(dim=(0, 2))
    elif vals.dim() == 2:
        max_per_head = vals.amax(dim=0)
    elif vals.dim() == 1:
        max_per_head = vals
    else:  # pragma: no cover - unexpected shape
        raise ValueError(f"Unsupported logits shape for QK-Clip tracking: {vals.shape}")

    buffer.copy_(torch.maximum(buffer, max_per_head.to(buffer.dtype)))


def auto_register_llama_attention(
    model: nn.Module,
    controller: "MuonClipController",
) -> int:
    """
    Automatically attach trackers to HuggingFace Llama attention modules.
    """

    registered = 0
    for name, module in model.named_modules():
        if module.__class__.__name__ not in SUPPORTED_ATTENTION_CLASSES:
            continue
        num_heads = (
            getattr(module, "num_heads", None)
            or getattr(module, "num_attention_heads", None)
            or (getattr(module, "config", None) and getattr(module.config, "num_attention_heads", None))
        )
        if not num_heads:
            continue
        controller.register_attention(module, name=name, num_heads=int(num_heads))
        registered += 1
    return registered
