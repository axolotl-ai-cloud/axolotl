from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist


@dataclass
class AuxFreeConfig:
    rate: float = 0.01
    momentum: float = 0.9
    bias_cap: float = 2.0
    warmup_steps: int = 0
    sync_group: str = "world"  # or "ep"


class AuxFreeState:
    """Holds per-layer bias and EMA load buffers."""

    def __init__(self, num_layers: int, num_experts: int, device: torch.device, cfg: AuxFreeConfig):
        self.bias = [torch.zeros(num_experts, device=device) for _ in range(num_layers)]
        self.ema_load = [torch.zeros(num_experts, device=device) for _ in range(num_layers)]
        self.cfg = cfg
        self.steps = 0


class AuxFreeShim:
    """Model-agnostic shim for aux-loss-free expert selection and bias updates."""

    def __init__(self, state: AuxFreeState, ep_group: Optional[dist.ProcessGroup] = None):
        self.state = state
        self.ep_group = ep_group

    @torch.no_grad()
    def select_experts(self, layer_idx: int, logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (topk_indices, weights) using biased selection and unbiased weights."""
        b = self.state.bias[layer_idx]
        biased = logits + b  # bias is a buffer
        topk_scores, topk_idx = torch.topk(biased, k=top_k, dim=-1)
        chosen_logits = torch.gather(logits, -1, topk_idx)
        weights = torch.softmax(chosen_logits.float(), dim=-1).to(logits.dtype)
        return topk_idx, weights

    @torch.no_grad()
    def all_reduce_counts(self, counts: torch.Tensor) -> torch.Tensor:
        if not dist.is_available() or not dist.is_initialized():
            return counts
        group = self.ep_group if self.ep_group is not None else dist.group.WORLD
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
        return counts

    @torch.no_grad()
    def update_bias(self, layer_idx: int, step_counts: torch.Tensor, tokens_seen: int):
        """Apply EMA-smoothed bias update toward uniform target, with clamp and optional mean-centering."""
        cfg = self.state.cfg
        self.state.steps += 1
        if self.state.steps <= cfg.warmup_steps:
            return

        nE = step_counts.numel()
        if tokens_seen <= 0:
            return
        freq = step_counts.float() / float(tokens_seen)
        ema = self.state.ema_load[layer_idx]
        ema.mul_(cfg.momentum).add_((1.0 - cfg.momentum) * freq)
        target = 1.0 / float(nE)
        delta = cfg.rate * (target - ema)
        # optional mean-centering to keep sum(bias) ~ 0
        delta = delta - delta.mean()
        bias = self.state.bias[layer_idx]
        bias.add_(delta)
        if cfg.bias_cap is not None and cfg.bias_cap > 0:
            bias.clamp_(-cfg.bias_cap, cfg.bias_cap)

