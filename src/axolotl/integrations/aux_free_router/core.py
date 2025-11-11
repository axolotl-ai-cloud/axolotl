from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


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

    def __init__(
        self,
        state: AuxFreeState,
        ep_group: Optional[dist.ProcessGroup] = None,
        ep_size: Optional[int] = None,
    ):
        self.state = state
        self.ep_group = ep_group
        self._ep_size = ep_size
        self._ep_group_pending = (
            self.state.cfg.sync_group == "ep" and self.ep_group is None
        )
        self._layer_modules: dict[int, torch.nn.Module] = {}
        self._prev_bias_sign: dict[int, torch.Tensor] = {}

    @torch.no_grad()
    def select_experts(self, layer_idx: int, logits: torch.Tensor, top_k: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (topk_indices, weights) using biased selection and unbiased weights."""
        module = self._layer_modules.get(layer_idx)
        if module is not None and hasattr(module, "_afb_bias"):
            b = getattr(module, "_afb_bias")
        else:
            b = self.state.bias[layer_idx]
        biased = logits + b  # bias is a buffer
        topk_scores, topk_idx = torch.topk(biased, k=top_k, dim=-1)
        chosen_logits = torch.gather(logits, -1, topk_idx)
        weights = torch.softmax(chosen_logits.float(), dim=-1).to(logits.dtype)
        return topk_idx, weights

    def register_layer_buffers(self, layer_idx: int, module: torch.nn.Module) -> None:
        """Bind model buffers so shim updates stay in sync with patched layers."""
        self._layer_modules[layer_idx] = module
        bias = getattr(module, "_afb_bias")
        ema = getattr(module, "_afb_ema")
        # Keep state views pointing to the same tensors to avoid drift.
        if layer_idx < len(self.state.bias):
            self.state.bias[layer_idx] = bias
        if layer_idx < len(self.state.ema_load):
            self.state.ema_load[layer_idx] = ema

    def begin_step(self) -> None:
        """Call once per optimizer step before per-layer updates."""
        self.state.steps += 1

    def get_prev_bias_sign(self, layer_idx: int) -> Optional[torch.Tensor]:
        return self._prev_bias_sign.get(layer_idx)

    @torch.no_grad()
    def all_reduce_counts(self, counts: torch.Tensor) -> torch.Tensor:
        self._maybe_init_ep_group()
        if not dist.is_available() or not dist.is_initialized():
            return counts
        group = self.ep_group if self.ep_group is not None else dist.group.WORLD
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)
        return counts

    @torch.no_grad()
    def update_bias(self, layer_idx: int, step_counts: torch.Tensor, tokens_seen: int):
        """Apply EMA-smoothed bias update toward uniform target, with clamp and optional mean-centering."""
        cfg = self.state.cfg
        if self.state.steps <= cfg.warmup_steps:
            return

        nE = step_counts.numel()
        if tokens_seen <= 0:
            return
        module = self._layer_modules.get(layer_idx)
        if module is not None and hasattr(module, "_afb_ema"):
            ema = getattr(module, "_afb_ema")
            bias = getattr(module, "_afb_bias")
        else:
            ema = self.state.ema_load[layer_idx]
            bias = self.state.bias[layer_idx]
        counts = step_counts.to(ema.device)
        freq = counts.float() / float(tokens_seen)
        ema.mul_(cfg.momentum).add_((1.0 - cfg.momentum) * freq)
        target = 1.0 / float(nE)
        delta = cfg.rate * (target - ema)
        # optional mean-centering to keep sum(bias) ~ 0
        delta = delta - delta.mean()
        bias.add_(delta)
        if cfg.bias_cap is not None and cfg.bias_cap > 0:
            bias.clamp_(-cfg.bias_cap, cfg.bias_cap)
        self._prev_bias_sign[layer_idx] = torch.sign(bias.detach())

    def _maybe_init_ep_group(self) -> None:
        if not self._ep_group_pending:
            return
        if not dist.is_available() or not dist.is_initialized():
            return
        ep_size = self._ep_size
        if not ep_size or ep_size <= 1:
            LOG.warning(
                "AuxFreeMoE: moe_bias_sync_group='ep' requested but expert_parallel_size<=1; defaulting to world group"
            )
            self.ep_group = dist.group.WORLD
            self._ep_group_pending = False
            return
        world = dist.get_world_size()
        if world % ep_size != 0:
            LOG.warning(
                "AuxFreeMoE: expert_parallel_size %s does not divide world size %s; defaulting to world group",
                ep_size,
                world,
            )
            self.ep_group = dist.group.WORLD
            self._ep_group_pending = False
            return
        if ep_size == world:
            self.ep_group = dist.group.WORLD
        else:
            rank = dist.get_rank()
            group_start = (rank // ep_size) * ep_size
            ranks = tuple(range(group_start, group_start + ep_size))
            self.ep_group = dist.new_group(ranks)
        LOG.info(
            "AuxFreeMoE: initialized expert-parallel reduction group (size=%s, world=%s)",
            ep_size,
            dist.get_world_size(),
        )
        self._ep_group_pending = False
