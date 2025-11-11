from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F

from axolotl.utils.logging import get_logger

from .core import AuxFreeShim

LOG = get_logger(__name__)


@dataclass
class LayerHandle:
    layer: nn.Module
    layer_idx: int
    num_experts: int
    top_k: int


class BaseMoEAdapter:
    """Base adapter that discovers MoE layers and wraps their forward.

    Concrete adapters should implement discovery and per-layer attribute extraction.
    """

    family: str = "generic"

    def matches(self, model: nn.Module) -> bool:  # pragma: no cover - thin shim
        return False

    def find_moe_layers(self, model: nn.Module) -> Iterable[nn.Module]:  # pragma: no cover
        return []

    def get_top_k(self, moe_layer: nn.Module) -> int:  # pragma: no cover
        return int(getattr(moe_layer, "num_experts_per_tok", getattr(moe_layer, "top_k", 2)))

    def get_num_experts(self, moe_layer: nn.Module) -> int:  # pragma: no cover
        return int(getattr(moe_layer, "num_experts"))

    def disable_aux_loss(self, model_or_layer: nn.Module) -> None:
        # Best-effort: zero router aux loss coef if present
        if hasattr(model_or_layer, "router_aux_loss_coef"):
            try:
                setattr(model_or_layer, "router_aux_loss_coef", 0.0)
            except Exception:  # pragma: no cover - non-critical
                pass

    def _register_aux_buffers(self, moe_layer: nn.Module, handle: LayerHandle, shim: AuxFreeShim) -> None:
        device = next(moe_layer.parameters(), torch.tensor(0)).device
        if not hasattr(moe_layer, "_afb_bias"):
            moe_layer.register_buffer("_afb_bias", torch.zeros(handle.num_experts, device=device))
        if not hasattr(moe_layer, "_afb_counts"):
            moe_layer.register_buffer("_afb_counts", torch.zeros(handle.num_experts, device=device))
        if not hasattr(moe_layer, "_afb_ema"):
            moe_layer.register_buffer("_afb_ema", torch.zeros(handle.num_experts, device=device))
        moe_layer._afb_layer_idx = handle.layer_idx  # type: ignore[attr-defined]
        moe_layer._afb_top_k = handle.top_k  # type: ignore[attr-defined]
        shim.register_layer_buffers(handle.layer_idx, moe_layer)

    def prepare(self, moe_layer: nn.Module, handle: LayerHandle, shim: AuxFreeShim) -> None:
        """Attach per-layer buffers and mark as aux-free enabled."""
        self._register_aux_buffers(moe_layer, handle, shim)
        self._patch_forward_with_aux_free(moe_layer)

    def _patch_forward_with_aux_free(self, moe_layer: nn.Module) -> None:
        """Replace the layer's forward with an aux-free gating version.

        Assumes the layer exposes attributes:
          - gate: linear router projecting hidden to num_experts
          - num_experts: int
          - experts: iterable of expert modules taking (tokens, H) -> (tokens, H)
        """
        if getattr(moe_layer, "_afb_patched", False):
            return

        if not hasattr(moe_layer, "gate") or not hasattr(moe_layer, "experts"):
            LOG.info("AuxFreeMoE: layer missing gate/experts; skipping forward patch")
            return

        def afb_forward(self, hidden_states: torch.Tensor):  # type: ignore[no-redef]
            # hidden_states: (B, T, H)
            bsz, seqlen, hdim = hidden_states.shape
            hs = hidden_states.view(-1, hdim)
            logits = self.gate(hs)
            # selection uses biased logits; weights from unbiased logits
            bias = getattr(self, "_afb_bias")
            top_k = int(getattr(self, "_afb_top_k", 2))
            biased = logits + bias  # broadcast over tokens
            topk_vals, topk_idx = torch.topk(biased, k=top_k, dim=-1, sorted=False)
            chosen_logits = torch.gather(logits, -1, topk_idx)
            weights = torch.softmax(chosen_logits.float(), dim=-1)
            weights = weights.to(hs.dtype)

            # accumulate counts for bias update callback
            flat_idx = topk_idx.reshape(-1)
            counts = torch.bincount(flat_idx, minlength=int(self.num_experts))
            getattr(self, "_afb_counts").add_(counts.to(getattr(self, "_afb_counts").dtype))

            # dispatch tokens to experts
            hs_rep = hs.repeat_interleave(top_k, dim=0)
            y = torch.empty_like(hs_rep)
            for eid in range(int(self.num_experts)):
                mask = flat_idx == eid
                if mask.any():
                    y[mask] = self.experts[eid](hs_rep[mask])

            y = (y.view(-1, top_k, hdim) * weights.unsqueeze(-1)).sum(dim=1)
            out = y.view(bsz, seqlen, hdim)
            return (out, logits)

        moe_layer.forward = afb_forward.__get__(moe_layer, moe_layer.__class__)  # type: ignore[attr-defined]
        setattr(moe_layer, "_afb_patched", True)


class MixtralAdapter(BaseMoEAdapter):
    family = "mixtral"

    def matches(self, model: nn.Module) -> bool:
        return getattr(getattr(model, "config", object()), "model_type", "") == "mixtral"

    def prepare(self, moe_layer: nn.Module, handle: LayerHandle, shim: AuxFreeShim) -> None:
        self._register_aux_buffers(moe_layer, handle, shim)
        self._patch_mixtral_forward(moe_layer, shim)

    def find_moe_layers(self, model: nn.Module) -> Iterable[nn.Module]:
        for m in model.modules():
            if m.__class__.__name__.endswith("SparseMoeBlock"):
                yield m

    def _patch_mixtral_forward(self, moe_layer: nn.Module, shim: AuxFreeShim) -> None:
        if getattr(moe_layer, "_afb_patched", False):
            return

        shim_ref = shim

        def afb_forward(self, hidden_states: torch.Tensor):  # type: ignore[no-redef]
            batch_size, sequence_length, hidden_dim = hidden_states.shape
            if self.training and getattr(self, "jitter_noise", 0) > 0:
                hidden_states = hidden_states * torch.empty_like(hidden_states).uniform_(
                    1.0 - self.jitter_noise, 1.0 + self.jitter_noise
                )
            flat_states = hidden_states.view(-1, hidden_dim)
            router_logits = self.gate(flat_states)

            layer_idx = int(getattr(self, "_afb_layer_idx", 0))
            top_k = int(getattr(self, "_afb_top_k", self.top_k))
            selected_experts, routing_weights = shim_ref.select_experts(layer_idx, router_logits, top_k)
            routing_weights = routing_weights.to(flat_states.dtype)

            flat_idx = selected_experts.reshape(-1)
            counts = torch.bincount(flat_idx, minlength=int(self.num_experts))
            self._afb_counts.add_(counts.to(self._afb_counts.dtype))

            final_hidden_states = torch.zeros(
                (batch_size * sequence_length, hidden_dim),
                dtype=flat_states.dtype,
                device=flat_states.device,
            )

            expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()
            for expert_idx_tensor in expert_hit:
                expert_idx = int(expert_idx_tensor.squeeze().item())
                expert_layer = self.experts[expert_idx]
                mask = expert_mask[expert_idx].squeeze(0)
                idx, top_x = torch.where(mask)
                current_state = flat_states[None, top_x].reshape(-1, hidden_dim)
                current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]
                final_hidden_states.index_add_(0, top_x, current_hidden_states.to(flat_states.dtype))

            final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
            return final_hidden_states, router_logits

        moe_layer.forward = afb_forward.__get__(moe_layer, moe_layer.__class__)  # type: ignore[attr-defined]
        setattr(moe_layer, "_afb_patched", True)


class Qwen3Adapter(MixtralAdapter):
    family = "qwen3_moe"

    def matches(self, model: nn.Module) -> bool:
        return getattr(getattr(model, "config", object()), "model_type", "") in ("qwen3_moe", "qwen2_moe")


class BailingAdapter(BaseMoEAdapter):
    family = "bailing_moe"

    def matches(self, model: nn.Module) -> bool:
        cfg = getattr(model, "config", None)
        if cfg is None:
            return False
        model_type = getattr(cfg, "model_type", "") or ""
        if model_type in ("bailing_moe", "bailing_moe_v2", "ring_moe", "ring"):
            return True
        cfg_name = cfg.__class__.__name__.lower()
        return "bailingmoev2" in cfg_name or "ring" in cfg_name

    def find_moe_layers(self, model: nn.Module) -> Iterable[nn.Module]:
        for m in model.modules():
            if m.__class__.__name__ == "BailingMoeV2SparseMoeBlock":
                yield m

    def get_num_experts(self, moe_layer: nn.Module) -> int:
        if hasattr(moe_layer, "num_experts"):
            return int(getattr(moe_layer, "num_experts"))
        cfg = getattr(moe_layer, "config", None)
        return int(getattr(cfg, "num_experts"))

    def prepare(self, moe_layer: nn.Module, handle: LayerHandle, shim: AuxFreeShim) -> None:
        self._register_aux_buffers(moe_layer, handle, shim)
        self._patch_bailing_gate(moe_layer)

    def _patch_bailing_gate(self, moe_layer: nn.Module) -> None:
        gate = getattr(moe_layer, "gate", None)
        if gate is None:
            LOG.info("BailingAdapter: layer missing gate; skipping aux-free patch")
            return
        if getattr(gate, "_afb_patched", False):
            return

        def afb_gate_forward(self, hidden_states: torch.Tensor):
            flat = hidden_states.view(-1, hidden_states.shape[-1])
            logits = F.linear(flat.float(), self.weight.float())
            scores_unbiased = torch.sigmoid(logits.float()).to(logits.dtype)
            bias = getattr(moe_layer, "_afb_bias")
            biased_scores = scores_unbiased + bias
            topk_vals, topk_idx = self.group_limited_topk(biased_scores)
            weights = torch.gather(scores_unbiased, 1, topk_idx)
            if self.top_k > 1:
                denom = weights.sum(dim=-1, keepdim=True).clamp_min_(1e-20)
                weights = weights / denom
            weights = weights * self.routed_scaling_factor

            flat_topk = topk_idx.reshape(-1)
            counts = torch.bincount(flat_topk, minlength=bias.numel())
            getattr(moe_layer, "_afb_counts").add_(counts.to(moe_layer._afb_counts.dtype))

            return topk_idx, weights.to(hidden_states.dtype), logits

        gate.forward = afb_gate_forward.__get__(gate, gate.__class__)  # type: ignore[attr-defined]
        setattr(gate, "_afb_patched", True)


class Llama4Adapter(BaseMoEAdapter):
    family = "llama4"

    def matches(self, model: nn.Module) -> bool:
        return getattr(getattr(model, "config", object()), "model_type", "") == "llama4"

    def find_moe_layers(self, model: nn.Module) -> Iterable[nn.Module]:
        for m in model.modules():
            if m.__class__.__name__ == "Llama4TextMoe":
                yield m

    def prepare(self, moe_layer: nn.Module, handle: LayerHandle, shim: AuxFreeShim) -> None:
        self._register_aux_buffers(moe_layer, handle, shim)
        self._patch_llama4_router(moe_layer)

    def _patch_llama4_router(self, moe_layer: nn.Module) -> None:
        router = getattr(moe_layer, "router", None)
        if router is None:
            LOG.info("Llama4Adapter: layer missing router; skipping aux-free patch")
            return
        if getattr(router, "_afb_patched", False):
            return

        def afb_router_forward(self, hidden_states: torch.Tensor):
            flat = hidden_states if hidden_states.dim() == 2 else hidden_states.view(-1, hidden_states.shape[-1])
            router_logits = F.linear(flat, self.weight, self.bias)
            bias = getattr(moe_layer, "_afb_bias")
            biased_logits = router_logits + bias
            _, router_indices = torch.topk(biased_logits, self.top_k, dim=1)
            unbiased_top = torch.gather(router_logits, 1, router_indices)
            router_scores = torch.full_like(router_logits, float("-inf"))
            router_scores.scatter_(1, router_indices, unbiased_top)
            router_scores = torch.sigmoid(router_scores.float()).to(router_scores.dtype)

            counts = torch.bincount(router_indices.reshape(-1), minlength=bias.numel())
            getattr(moe_layer, "_afb_counts").add_(counts.to(moe_layer._afb_counts.dtype))

            return router_scores, router_logits

        router.forward = afb_router_forward.__get__(router, router.__class__)  # type: ignore[attr-defined]
        setattr(router, "_afb_patched", True)


def discover_and_prepare_layers(model: nn.Module, adapters: list[BaseMoEAdapter], shim: AuxFreeShim) -> list[LayerHandle]:
    """Discover MoE layers using the first matching adapter and attach per-layer buffers.

    Returns a list of layer handles for later routing patching and updates.
    """
    handles: list[LayerHandle] = []
    adapter: Optional[BaseMoEAdapter] = None
    for a in adapters:
        if a.matches(model):
            adapter = a
            break

    if adapter is None:
        LOG.info("AuxFreeMoE: no matching adapter found; skipping aux-free routing")
        return handles

    # disable aux loss at model level if possible
    adapter.disable_aux_loss(getattr(model, "config", model))

    idx = 0
    for layer in adapter.find_moe_layers(model):
        try:
            top_k = adapter.get_top_k(layer)
            nE = adapter.get_num_experts(layer)
        except Exception:
            continue

        handle = LayerHandle(layer=layer, layer_idx=idx, num_experts=nE, top_k=top_k)
        adapter.prepare(layer, handle, shim)
        handles.append(handle)
        idx += 1

    LOG.info(f"AuxFreeMoE: prepared {len(handles)} {adapter.family} layers for aux-free routing")
    return handles
