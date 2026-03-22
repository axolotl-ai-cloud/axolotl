"""Architecture-specific adapters for aux-loss-free MoE routing.

Each adapter discovers MoE layers for a model family and patches only the
router/gate to inject per-expert bias into expert selection while keeping
mixture weights from unbiased logits.  Expert dispatch is left untouched so
the patching composes with any expert backend (eager, ScatterMoE, SonicMoE).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn

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
    """Base adapter that discovers MoE layers and patches their routing.

    Concrete adapters implement discovery, attribute extraction, and
    architecture-specific router patching.
    """

    family: str = "generic"

    def matches(self, model: nn.Module) -> bool:  # pragma: no cover - thin shim
        return False

    def find_moe_layers(
        self, model: nn.Module
    ) -> Iterable[nn.Module]:  # pragma: no cover
        return []

    def get_top_k(self, moe_layer: nn.Module) -> int:
        """Resolve top_k from the MoE layer, checking common attribute paths."""
        for attr_path in [
            ("top_k",),
            ("num_experts_per_tok",),
            ("gate", "top_k"),
            ("router", "top_k"),
        ]:
            obj: object = moe_layer
            for attr in attr_path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if isinstance(obj, int):
                return obj
        return 2

    def get_num_experts(self, moe_layer: nn.Module) -> int:
        """Resolve num_experts from the MoE layer, checking common attribute paths."""
        for attr_path in [
            ("num_experts",),
            ("num_local_experts",),
            ("gate", "num_experts"),
            ("router", "num_experts"),
            ("experts", "num_experts"),
        ]:
            obj: object = moe_layer
            for attr in attr_path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if isinstance(obj, int):
                return obj
        raise AttributeError(f"Cannot determine num_experts for {type(moe_layer)}")

    def disable_aux_loss(self, model_or_layer: nn.Module) -> None:
        # Best-effort: zero router aux loss coef if present
        if hasattr(model_or_layer, "router_aux_loss_coef"):
            try:
                model_or_layer.router_aux_loss_coef = 0.0
            except Exception:  # pragma: no cover - non-critical
                pass

    def _register_aux_buffers(
        self, moe_layer: nn.Module, handle: LayerHandle, shim: AuxFreeShim
    ) -> None:
        device = next(moe_layer.parameters(), torch.tensor(0)).device
        if not hasattr(moe_layer, "_afb_bias"):
            moe_layer.register_buffer(
                "_afb_bias", torch.zeros(handle.num_experts, device=device)
            )
        if not hasattr(moe_layer, "_afb_counts"):
            moe_layer.register_buffer(
                "_afb_counts", torch.zeros(handle.num_experts, device=device)
            )
        if not hasattr(moe_layer, "_afb_ema"):
            moe_layer.register_buffer(
                "_afb_ema", torch.zeros(handle.num_experts, device=device)
            )
        moe_layer._afb_layer_idx = handle.layer_idx  # type: ignore[attr-defined]
        moe_layer._afb_top_k = handle.top_k  # type: ignore[attr-defined]
        shim.register_layer_buffers(handle.layer_idx, moe_layer)

    def prepare(
        self, moe_layer: nn.Module, handle: LayerHandle, shim: AuxFreeShim
    ) -> None:
        """Attach per-layer buffers.  Subclasses override to also patch routing."""
        self._register_aux_buffers(moe_layer, handle, shim)

    def uses_kernel_routing(self, moe_layer: nn.Module) -> bool:
        """Return True when a kernel backend (SonicMoE / ScatterMoE) has
        already replaced the block forward, meaning the routing is handled
        inside the kernel forward and we should NOT patch the router."""
        cls = type(moe_layer)
        # SonicMoE stores the original forward when it patches a class.
        if hasattr(cls, "_original_forward"):
            return True
        # ScatterMoE replaces via kernels library; check for the marker.
        if hasattr(cls, "_kernel_forward"):
            return True
        return False


class MixtralAdapter(BaseMoEAdapter):
    """Patches the TopKRouter for Mixtral / Qwen-MoE style softmax→topk
    routing so that biased logits drive expert *selection* while unbiased
    softmax scores drive mixture *weights*.

    Works with transformers v5 where experts are fused 3D tensors and
    the router is ``MixtralTopKRouter`` (returns a 3-tuple).
    """

    family = "mixtral"

    def matches(self, model: nn.Module) -> bool:
        return (
            getattr(getattr(model, "config", object()), "model_type", "") == "mixtral"
        )

    def find_moe_layers(self, model: nn.Module) -> Iterable[nn.Module]:
        for m in model.modules():
            if m.__class__.__name__.endswith("SparseMoeBlock"):
                yield m

    def prepare(
        self, moe_layer: nn.Module, handle: LayerHandle, shim: AuxFreeShim
    ) -> None:
        self._register_aux_buffers(moe_layer, handle, shim)
        if not self.uses_kernel_routing(moe_layer):
            self._patch_router(moe_layer)
        else:
            LOG.info(
                "AuxFreeMoE: kernel backend detected on %s; "
                "skipping router patch (kernel routing handles bias)",
                type(moe_layer).__name__,
            )

    def _patch_router(self, moe_layer: nn.Module) -> None:
        """Patch the TopKRouter to inject aux-free bias into expert selection."""
        gate = getattr(moe_layer, "gate", None)
        if gate is None:
            LOG.info("MixtralAdapter: layer missing gate; skipping aux-free patch")
            return
        if getattr(gate, "_afb_patched", False):
            return

        # Capture reference to the MoE block for bias / counts access.
        block_ref = moe_layer

        def afb_router_forward(self, hidden_states: torch.Tensor):
            hidden_states = hidden_states.reshape(-1, self.hidden_dim)
            router_logits = F.linear(hidden_states, self.weight)
            router_probs = F.softmax(router_logits.float(), dim=-1)

            # Biased selection, unbiased weights
            bias = block_ref._afb_bias
            biased = router_probs + bias
            _, router_indices = torch.topk(biased, self.top_k, dim=-1)
            router_scores = torch.gather(router_probs, 1, router_indices)

            # Renormalize (Mixtral always normalizes; Qwen checks config)
            if getattr(self, "norm_topk_prob", True):
                router_scores = router_scores / router_scores.sum(dim=-1, keepdim=True)

            # Accumulate counts for the bias-update callback
            flat_idx = router_indices.reshape(-1)
            counts = torch.bincount(flat_idx, minlength=self.num_experts)
            block_ref._afb_counts.add_(counts.to(block_ref._afb_counts.dtype))

            return router_probs, router_scores, router_indices

        gate.forward = afb_router_forward.__get__(gate, gate.__class__)  # type: ignore[attr-defined]
        gate._afb_patched = True
        moe_layer._afb_patched = True


class Qwen3Adapter(MixtralAdapter):
    family = "qwen3_moe"

    def matches(self, model: nn.Module) -> bool:
        return getattr(getattr(model, "config", object()), "model_type", "") in (
            "qwen3_moe",
            "qwen2_moe",
        )


class Qwen35MoeAdapter(MixtralAdapter):
    """Adapter for Qwen 3.5 MoE models.

    Same softmax→topk router pattern as Mixtral/Qwen3.  The shared expert
    is handled by the block forward (untouched by router-level patching).
    """

    family = "qwen3_5_moe"

    def matches(self, model: nn.Module) -> bool:
        return getattr(getattr(model, "config", object()), "model_type", "") in (
            "qwen3_5_moe",
            "qwen3_5_moe_text",
        )


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
            return int(moe_layer.num_experts)
        cfg = getattr(moe_layer, "config", None)
        if cfg is None:
            raise AttributeError(f"Cannot determine num_experts for {type(moe_layer)}")
        return int(cfg.num_experts)

    def prepare(
        self, moe_layer: nn.Module, handle: LayerHandle, shim: AuxFreeShim
    ) -> None:
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
            bias = moe_layer._afb_bias
            biased_scores = scores_unbiased + bias
            topk_vals, topk_idx = self.group_limited_topk(biased_scores)
            weights = torch.gather(scores_unbiased, 1, topk_idx)
            if self.top_k > 1:
                denom = weights.sum(dim=-1, keepdim=True).clamp_min_(1e-20)
                weights = weights / denom
            weights = weights * self.routed_scaling_factor

            flat_topk = topk_idx.reshape(-1)
            counts = torch.bincount(flat_topk, minlength=bias.numel())
            moe_layer._afb_counts.add_(counts.to(moe_layer._afb_counts.dtype))

            return topk_idx, weights.to(hidden_states.dtype), logits

        gate.forward = afb_gate_forward.__get__(gate, gate.__class__)  # type: ignore[attr-defined]
        gate._afb_patched = True


class Llama4Adapter(BaseMoEAdapter):
    family = "llama4"

    def matches(self, model: nn.Module) -> bool:
        return getattr(getattr(model, "config", object()), "model_type", "") == "llama4"

    def find_moe_layers(self, model: nn.Module) -> Iterable[nn.Module]:
        for m in model.modules():
            if m.__class__.__name__ == "Llama4TextMoe":
                yield m

    def prepare(
        self, moe_layer: nn.Module, handle: LayerHandle, shim: AuxFreeShim
    ) -> None:
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
            flat = (
                hidden_states
                if hidden_states.dim() == 2
                else hidden_states.view(-1, hidden_states.shape[-1])
            )
            router_logits = F.linear(flat, self.weight, self.bias)
            bias = moe_layer._afb_bias
            biased_logits = router_logits + bias
            _, router_indices = torch.topk(biased_logits, self.top_k, dim=1)
            unbiased_top = torch.gather(router_logits, 1, router_indices)
            router_scores = torch.full_like(router_logits, float("-inf"))
            router_scores.scatter_(1, router_indices, unbiased_top)
            router_scores = torch.sigmoid(router_scores.float()).to(router_scores.dtype)

            counts = torch.bincount(router_indices.reshape(-1), minlength=bias.numel())
            moe_layer._afb_counts.add_(counts.to(moe_layer._afb_counts.dtype))

            return router_scores, router_logits

        router.forward = afb_router_forward.__get__(router, router.__class__)  # type: ignore[attr-defined]
        router._afb_patched = True


def discover_and_prepare_layers(
    model: nn.Module, adapters: list[BaseMoEAdapter], shim: AuxFreeShim
) -> list[LayerHandle]:
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
        except (AttributeError, TypeError, ValueError):
            continue

        handle = LayerHandle(layer=layer, layer_idx=idx, num_experts=nE, top_k=top_k)
        adapter.prepare(layer, handle, shim)
        handles.append(handle)
        idx += 1

    LOG.info(
        f"AuxFreeMoE: prepared {len(handles)} {adapter.family} layers for aux-free routing"
    )
    return handles
