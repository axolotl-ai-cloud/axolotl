# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expert-granularity CPU offload for 4-bit MoE QLoRA on a single GPU.

Only the *frozen 4-bit expert* weights are moved; attention, router/gate, norms and the
trainable LoRA adapters stay GPU-resident. Each expert is a ``bitsandbytes`` ``Linear4bit`` whose
big packed tensor is ``weight.data`` (the ``quant_state`` scales are ~1/32 the size and are left
resident). We home every offloaded block's ``weight.data`` in *pinned* CPU RAM, and a **forward
pre-hook** on the MoE block copies that block's experts onto the GPU just before the block runs.

Eviction is driven entirely by a **single-resident-slot** policy: staging a block first evicts the
previously-staged one. There is deliberately **no evict post-hook**. Under ``use_reentrant=False``
gradient checkpointing each decoder layer's forward is *recomputed* in the backward pass; the same
pre-hook re-stages the block's experts for that recompute, and because nothing evicts a block until
the *next* block stages — which, in backward (processed last-layer-first), only happens after the
current block's recomputed backward has finished — the staged weights are always present when the
recomputed backward reads them. So at most **one block's** experts are GPU-resident at any instant,
in forward and backward alike, without depending on exactly when PyTorch stops a recompute.

This lets a fused MoE whose 4-bit experts exceed VRAM QLoRA-train on a small card, at the cost of
one host->device expert transfer per block per pass — a memory-for-compute trade.

**Why gradient checkpointing is required (correctness *and* the memory win).**
``bnb.matmul_4bit``'s autograd ``Function`` re-reads the packed weight in its backward (to
re-dequantize for the input gradient) via ``save_for_backward``. Eviction repoints
``weight.data`` at a 0-element placeholder, so the saved reference would read that placeholder in a
backward that runs against the *initial* forward's graph. Gradient checkpointing discards the
initial-forward saved tensors and **recomputes** each layer in backward (re-staging via the
pre-hook and rebuilding the saved tensors from the staged weights), which is what makes eviction
both correct and actually memory-freeing rather than pinning every staged weight alive as a saved
tensor. The plugin enforces ``gradient_checkpointing: true`` with ``use_reentrant: false``.

Single-GPU only: FSDP / DeepSpeed / expert-parallel move or shard these same weights and would
race the stage/evict swaps. The plugin refuses to enable under any of them.
"""

from __future__ import annotations

import torch
from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# 0-element GPU placeholders that an evicted expert's ``weight.data`` points at while offloaded.
# Shared across all offloaded experts (reads never mutate them) and cached per (device, dtype) —
# the 4-bit storage dtype varies with ``bnb_4bit_quant_storage`` (uint8 / bfloat16 / float32), so
# the placeholder must match the real tensor's dtype or a restage would change it. Keeping the real
# "home" data OFF the module — only a 0-element placeholder is registered while evicted — means a
# stray ``model.to(device)`` never drags the big expert tensors back onto the GPU.
_PLACEHOLDERS: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _placeholder(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    ph = _PLACEHOLDERS.get(key)
    if ph is None:
        ph = torch.empty(0, dtype=dtype, device=device)
        _PLACEHOLDERS[key] = ph
    return ph


def _is_pinned(t: torch.Tensor) -> bool:
    """Whether ``t`` is pinned (so a ``non_blocking`` H2D copy is truly async). Robust on hosts
    where ``is_pinned`` is unavailable/raises without CUDA."""
    try:
        return bool(t.is_pinned())
    except Exception:  # pragma: no cover - platform dependent
        return False


def _is_linear4bit(module: nn.Module) -> bool:
    """A ``bitsandbytes`` ``Linear4bit`` whose ``weight`` is a packed 4-bit ``Params4bit``.

    Matched structurally (by the ``Params4bit`` weight type) rather than by ``isinstance`` so we do
    not hard-import bitsandbytes here and so PEFT/other subclasses of ``Linear4bit`` still match.
    """
    weight = getattr(module, "weight", None)
    return weight is not None and type(weight).__name__ == "Params4bit"


def _base_layer(module: nn.Module) -> nn.Module:
    """Unwrap a PEFT adapter wrapper (``lora.Linear4bit`` etc.) to the frozen base ``Linear4bit``.

    PEFT wraps the quantized expert and delegates ``.weight`` to ``.base_layer``; the packed tensor
    we offload lives on that base, and the (tiny, trainable) LoRA ``A``/``B`` matrices are separate
    siblings that must stay GPU-resident. Returns ``module`` unchanged if it is not wrapped.
    """
    return getattr(module, "base_layer", module)


def find_moe_expert_blocks(
    model: nn.Module,
) -> list[tuple[str, nn.Module, list[nn.Module]]]:
    """Discover offloadable MoE blocks and their frozen 4-bit expert base layers.

    A block is any module exposing an ``experts`` ``ModuleList`` of length >= 2 whose leaves include
    ``Linear4bit`` weights — the canonical per-expert layout of Mixtral, Qwen2/3-MoE, OLMoE,
    DeepSeek-MoE, Jamba, etc. (``block_sparse_moe.experts`` / ``mlp.experts``). Fused-experts
    layouts that pack all experts into one 3D parameter (GPT-OSS, DBRX, and the
    ``@use_experts_implementation`` path handled by the ``expert_parallel`` integration) are **not**
    matched — bitsandbytes does not 4-bit-quantize raw 3D parameters, so there is nothing to offload
    there; those are out of scope for this integration.

    Returns ``(block_name, block_module, expert_base_layers)`` triples. Deduplicates base layers so a
    projection shared across the list is homed once. The hook attaches to ``block_module`` because
    its ``forward`` runs the experts (and is what gradient checkpointing recomputes).
    """
    blocks: list[tuple[str, nn.Module, list[nn.Module]]] = []
    for name, block in model.named_modules():
        experts = getattr(block, "experts", None)
        if not isinstance(experts, nn.ModuleList) or len(experts) < 2:
            continue
        seen: set[int] = set()
        base_layers: list[nn.Module] = []
        for module in experts.modules():
            if not _is_linear4bit(module):
                continue
            base = _base_layer(module)
            if not _is_linear4bit(base) or id(base) in seen:
                continue
            seen.add(id(base))
            base_layers.append(base)
        if base_layers:
            blocks.append((name, block, base_layers))
    return blocks


class _BlockOffload:
    """Owns the pinned-CPU home copies of one MoE block's expert ``weight.data`` tensors and streams
    them to ``device`` for the duration of each forward / gradient-checkpoint recompute.

    While evicted, each expert's ``weight.data`` holds a shared 0-element GPU placeholder, so nothing
    that walks the module tree (``.to()``, checkpoint-save) drags the offloaded data back onto the
    GPU. ``quant_state`` (the small NF4 scales) stays GPU-resident throughout, as do the LoRA
    adapters and everything outside ``experts``. A ``state_dict`` post-hook substitutes the CPU homes
    for the placeholders so a full-model save stays correct (adapter-only saves never touch the base
    keys, so they are unaffected).
    """

    # The single block whose experts are currently GPU-staged. Class-wide, so it assumes one
    # offloaded model per process (the training case). Under use_reentrant=False gradient
    # checkpointing the backward RECOMPUTE re-runs a layer's forward to rebuild its saved tensors;
    # staging a new block first evicts this previously-staged one, so at most one block is
    # GPU-resident at any instant, in forward AND backward.
    _resident: _BlockOffload | None = None

    def __init__(
        self, name: str, base_layers: list[nn.Module], device, pin: bool = True
    ):
        self.name = name
        self.device = torch.device(device)
        self.base_layers = base_layers
        # Capture each packed weight as a SEPARATE (pinned) CPU tensor BEFORE any placeholder swap.
        # The source is on the GPU at install time, so ``.to("cpu")`` is a real device->host copy
        # that decouples the home from the live parameter we then overwrite with a placeholder.
        self.homes: list[torch.Tensor] = [
            self._to_home(base.weight.data.detach(), pin) for base in base_layers
        ]
        self.pinned = all(_is_pinned(t) for t in self.homes)
        self.staged = False
        for base in base_layers:
            self._install_state_dict_hook(base)
        self.evict()  # start evicted: experts hold placeholders, ~0 GPU footprint

    @staticmethod
    def _to_home(t: torch.Tensor, pin: bool) -> torch.Tensor:
        cpu = t.to("cpu")
        if pin:
            try:
                return cpu.pin_memory()
            except (RuntimeError, AssertionError):  # pragma: no cover - best-effort
                pass  # pinning is best-effort; pageable fallback is correct, just no async H2D
        return cpu

    def _install_state_dict_hook(self, base: nn.Module) -> None:
        """Keep full-model ``state_dict()`` correct while evicted: substitute the (pinned) CPU home
        for the 0-element ``weight`` placeholder. References, not copies, so adapter-only saves stay
        cheap and while *staged* it is a no-op (the entry is the real GPU tensor)."""
        idx = self.base_layers.index(base)

        def hook(module, state_dict, prefix, local_metadata):
            key = prefix + "weight"
            t = state_dict.get(key)
            if t is not None and t.numel() == 0:
                state_dict[key] = self.homes[idx]

        register = getattr(base, "register_state_dict_post_hook", None)
        if (
            register is None
        ):  # older torch: private hook, same (mod, sd, prefix, meta) signature
            register = base._register_state_dict_hook
        register(hook)

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.homes)

    def stage(self) -> None:
        """Copy this block's packed expert weights onto ``device`` (idempotent), first evicting the
        previously staged block so at most one block's experts are GPU-resident. The H2D copies are
        enqueued on the current stream, so the dequant kernels that immediately follow are ordered
        after them."""
        if self.staged:
            return
        cls = type(self)
        if cls._resident is not None and cls._resident is not self:
            cls._resident.evict()  # single-slot: free the prior block before staging this one
        for base, home in zip(self.base_layers, self.homes, strict=True):
            base.weight.data = home.to(self.device, non_blocking=True)
        self.staged = True
        cls._resident = self

    def evict(self) -> None:
        """Point this block's expert weights back at shared 0-element placeholders (idempotent),
        dropping the GPU copies so the caching allocator can reuse the memory for the next block."""
        for base in self.base_layers:
            w = base.weight
            w.data = _placeholder(self.device, w.data.dtype)
        self.staged = False
        cls = type(self)
        if cls._resident is self:
            cls._resident = None


def install_expert_offload(
    model: nn.Module, device=None, pin: bool = True
) -> list[_BlockOffload]:
    """Offload every discoverable MoE block's frozen 4-bit experts to (pinned) CPU RAM.

    For each block, homes its expert ``weight.data`` tensors on the CPU, evicts them from the GPU,
    and registers a forward pre-hook (stage) on the block module. The handles are stashed on
    ``model._expert_offload_handles`` so they live as long as the model. Returns the handles (empty
    if no offloadable MoE block was found).
    """
    blocks = find_moe_expert_blocks(model)
    if not blocks:
        LOG.warning(
            "expert_offload: no 4-bit MoE expert blocks found. This integration offloads "
            "per-expert bitsandbytes Linear4bit weights (Mixtral / Qwen-MoE / OLMoE / DeepSeek "
            "style). Fused-expert or non-quantized models are unaffected."
        )
        return []

    if device is None:
        device = blocks[0][2][0].weight.data.device
    device = torch.device(device)

    handles: list[_BlockOffload] = []
    for name, block, base_layers in blocks:
        handle = _BlockOffload(name, base_layers, device, pin=pin)
        block.register_forward_pre_hook(lambda module, args, h=handle: h.stage())
        handles.append(handle)

    model._expert_offload_handles = handles
    total_gb = sum(h.bytes for h in handles) / 1e9
    n_experts = sum(len(h.base_layers) for h in handles)
    pinned = "pinned" if all(h.pinned for h in handles) else "pageable (no async H2D)"
    LOG.info(
        f"expert_offload: homed {n_experts} expert layers across {len(handles)} MoE blocks "
        f"({total_gb:.2f} GB) to {pinned} CPU RAM; one block resident on {device} at a time."
    )
    return handles
