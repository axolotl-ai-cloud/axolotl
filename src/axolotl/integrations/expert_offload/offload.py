# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Expert-granularity CPU offload for 4-bit MoE QLoRA.

Only the frozen 4-bit expert weights move; attention, router/gate, norms and the trainable LoRA
adapters stay GPU-resident. Two layouts are supported: per-expert ``bitsandbytes`` ``Linear4bit``
(an ``experts`` ``ModuleList``) and grouped 3D stacks quantized via ``quantize_moe_experts``.

Each block's packed tensors are homed in pinned CPU RAM and staged to the GPU by a forward
pre-hook; staging evicts the previously staged block (single resident slot, deliberately no evict
post-hook). Under ``use_reentrant=False`` gradient checkpointing — required and schema-enforced —
the backward recompute re-runs the pre-hook, so at most one block's experts are GPU-resident in
forward and backward alike. Single-GPU or plain DDP; see this integration's README for details.
"""

from __future__ import annotations

import os
from typing import NamedTuple

import torch
from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class _Slot(NamedTuple):
    """One offloadable packed tensor and its candidate ``state_dict`` keys relative to ``owner``.

    The parametrized layout needs two keys: bnb's own state-dict hook renames
    ``parametrizations.<p>.original`` to ``<p>``.
    """

    param: nn.Parameter
    owner: nn.Module
    keys: tuple[str, ...]


# Shared 0-element GPU placeholders that evicted experts' ``.data`` points at, cached per
# (device, dtype) — the storage dtype varies with ``bnb_4bit_quant_storage``. Keeping the real
# homes off the module means a stray ``model.to(device)`` can't drag the experts back.
_PLACEHOLDERS: dict[tuple[torch.device, torch.dtype], torch.Tensor] = {}


def _placeholder(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    key = (device, dtype)
    ph = _PLACEHOLDERS.get(key)
    if ph is None:
        ph = torch.empty(0, dtype=dtype, device=device)
        _PLACEHOLDERS[key] = ph
    return ph


def _is_pinned(t: torch.Tensor) -> bool:
    """``t.is_pinned()``, robust on hosts where it raises without CUDA."""
    try:
        return bool(t.is_pinned())
    except (RuntimeError, AssertionError):  # pragma: no cover - platform dependent
        return False


def _is_linear4bit(module: nn.Module) -> bool:
    """Structural ``Linear4bit`` check (``Params4bit`` weight type) — no hard bitsandbytes
    import, and PEFT subclasses still match."""
    weight = getattr(module, "weight", None)
    return weight is not None and type(weight).__name__ == "Params4bit"


def _base_layer(module: nn.Module) -> nn.Module:
    """Unwrap a PEFT adapter wrapper to the frozen base ``Linear4bit`` (the trainable LoRA
    matrices are separate siblings that stay GPU-resident)."""
    return getattr(module, "base_layer", module)


def find_moe_expert_blocks(
    model: nn.Module,
) -> list[tuple[str, nn.Module, list[nn.Module]]]:
    """Discover per-expert MoE blocks: an ``experts`` ``ModuleList`` (len >= 2) with 4-bit leaves.

    Fused 3D-parameter layouts are only 4-bit under ``quantize_moe_experts`` and are discovered by
    :func:`find_parametrized_expert_stacks` instead. Returns ``(name, block, base_layers)`` triples
    with base layers deduped; the block module is the hook site (its ``forward`` runs the experts
    and is what gradient checkpointing recomputes).
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


def find_parametrized_expert_stacks(
    model: nn.Module,
) -> list[tuple[str, nn.Module, list[_Slot]]]:
    """Discover grouped 3D expert stacks quantized via ``quantize_moe_experts``.

    Matches modules carrying a ``Bnb4bitParametrization`` (structurally, like ``_is_linear4bit``).
    The packed tensor is ``parametrizations[<p>].original``; the module is the hook site.
    """
    blocks: list[tuple[str, nn.Module, list[_Slot]]] = []
    for name, module in model.named_modules():
        plists = getattr(module, "parametrizations", None)
        if not isinstance(plists, nn.ModuleDict):
            continue
        slots: list[_Slot] = []
        for pname, plist in plists.items():
            if not any(type(p).__name__ == "Bnb4bitParametrization" for p in plist):
                continue
            packed = plist.original
            if not isinstance(packed, nn.Parameter) or packed.requires_grad:
                continue  # trainable parametrized params are not ours to move
            slots.append(
                _Slot(
                    param=packed,
                    owner=module,
                    keys=(pname, f"parametrizations.{pname}.original"),
                )
            )
        if slots:
            blocks.append((name, module, slots))
    return blocks


class _BlockOffload:
    """Owns one block's pinned-CPU expert homes and streams them to ``device`` for each forward /
    checkpoint recompute. While evicted the params hold 0-element placeholders; a ``state_dict``
    post-hook substitutes the CPU homes so full-model saves stay correct. ``quant_state`` and the
    LoRA adapters stay GPU-resident throughout.
    """

    # The block currently GPU-staged. Class-wide: assumes one offloaded model per process (each
    # DDP rank is its own process). Staging a block evicts this one, so at most one block is
    # resident in forward AND backward (the checkpoint recompute re-stages via the pre-hook).
    _resident: _BlockOffload | None = None

    def __init__(self, name: str, slots: list[_Slot], device, pin: bool = True):
        self.name = name
        self.device = torch.device(device)
        self.slots = slots
        # ``.to("cpu")`` copies, decoupling each home from the live param before the placeholder swap.
        self.homes: list[torch.Tensor] = [
            self._to_home(slot.param.data.detach(), pin) for slot in slots
        ]
        self.pinned = all(_is_pinned(t) for t in self.homes)
        self.staged = False
        for idx, slot in enumerate(slots):
            self._install_state_dict_hook(slot, idx)
        self.evict()  # start evicted: experts hold placeholders, ~0 GPU footprint

    @property
    def params(self) -> list[nn.Parameter]:
        return [slot.param for slot in self.slots]

    @staticmethod
    def _to_home(t: torch.Tensor, pin: bool) -> torch.Tensor:
        cpu = t.to("cpu")
        if pin:
            try:
                return cpu.pin_memory()
            except (RuntimeError, AssertionError):  # pragma: no cover - best-effort
                pass  # pinning is best-effort; pageable fallback is correct, just no async H2D
        return cpu

    def _install_state_dict_hook(self, slot: _Slot, idx: int) -> None:
        """Substitute the CPU home for the 0-element placeholder in ``state_dict()`` while evicted
        (references, not copies; a no-op while staged)."""

        def hook(module, state_dict, prefix, local_metadata):
            for key in slot.keys:
                t = state_dict.get(prefix + key)
                if t is not None and t.numel() == 0:
                    state_dict[prefix + key] = self.homes[idx]

        register = getattr(slot.owner, "register_state_dict_post_hook", None)
        if (
            register is None
        ):  # older torch: private hook, same (mod, sd, prefix, meta) signature
            register = slot.owner._register_state_dict_hook
        register(hook)

    @property
    def bytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.homes)

    def stage(self) -> None:
        """Stage this block's experts on ``device`` (idempotent), evicting the previously staged
        block. H2D copies go on the current stream, ordering them before the dequant kernels."""
        if self.staged:
            return
        cls = type(self)
        if cls._resident is not None and cls._resident is not self:
            cls._resident.evict()  # single-slot: free the prior block before staging this one
        for slot, home in zip(self.slots, self.homes, strict=True):
            slot.param.data = home.to(self.device, non_blocking=True)
        self.staged = True
        cls._resident = self

    def evict(self) -> None:
        """Point the expert weights back at shared 0-element placeholders (idempotent)."""
        for slot in self.slots:
            slot.param.data = _placeholder(self.device, slot.param.data.dtype)
        self.staged = False
        cls = type(self)
        if cls._resident is self:
            cls._resident = None


_BNB_CACHE_HOOK_NAMES = (
    "_enable_parametrization_cache",  # forward_pre_hook: parametrize._cache_enabled += 1
    "_disable_parametrization_cache",  # forward_hook: -= 1, clear cache at 0
)


def _strip_bnb_parametrize_cache_hooks(module: nn.Module) -> int:
    """Remove bitsandbytes' parametrization-cache hook pair from ``module``.

    The pair bumps the global ``parametrize._cache_enabled`` counter and clears the cache when it
    returns to 0 — but the ``use_reentrant=False`` checkpoint recompute early-stops mid-forward,
    skipping the forward_hook: the counter leaks and every dequantized expert stays cached (pool
    x4 bytes). Experts are read once per forward, so the cache buys them nothing anyway. bnb's
    state-dict post-hook is left in place.
    """
    removed = 0
    for hooks in (
        module._forward_pre_hooks,
        module._forward_hooks,
    ):
        stale = [
            k
            for k, fn in hooks.items()
            if getattr(fn, "__name__", "") in _BNB_CACHE_HOOK_NAMES
        ]
        for k in stale:
            del hooks[k]
            for extra in ("_forward_hooks_with_kwargs", "_forward_hooks_always_called"):
                d = getattr(module, extra, None)
                if d is not None and k in d:
                    del d[k]
            removed += 1
    return removed


def _reset_parametrize_cache_state() -> None:
    """Defensively zero torch's global parametrization cache (idempotent)."""
    import torch.nn.utils.parametrize as P

    P._cache_enabled = 0
    P._cache = {}


def install_expert_offload(
    model: nn.Module, device=None, pin: bool = True
) -> list[_BlockOffload]:
    """Home every discoverable MoE block's frozen 4-bit experts in (pinned) CPU RAM and register
    the stage pre-hooks. Handles are stashed on ``model._expert_offload_handles`` so they live as
    long as the model."""
    slot_blocks: list[tuple[str, nn.Module, list[_Slot]]] = [
        (
            name,
            block,
            [
                _Slot(param=base.weight, owner=base, keys=("weight",))
                for base in base_layers
            ],
        )
        for name, block, base_layers in find_moe_expert_blocks(model)
    ]
    slot_blocks += find_parametrized_expert_stacks(model)
    if not slot_blocks:
        raise RuntimeError(
            "expert_offload is enabled but no 4-bit MoE expert weights were found. This "
            "integration offloads per-expert bitsandbytes Linear4bit weights (an ``experts`` "
            "ModuleList) or grouped 3D expert stacks quantized via ``quantize_moe_experts: "
            "true`` (OLMoE / Qwen3-MoE style fused layouts on current transformers). If your "
            "model stores experts as fused 3D parameters, set ``quantize_moe_experts: true`` — "
            "plain ``load_in_4bit`` leaves those stacks unquantized, so there is nothing 4-bit "
            "to offload. Otherwise disable expert_offload for this model."
        )

    if device is None:
        device = slot_blocks[0][2][0].param.data.device
    device = torch.device(device)

    handles: list[_BlockOffload] = []
    stripped = 0
    keep_cache_hooks = os.environ.get("AXOLOTL_EXPERT_OFFLOAD_KEEP_BNB_CACHE_HOOKS", "")
    for name, block, slots in slot_blocks:
        handle = _BlockOffload(name, slots, device, pin=pin)
        block.register_forward_pre_hook(lambda module, args, h=handle: h.stage())
        handles.append(handle)
        if not keep_cache_hooks:
            for owner in {id(s.owner): s.owner for s in slots}.values():
                stripped += _strip_bnb_parametrize_cache_hooks(owner)
    if not keep_cache_hooks:
        _reset_parametrize_cache_state()
        if stripped:
            LOG.info(
                "expert_offload: removed %d bnb parametrize-cache hooks from offloaded "
                "expert modules (checkpoint early-stop leaks the global cache counter; "
                "dequants would be retained at pool x4 bytes). Set "
                "AXOLOTL_EXPERT_OFFLOAD_KEEP_BNB_CACHE_HOOKS=1 to keep them.",
                stripped,
            )

    model._expert_offload_handles = handles
    _register_ddp_ignore(model, handles)
    total_gb = sum(h.bytes for h in handles) / 1e9
    n_experts = sum(len(h.slots) for h in handles)
    pinned = "pinned" if all(h.pinned for h in handles) else "pageable (no async H2D)"
    rank = (
        f" [rank {torch.distributed.get_rank()}]"
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else ""
    )
    LOG.info(
        f"expert_offload{rank}: homed {n_experts} expert layers across {len(handles)} MoE blocks "
        f"({total_gb:.2f} GB) to {pinned} CPU RAM; one block resident on {device} at a time."
    )
    return handles


def _register_ddp_ignore(model: nn.Module, handles: list[_BlockOffload]) -> None:
    """Register the offloaded expert weights on DDP's ignore list: the initial module-state sync
    must not broadcast the evicted 0-element placeholders (frozen, so they never enter gradient
    buckets either). ``_ddp_params_and_buffers_to_ignore`` is read at DDP construction, which
    happens after ``post_model_load``."""
    offloaded_ids = {id(param) for handle in handles for param in handle.params}
    names = [
        name
        for name, param in model.named_parameters(remove_duplicate=False)
        if id(param) in offloaded_ids
    ]
    existing = list(getattr(model, "_ddp_params_and_buffers_to_ignore", []) or [])
    model._ddp_params_and_buffers_to_ignore = existing + [
        n for n in names if n not in existing
    ]
