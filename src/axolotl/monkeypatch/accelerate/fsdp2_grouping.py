"""Keep each FSDP2 unit's trainable parameters in one dtype.

FSDP2 asserts a uniform original dtype over the *trainable* parameters of a
unit and does not care about frozen ones, so fp32 LoRA factors can live in the
decoder-layer unit next to frozen bf16 or packed uint8 base weights. The only
layout that trips the assertion is two trainable dtypes in one unit, e.g. fp32
adapters next to trainable bf16 weights from ``unfrozen_parameters`` or
``lora_modules_to_save``. Carve the smaller dtype out into its own units then.
"""

from collections import defaultdict
from collections.abc import Callable, Collection

import torch
from torch import nn

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_CONTAINERS = (nn.ModuleList, nn.ModuleDict, nn.ParameterList, nn.ParameterDict)


def _is_fsdp_module(module: nn.Module) -> bool:
    from torch.distributed.fsdp import FSDPModule

    return isinstance(module, FSDPModule)


def _unit_members(
    root: nn.Module, is_wrapped: Callable[[nn.Module], bool]
) -> list[tuple[nn.Module, nn.Module | None]]:
    """(module, parent) pairs whose parameters ``fully_shard(root)`` would own."""
    members: list[tuple[nn.Module, nn.Module | None]] = []

    def visit(module: nn.Module, parent: nn.Module | None):
        members.append((module, parent))
        for child in module.children():
            if is_wrapped(child):
                continue
            visit(child, module)

    visit(root, None)
    return members


def _owner_with_forward(
    module: nn.Module, parents: dict[int, nn.Module | None]
) -> nn.Module | None:
    """Parameter containers never run forward, so FSDP hooks on them never fire."""
    while isinstance(module, _CONTAINERS):
        parent = parents.get(id(module))
        if parent is None:
            return None
        module = parent
    return module


def _is_lora_base_layer(module: nn.Module, parent: nn.Module | None) -> bool:
    return parent is not None and getattr(parent, "base_layer", None) is module


def plan_trainable_dtype_carveout(
    root: nn.Module,
    is_wrapped: Callable[[nn.Module], bool] | None = None,
    pinned: Collection[nn.Module] = (),
) -> list[nn.Module]:
    """Modules to shard on their own so that ``fully_shard(root)`` sees one trainable dtype.

    Returns an empty list when the unit already has at most one trainable dtype.
    One dtype stays in the unit: the one owned directly by ``root`` if any, else
    the one with the most parameters. Every other dtype is carved out through the
    modules that own it, which must be pure in that dtype.

    ``pinned`` modules, and a LoRA layer's ``base_layer`` (the fused kernels read
    its weight directly, bypassing any hook a unit of its own would need), count
    as owned by ``root``: they are never carved.
    """
    is_wrapped = is_wrapped or _is_fsdp_module
    pinned_ids = {id(m) for m in pinned}
    members = _unit_members(root, is_wrapped)
    parents = {id(module): parent for module, parent in members}

    numel_by_dtype: dict[torch.dtype, int] = defaultdict(int)
    owners_by_dtype: dict[torch.dtype, list[nn.Module]] = defaultdict(list)
    root_dtypes: set[torch.dtype] = set()
    for module, _ in members:
        for param in module.parameters(recurse=False):
            if not param.requires_grad:
                continue
            numel_by_dtype[param.dtype] += param.numel()
            owner = _owner_with_forward(module, parents)
            if (
                owner is None
                or owner is root
                or id(owner) in pinned_ids
                or _is_lora_base_layer(owner, parents.get(id(owner)))
            ):
                root_dtypes.add(param.dtype)
            elif owner not in owners_by_dtype[param.dtype]:
                owners_by_dtype[param.dtype].append(owner)
    if len(numel_by_dtype) <= 1:
        return []

    def trainable_dtypes(module: nn.Module) -> set[torch.dtype]:
        found = set()
        for sub, _ in _unit_members(module, is_wrapped):
            for param in sub.parameters(recurse=False):
                if param.requires_grad:
                    found.add(param.dtype)
        return found

    if len(root_dtypes) > 1:
        candidates: list[torch.dtype] = []
    elif root_dtypes:
        candidates = list(root_dtypes)
    else:
        candidates = sorted(
            numel_by_dtype, key=numel_by_dtype.__getitem__, reverse=True
        )

    for keep in candidates:
        carved = [
            owner
            for dtype, owners in owners_by_dtype.items()
            if dtype != keep
            for owner in owners
        ]
        if carved and all(
            trainable_dtypes(owner) == {dtype}
            for dtype, owners in owners_by_dtype.items()
            if dtype != keep
            for owner in owners
        ):
            return carved
    LOG.warning(
        "FSDP2 unit %s mixes trainable dtypes %s in modules that cannot be split; "
        "leaving it to FSDP2, which will reject the unit",
        type(root).__name__,
        sorted(str(d) for d in numel_by_dtype),
    )
    return []


def carve_trainable_dtype(
    root: nn.Module, fsdp2_kwargs: dict, pinned: Collection[nn.Module] = ()
) -> int:
    """Shard the minority trainable-dtype modules under ``root`` as their own units."""
    from torch.distributed.fsdp import fully_shard

    carved = plan_trainable_dtype_carveout(root, pinned=pinned)
    for module in carved:
        fully_shard(module, **fsdp2_kwargs)
    if carved:
        LOG.info(
            "fsdp2: sharded %d mixed-dtype trainable module(s) under %s separately",
            len(carved),
            type(root).__name__,
        )
    return len(carved)
