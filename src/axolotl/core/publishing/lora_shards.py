"""Where each rank's slice of the LoRA adapter lives, and how to regroup it per builder.

Training keeps every adapter tensor sharded: FSDP2 splits rows, expert
parallelism keeps each rank's own experts. A ``Piece`` records one rank's slice
as a box in a canonical view of the full adapter tensor, so slices from
different shardings can be intersected:

- linear LoRA and an expert ``lora_A`` (``[E*r, in]``, expert-major rows): the
  tensor's own 2D shape;
- an expert ``lora_B`` (``[out, r*E]``, experts on the fastest axis): the
  ``[out, r, E]`` view, in which one rank's experts are a contiguous range.

A ``Need`` is the box a builder must assemble. ``plan_transfers`` turns every
rank's pieces and needs into point-to-point copies; it is computed identically
on all ranks from gathered metadata, so no rank has to ask another what to send.
"""

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import prod

import torch

Box = tuple[tuple[int, int], ...]
PREFIX = "base_model.model."
_CHECKPOINT_WRAPPER = "_checkpoint_wrapped_module."


@dataclass(frozen=True)
class Piece:
    """One rank's slice of the adapter tensor ``key``."""

    key: str
    shape: tuple[int, ...]
    view: tuple[int, ...]
    box: Box
    dtype: str

    @property
    def local_shape(self) -> tuple[int, ...]:
        return tuple(stop - start for start, stop in self.box)


@dataclass(frozen=True)
class Need:
    """A box of the adapter tensor ``key`` that one builder must hold in full."""

    key: str
    view: tuple[int, ...]
    box: Box


@dataclass(frozen=True)
class Transfer:
    """Copy ``box`` of ``key`` from piece ``piece`` on ``src`` into need ``need`` on ``dst``."""

    tag: int
    src: int
    dst: int
    piece: int
    need: int
    box: Box


def full_box(shape: Sequence[int]) -> Box:
    return tuple((0, size) for size in shape)


def expert_view(kind: str, shape: Sequence[int], experts: int) -> tuple[int, ...]:
    """Canonical view of a fused expert adapter tensor (see the module docstring)."""
    if kind == "A":
        return tuple(shape)
    out, packed = shape
    if packed % experts:
        raise ValueError(f"lora_B columns {packed} do not split into {experts} experts")
    return (out, packed // experts, experts)


def expert_need(
    key: str, kind: str, shape: Sequence[int], experts: int, start: int, stop: int
) -> Need:
    view = expert_view(kind, shape, experts)
    if kind == "A":
        rank = shape[0] // experts
        return Need(key, view, ((start * rank, stop * rank), (0, shape[1])))
    return Need(key, view, ((0, view[0]), (0, view[1]), (start, stop)))


def need_shape(need: Need) -> tuple[int, ...]:
    """The natural 2D shape of an assembled need (experts sliced out of ``lora_B``)."""
    sizes = tuple(stop - start for start, stop in need.box)
    if len(sizes) == 3:
        return (sizes[0], sizes[1] * sizes[2])
    return sizes


def _intersect(a: Box, b: Box) -> Box | None:
    box = tuple(
        (max(x0, y0), min(x1, y1)) for (x0, x1), (y0, y1) in zip(a, b, strict=True)
    )
    return box if all(start < stop for start, stop in box) else None


def _volume(box: Box) -> int:
    return prod(stop - start for start, stop in box)


def _relative(box: Box, origin: Box) -> tuple[slice, ...]:
    return tuple(
        slice(start - base, stop - base)
        for (start, stop), (base, _) in zip(box, origin, strict=True)
    )


def plan_transfers(
    pieces: Sequence[Sequence[Piece]], needs: Sequence[Sequence[Need]]
) -> list[Transfer]:
    """Every copy that assembles all ranks' needs from all ranks' pieces.

    A box held by several ranks (replicated LoRA) is read from the needing rank
    when it holds it, else from the lowest rank holding it. Distinct boxes of one
    tensor must not overlap, and every need must be covered exactly.
    """
    by_key: dict[str, dict[Box, list[tuple[int, int]]]] = {}
    views: dict[str, tuple[int, ...]] = {}
    for rank, rank_pieces in enumerate(pieces):
        for number, piece in enumerate(rank_pieces):
            if views.setdefault(piece.key, piece.view) != piece.view:
                raise ValueError(f"Ranks disagree on the view of {piece.key}")
            by_key.setdefault(piece.key, {}).setdefault(piece.box, []).append(
                (rank, number)
            )
    for key, boxes in by_key.items():
        distinct = list(boxes)
        for i, first in enumerate(distinct):
            for second in distinct[i + 1 :]:
                if _intersect(first, second) is not None:
                    raise ValueError(f"Overlapping LoRA slices of {key}")
    transfers: list[Transfer] = []
    for dst, rank_needs in enumerate(needs):
        for need_number, need in enumerate(rank_needs):
            if views.get(need.key) != need.view:
                raise ValueError(f"No rank holds {need.key} in view {need.view}")
            covered = 0
            for box, holders in by_key[need.key].items():
                overlap = _intersect(box, need.box)
                if overlap is None:
                    continue
                src, number = next(
                    ((r, n) for r, n in holders if r == dst), min(holders)
                )
                transfers.append(
                    Transfer(len(transfers), src, dst, number, need_number, overlap)
                )
                covered += _volume(overlap)
            if covered != _volume(need.box):
                raise ValueError(f"LoRA slices do not cover {need.key} {need.box}")
    return transfers


def assemble(
    comm,
    rank: int,
    data: Sequence[torch.Tensor],
    needs: Sequence[Need],
    transfers: Iterable[Transfer],
    all_pieces: Sequence[Sequence[Piece]],
) -> list[torch.Tensor]:
    """Run this rank's side of ``transfers``; one CPU tensor per need, in its view."""
    dtypes = {p.key: p.dtype for rank_pieces in all_pieces for p in rank_pieces}
    out = [
        torch.empty(
            tuple(stop - start for start, stop in need.box),
            dtype=getattr(torch, dtypes[need.key]),
        )
        for need in needs
    ]
    pieces = all_pieces[rank]
    sends, recvs, local = [], [], []
    for transfer in transfers:
        if transfer.src == rank:
            piece = pieces[transfer.piece]
            region = (
                data[transfer.piece]
                .to("cpu")
                .reshape(piece.local_shape)[_relative(transfer.box, piece.box)]
                .contiguous()
            )
            if transfer.dst == rank:
                local.append((transfer, region))
            else:
                sends.append((transfer.dst, transfer.tag, region))
        elif transfer.dst == rank:
            source = all_pieces[transfer.src][transfer.piece]
            buffer = torch.empty(
                tuple(stop - start for start, stop in transfer.box),
                dtype=getattr(torch, source.dtype),
            )
            recvs.append((transfer.src, transfer.tag, buffer))
            local.append((transfer, buffer))
    comm.exchange(sends, recvs)
    for transfer, region in local:
        target = out[transfer.need]
        target[_relative(transfer.box, needs[transfer.need].box)] = region
    return out


def global_shapes(pieces: Iterable[Iterable[Piece]]) -> Mapping[str, tuple]:
    """``key -> (dtype, shape)`` of the full adapter from every rank's pieces."""
    shapes: dict[str, tuple] = {}
    for rank_pieces in pieces:
        for piece in rank_pieces:
            entry = (piece.dtype, piece.shape)
            if shapes.setdefault(piece.key, entry) != entry:
                raise ValueError(f"Ranks disagree on the shape of {piece.key}")
    return shapes


def canonical_adapter_key(name: str, adapter: str = "default") -> str:
    """PEFT's saved key for a live parameter name (``...lora_A.default.weight``)."""
    name = name.replace(_CHECKPOINT_WRAPPER, "")
    name = re.sub(rf"\.(lora_[AB])\.{re.escape(adapter)}\.weight$", r".\1.weight", name)
    return name if name.startswith(PREFIX) else PREFIX + name


def _dtype(tensor: torch.Tensor) -> str:
    return str(tensor.dtype).removeprefix("torch.")


def _local_and_offset(parameter: torch.Tensor):
    if type(parameter).__name__ != "DTensor":
        return parameter.detach(), (0,) * parameter.ndim, tuple(parameter.shape)
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    _, offset = compute_local_shape_and_global_offset(
        parameter.shape, parameter.device_mesh, parameter.placements
    )
    return parameter.to_local().detach(), tuple(offset), tuple(parameter.shape)


def collect_model_pieces(
    model: torch.nn.Module, adapter: str = "default"
) -> list[tuple[Piece, torch.Tensor]]:
    """This rank's adapter slices of a live (possibly FSDP2 and EP sharded) PEFT model.

    The tensors are the live local shards; the stager copies them after each
    optimizer step.
    """
    from axolotl.integrations.expert_parallel.shard import (
        _is_param_wrapper,
        _real_experts_base,
    )

    result = []
    for module_name, module in model.named_modules():
        for factor in ("lora_A", "lora_B"):
            layers = getattr(module, factor, None)
            if not isinstance(layers, torch.nn.ModuleDict) or adapter not in layers:
                continue
            local, offset, shape = _local_and_offset(layers[adapter].weight)
            key = canonical_adapter_key(
                f"{module_name}.{factor}.{adapter}.weight", adapter
            )
            if not _is_param_wrapper(module) or module.get_param().ndim != 3:
                box = tuple(
                    (start, start + size)
                    for start, size in zip(offset, local.shape, strict=True)
                )
                result.append((Piece(key, shape, shape, box, _dtype(local)), local))
                continue
            kind = factor[-1]
            rank = module.r[adapter]
            base = _real_experts_base(module)
            e_local = getattr(base, "num_local_experts", None)
            e_global = getattr(base, "num_experts_global", None)
            if (
                not getattr(module, "_ep_lora_sharded", False)
                or e_local is None
                or e_global is None
            ):
                e_global = e_local = (shape[0] if kind == "A" else shape[1]) // rank
                first = 0
            else:
                first = base.local_expert_offset
            if kind == "A":
                full = (e_global * rank, shape[1])
                row = first * rank + offset[0]
                box = (
                    (row, row + local.shape[0]),
                    (offset[1], offset[1] + local.shape[1]),
                )
            else:
                if offset[1] != 0 or local.shape[1] != shape[1]:
                    raise NotImplementedError(
                        f"{key} is sharded along its expert-rank columns; per-step "
                        "publishing needs expert lora_B sharded by rows only"
                    )
                full = (shape[0], rank * e_global)
                box = (
                    (offset[0], offset[0] + local.shape[0]),
                    (0, rank),
                    (first, first + e_local),
                )
            view = expert_view(kind, full, e_global)
            result.append((Piece(key, full, view, box, _dtype(local)), local))
    if not result:
        raise ValueError("The model has no LoRA parameters to publish")
    return result
