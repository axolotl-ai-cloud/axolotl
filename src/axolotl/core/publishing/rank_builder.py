"""Build one rank's share of a merged checkpoint from its own LoRA, no base traffic.

Every rank reads the base checkpoint from node-local disk and merges only the
tensors assigned to it:

- the experts whose LoRA it holds (the EP owner), merged in contiguous runs;
- the linear (attention) targets assigned round-robin, after gathering their
  small LoRA A/B slices;
- in full checkpoints only, the untargeted tensors of the base files assigned to
  it, copied as they are.

The merges call the offline merge's own per-tensor functions with the same
inputs, so each tensor is byte-identical to ``merge_lora_sharded_efficient``.
An expert run is merged as a group of ``len(run)`` experts, which yields the
same bytes per expert as the full group because every step of the expert merge
(dequantize, per-expert delta, max over projections, re-quantize) is computed
independently per expert.
"""

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open

from .lora_shards import Need, Piece, expert_need, full_box, need_shape
from .pinned_base import FLOAT_DTYPES, PinnedBase

Sink = Callable[[str, torch.Tensor], None]


@dataclass(frozen=True)
class ExpertGroup:
    """One fused expert LoRA (e.g. ``gate_up_proj``) over per-expert NVFP4 tensors."""

    prefix: str
    fused: str
    members: tuple[str, ...]
    a_key: str
    b_key: str
    experts: int

    def tensors(self, start: int, stop: int) -> list[str]:
        from axolotl.cli.utils.lora_merge import _EXPERT_LEAVES

        return [
            f"{self.prefix}.{expert}.{member}.{leaf}"
            for member in self.members
            for expert in range(start, stop)
            for leaf in _EXPERT_LEAVES
        ]


@dataclass
class Inventory:
    """What a merge of this adapter writes, independent of how ranks split it."""

    linear: dict[str, tuple[str, str]]
    groups: list[ExpertGroup]
    targeted: set[str]
    files: dict[str, list[str]]


class MergeContext:
    """The offline merge's setup, built once per publisher instead of once per merge."""

    def __init__(
        self,
        base: PinnedBase,
        adapter_config: dict,
        adapter_shapes: Mapping[str, tuple[str, Sequence[int]]],
        device: str = "cpu",
    ):
        from axolotl.cli.utils import lora_merge as merge
        from axolotl.cli.utils.param_wrapper_merge import build_param_wrapper_map

        if adapter_config.get("use_dora"):
            raise ValueError("Per-step publishing supports plain LoRA only")
        self.base, self.config, self.device = base, adapter_config, device
        self.scale_mode = merge._resolve_nvfp4_scale_mode(adapter_config)
        self.canonical_fp32 = bool(adapter_config.get("nvfp4_merge_aware"))
        rank = float(adapter_config["r"])
        alpha = float(adapter_config["lora_alpha"])
        self.scale = (
            alpha / math.sqrt(rank)
            if adapter_config.get("use_rslora")
            else alpha / rank
        )
        meta_model = merge._build_meta_model(base.path)
        self.renamings, _ = merge._get_conversion_info(base.path, meta_model=meta_model)
        self.layer_type_map = merge._build_layer_type_map(
            base.path, meta_model=meta_model
        )
        self.meta_state = {
            key: torch.empty(tuple(shape), dtype=getattr(torch, dtype), device="meta")
            for key, (dtype, shape) in adapter_shapes.items()
        }
        self._keys = {id(tensor): key for key, tensor in self.meta_state.items()}
        mapping = build_param_wrapper_map(meta_model, adapter_config, self.meta_state)
        self.param_wrapper_map = (
            None if mapping is None else merge._MemoizedParamWrapperMap(mapping)
        )
        del meta_model
        self.experts: int | None = (
            merge._get_expected_num_experts(base.path) or base.num_experts
        )
        self.inventory = self._inventory()

    def _key_of(self, tensor: torch.Tensor) -> str:
        return self._keys[id(tensor)]

    def _inventory(self) -> Inventory:
        from axolotl.cli.utils import lora_merge as merge

        layout = self.base.layout
        linear: dict[str, tuple[str, str]] = {}
        for name, (dtype, shape) in layout.items():
            if len(shape) != 2:
                continue
            a, b = merge.find_lora_weights(self.meta_state, name, self.renamings)
            if a is None or b is None:
                continue
            if dtype not in FLOAT_DTYPES:
                raise NotImplementedError(
                    f"Per-step publishing cannot merge LoRA into quantized linear {name}"
                )
            linear[name] = (self._key_of(a), self._key_of(b))
        groups: list[ExpertGroup] = []
        prefixes = sorted(
            {
                match["prefix"]
                for name, (dtype, _) in layout.items()
                if (match := merge._EXPERT_TRIPLE_RE.match(name))
                and match["leaf"] == "weight"
                and dtype == "uint8"
            }
        )
        experts = self.experts or 0
        if prefixes:
            if not experts:
                raise ValueError("The base config does not give its expert count")
            probe = merge._Nvfp4ExpertMergeWriter(
                self.meta_state,
                self.config,
                experts,
                "cpu",
                scale_mode=self.scale_mode,
                param_wrapper_map=self.param_wrapper_map,
                weight_renamings=self.renamings,
            )
        for prefix in prefixes:
            if not probe._prefix_has_fused_lora(prefix):
                continue
            for fused, members in merge._EXPERT_FUSED_GROUPS:
                stems = [f"{prefix}.0.{member}.weight" for member in members]
                if any(stem not in layout for stem in stems):
                    continue
                rows = sum(layout[stem][1][0] for stem in stems)
                columns = layout[stems[0]][1][1] * 2
                a, b, _ = merge._find_param_wrapper_lora(
                    self.meta_state,
                    f"{prefix}.{fused}",
                    tensor_shape=(experts, rows, columns),
                    param_wrapper_map=self.param_wrapper_map,
                    weight_renamings=self.renamings,
                )
                if a is None or b is None:
                    continue
                groups.append(
                    ExpertGroup(
                        prefix,
                        fused,
                        members,
                        self._key_of(a),
                        self._key_of(b),
                        experts,
                    )
                )
        used = {key for pair in linear.values() for key in pair} | {
            key for group in groups for key in (group.a_key, group.b_key)
        }
        if unused := self.meta_state.keys() - used:
            raise ValueError(
                f"{len(unused)} adapter tensors match no base tensor, "
                f"e.g. {sorted(unused)[0]}"
            )
        targeted = set(linear)
        for group in groups:
            targeted.update(group.tensors(0, group.experts))
        files: dict[str, list[str]] = {}
        for name, filename in self.base.files.items():
            if name not in targeted:
                files.setdefault(filename, []).append(name)
        return Inventory(linear, groups, targeted, files)


def _runs(experts: Sequence[int], chunk: int) -> list[tuple[int, int]]:
    runs: list[list[int]] = []
    for expert in sorted(experts):
        if runs and runs[-1][1] == expert and expert - runs[-1][0] < chunk:
            runs[-1][1] = expert + 1
        else:
            runs.append([expert, expert + 1])
    return [(start, stop) for start, stop in runs]


def _held(pieces: Sequence[Piece], need: Need) -> int:
    total = 0
    for piece in pieces:
        if piece.key != need.key or piece.view != need.view:
            continue
        sizes = [
            max(0, min(p1, n1) - max(p0, n0))
            for (p0, p1), (n0, n1) in zip(piece.box, need.box, strict=True)
        ]
        total += math.prod(sizes)
    return total


@dataclass
class RankWork:
    """One rank's assignment and the LoRA boxes it must assemble for it."""

    linear: list[str] = field(default_factory=list)
    runs: list[tuple[int, int, int]] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    needs: list[Need] = field(default_factory=list)

    def owned(self, inventory: Inventory) -> set[str]:
        names = set(self.linear)
        for group, start, stop in self.runs:
            names.update(inventory.groups[group].tensors(start, stop))
        for filename in self.files:
            names.update(inventory.files[filename])
        return names


def plan_work(
    context: MergeContext, pieces: Sequence[Sequence[Piece]], *, expert_chunk: int = 8
) -> list[RankWork]:
    """Assign every tensor to one rank; identical on every rank given the same pieces.

    Each rank merges its experts ``expert_chunk`` at a time, which bounds the
    merge's GPU working set.
    """
    if expert_chunk <= 0:
        raise ValueError("expert_chunk must be positive")
    inventory = context.inventory
    world = len(pieces)
    work = [RankWork() for _ in range(world)]
    shapes = {key: tensor.shape for key, tensor in context.meta_state.items()}
    for number, name in enumerate(sorted(inventory.linear)):
        rank = work[number % world]
        rank.linear.append(name)
        for key in inventory.linear[name]:
            rank.needs.append(Need(key, tuple(shapes[key]), full_box(shapes[key])))
    for number, group in enumerate(inventory.groups):
        owners: dict[int, list[int]] = {}
        for expert in range(group.experts):
            need = expert_need(
                group.a_key, "A", shapes[group.a_key], group.experts, expert, expert + 1
            )
            held = [_held(rank_pieces, need) for rank_pieces in pieces]
            owners.setdefault(held.index(max(held)), []).append(expert)
        for owner, experts in sorted(owners.items()):
            for start, stop in _runs(experts, expert_chunk):
                work[owner].runs.append((number, start, stop))
                for key, kind in ((group.a_key, "A"), (group.b_key, "B")):
                    work[owner].needs.append(
                        expert_need(key, kind, shapes[key], group.experts, start, stop)
                    )
    for number, filename in enumerate(sorted(inventory.files)):
        work[number % world].files.append(filename)
    return work


def work_digest(work: Sequence[RankWork]) -> str:
    """Identify the split so a chain never continues across a different one."""
    encoded = json.dumps(
        [dataclasses.asdict(rank) for rank in work], sort_keys=True, default=list
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


class BaseReader:
    """Read named base tensors, opening each shard once per call."""

    def __init__(self, base: PinnedBase):
        self.base = base

    def read(self, names: Sequence[str]) -> dict[str, torch.Tensor]:
        by_file: dict[str, list[str]] = {}
        for name in names:
            by_file.setdefault(self.base.files[name], []).append(name)
        out = {}
        for filename, keys in by_file.items():
            with safe_open(
                Path(self.base.path) / filename, framework="pt", device="cpu"
            ) as shard:
                for key in keys:
                    out[key] = shard.get_tensor(key)
        return out


class RankBuilder:
    """Merge one rank's assigned tensors and stream them to a sink."""

    def __init__(self, context: MergeContext, work: RankWork):
        self.context, self.work = context, work
        self.reader = BaseReader(context.base)
        self.inventory = context.inventory
        self.untargeted = sorted(
            name for filename in work.files for name in self.inventory.files[filename]
        )

    def lora_state(self, assembled: Sequence[torch.Tensor]) -> list[dict]:
        """Group the assembled needs per linear target and per expert run."""
        tensors = [
            tensor.reshape(need_shape(need))
            for tensor, need in zip(assembled, self.work.needs, strict=True)
        ]
        states, position = [], 0
        for _ in self.work.linear:
            states.append(
                {
                    self.work.needs[position].key: tensors[position],
                    self.work.needs[position + 1].key: tensors[position + 1],
                }
            )
            position += 2
        for _ in self.work.runs:
            states.append(
                {
                    self.work.needs[position].key: tensors[position],
                    self.work.needs[position + 1].key: tensors[position + 1],
                }
            )
            position += 2
        return states

    def build(
        self, assembled: Sequence[torch.Tensor], sink: Sink, *, full: bool
    ) -> None:
        states = self.lora_state(assembled)
        for name, state in zip(self.work.linear, states, strict=False):
            sink(name, self._merge_linear(name, state))
        for (group, start, stop), state in zip(
            self.work.runs, states[len(self.work.linear) :], strict=True
        ):
            for name, tensor in self._merge_experts(
                self.inventory.groups[group], start, stop, state
            ).items():
                sink(name, tensor)
        if full:
            for filename in self.work.files:
                for name, tensor in self.reader.read(
                    self.inventory.files[filename]
                ).items():
                    sink(name, tensor)

    def _merge_linear(self, name: str, state: dict) -> torch.Tensor:
        from axolotl.cli.utils import lora_merge as merge

        context = self.context
        tensor = self.reader.read([name])[name]
        merged, did_merge = merge._merge_tensor_with_lora(
            tensor,
            name,
            state,
            context.scale,
            context.config,
            context.device,
            weight_renamings=context.renamings,
            layer_type_map=context.layer_type_map,
            param_wrapper_map=context.param_wrapper_map,
            canonical_fp32=context.canonical_fp32,
        )
        if not did_merge:
            raise RuntimeError(f"LoRA for {name} was not applied")
        return merged

    def _merge_experts(
        self, group: ExpertGroup, start: int, stop: int, state: dict
    ) -> dict[str, torch.Tensor]:
        from axolotl.cli.utils import lora_merge as merge

        context = self.context
        count = stop - start
        mapping = context.param_wrapper_map
        if mapping is not None:
            target = merge._param_wrapper_target(
                f"{group.prefix}.{group.fused}", mapping, context.renamings
            )
            mapping = merge._MemoizedParamWrapperMap(
                {
                    name: dataclasses.replace(value, shape=(count, *value.shape[1:]))
                    for name, value in mapping.items()
                    if value == target
                }
            )
        writer = merge._Nvfp4ExpertMergeWriter(
            state,
            context.config,
            count,
            context.device,
            scale_mode=context.scale_mode,
            param_wrapper_map=mapping,
            weight_renamings=context.renamings,
        )
        local, names = {}, {}
        for name, tensor in self.reader.read(group.tensors(start, stop)).items():
            match = merge._EXPERT_TRIPLE_RE.match(name)
            assert match is not None
            expert = int(match["e"]) - start
            renamed = f"{group.prefix}.{expert}.{match['proj']}.{match['leaf']}"
            local[renamed] = tensor
            names[renamed] = name
        remaining, emitted, merged = writer.consume(local)
        if merged != 1 or remaining or writer.pending:
            raise RuntimeError(
                f"Expert merge of {group.prefix}.{group.fused} experts "
                f"{start}..{stop - 1} did not complete"
            )
        if emitted.keys() != names.keys():
            raise RuntimeError(f"Expert merge of {group.prefix} emitted other tensors")
        return {names[key]: tensor for key, tensor in emitted.items()}
