"""Gradient clipping that keeps CPU-offloaded DTensor shards local."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Partial, Replicate, Shard


def _local_gradient(gradient):
    return gradient.to_local() if isinstance(gradient, DTensor) else gradient


def _all_reduce_scalar(
    value: torch.Tensor, op: dist.ReduceOp, group=None
) -> torch.Tensor:
    """All-reduce a (small) tensor, staging CPU values through CUDA for NCCL."""
    if not (dist.is_available() and dist.is_initialized()):
        return value
    if dist.get_backend(group) != "nccl" or value.device.type != "cpu":
        dist.all_reduce(value, op=op, group=group)
        return value
    staged = value.to(torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(staged, op=op, group=group)
    return staged.to(value.device)


def has_cpu_offloaded_dtensor_parameters(parameters: Iterable[torch.Tensor]) -> bool:
    return any(
        isinstance(parameter, DTensor)
        and parameter.requires_grad
        and parameter.to_local().device.type == "cpu"
        for parameter in parameters
    )


def has_cpu_offloaded_dtensor_gradients(parameters: Iterable[torch.Tensor]) -> bool:
    return any(
        isinstance(parameter.grad, DTensor)
        and parameter.grad.to_local().device.type == "cpu"
        for parameter in parameters
        if parameter.grad is not None
    )


def get_grad_norm_local_shards_(
    parameters: Iterable[torch.Tensor], norm_type: float = 2.0
) -> torch.Tensor:
    """Return a global norm without modifying CPU-offloaded DTensor gradients."""
    parameters = [
        parameter
        for parameter in parameters
        if parameter.grad is not None or parameter.requires_grad
    ]
    norm_type = float(norm_type)
    if norm_type <= 0 or math.isnan(norm_type):
        raise ValueError(f"norm_type must be positive or inf, got {norm_type}")

    layouts = [
        gradient
        if isinstance(gradient := parameter.grad, DTensor)
        else parameter
        if isinstance(parameter, DTensor)
        else None
        for parameter in parameters
    ]
    for layout in layouts:
        if isinstance(layout, DTensor) and any(
            isinstance(placement, Partial) for placement in layout.placements
        ):
            raise NotImplementedError(
                "CPU-offloaded DTensor gradient clipping does not support Partial placements"
            )

    first_local = next(
        (
            _local_gradient(parameter.grad)
            if parameter.grad is not None
            else parameter.to_local()
            for parameter in parameters
            if parameter.grad is not None or isinstance(parameter, DTensor)
        ),
        torch.zeros(()),
    )
    accumulator = torch.zeros((), dtype=torch.float32, device=first_local.device)
    is_inf = math.isinf(norm_type)
    reduction = dist.ReduceOp.MAX if is_inf else dist.ReduceOp.SUM
    for parameter, layout in zip(parameters, layouts, strict=True):
        gradient = parameter.grad
        local = _local_gradient(gradient) if gradient is not None else None
        value = (
            local.detach().to(torch.float32).abs()
            if local is not None
            else accumulator.new_zeros((0,))
        )
        if is_inf:
            contribution = value.max() if value.numel() else value.new_zeros(())
        else:
            contribution = value.pow(norm_type).sum()
        if (
            isinstance(layout, DTensor)
            and dist.is_available()
            and dist.is_initialized()
        ):
            for axis, placement in enumerate(layout.placements):
                if isinstance(placement, Shard):
                    contribution = _all_reduce_scalar(
                        contribution, reduction, layout.device_mesh.get_group(axis)
                    )
        contribution = contribution.to(accumulator.device)
        if is_inf:
            accumulator = torch.maximum(accumulator, contribution)
        else:
            accumulator = accumulator + contribution

    total = accumulator
    if not is_inf:
        total = total.pow(1.0 / norm_type)
    return total.to(first_local.device)


def clip_grad_norm_local_shards_(
    parameters: Iterable[torch.Tensor], max_norm: float, norm_type: float = 2.0
) -> torch.Tensor:
    """Clip CPU-offloaded DTensor gradients by their global norm."""
    parameters = list(parameters)
    total = get_grad_norm_local_shards_(parameters, norm_type=norm_type)
    coefficient = (float(max_norm) / (total + 1e-6)).clamp(max=1.0)
    for parameter in parameters:
        if parameter.grad is not None:
            local = _local_gradient(parameter.grad)
            local.detach().mul_(coefficient.to(device=local.device))
    return total


def ep_local_parameter_ids(model: torch.nn.Module) -> set[int]:
    """Return trainable parameters whose identity is local to an EP rank."""
    parameters: set[int] = set()
    for module in model.modules():
        base = getattr(module, "num_local_experts", None)
        global_count = getattr(module, "num_experts_global", None)
        if getattr(module, "_ep_lora_sharded", False) or (
            base is not None and global_count is not None and base < global_count
        ):
            parameters.update(id(parameter) for parameter in module.parameters())
    return parameters


def _mesh_axis_global_names(mesh, mesh_axis: int, global_mesh) -> set[str]:
    """Names of the ``global_mesh`` axes that ``mesh``'s ``mesh_axis`` group spans."""
    names = global_mesh.mesh_dim_names or ()
    ranks = dist.get_process_group_ranks(mesh.get_group(mesh_axis))
    coordinates = [(global_mesh.mesh == rank).nonzero()[0] for rank in ranks]
    return {
        name
        for axis, name in enumerate(names)
        if len({int(coordinate[axis]) for coordinate in coordinates}) > 1
    }


def _mesh_covered_axes(mesh, global_mesh) -> set[str]:
    if global_mesh is None:
        return set(mesh.mesh_dim_names or ())
    covered: set[str] = set()
    for mesh_axis in range(mesh.ndim):
        covered |= _mesh_axis_global_names(mesh, mesh_axis, global_mesh)
    return covered


def _owns_omitted_mesh_axes(mesh, global_mesh) -> bool:
    if global_mesh is None:
        return True
    coordinate = global_mesh.get_coordinate()
    if coordinate is None:
        return False
    covered = _mesh_covered_axes(mesh, global_mesh)
    return all(
        name in covered or coordinate[axis] == 0
        for axis, name in enumerate(global_mesh.mesh_dim_names or ())
    )


def _is_ep_plain_owner(global_mesh) -> bool:
    if global_mesh is None:
        return True
    coordinate = global_mesh.get_coordinate()
    if coordinate is None:
        return False
    return all(
        name == "ep" or coordinate[axis] == 0
        for axis, name in enumerate(global_mesh.mesh_dim_names or ())
    )


def _owns_dtensor_local_piece(
    gradient: DTensor, *, ep_local: bool, global_mesh
) -> bool:
    coordinate = gradient.device_mesh.get_coordinate()
    if coordinate is None or any(
        isinstance(placement, Replicate) and coordinate[axis] != 0
        for axis, placement in enumerate(gradient.placements)
    ):
        return False
    return ep_local or _owns_omitted_mesh_axes(gradient.device_mesh, global_mesh)


def get_grad_norm_ep_local_shards_(
    parameters: Iterable[torch.Tensor],
    norm_type: float = 2.0,
    *,
    ep_local_parameters: set[int],
    global_mesh=None,
) -> torch.Tensor:
    """Return an ownership-filtered global norm without modifying gradients."""
    parameters = list(parameters)
    pairs = [
        (parameter, parameter.grad)
        for parameter in parameters
        if parameter.grad is not None
    ]
    norm_type = float(norm_type)
    if norm_type <= 0 or math.isnan(norm_type):
        raise ValueError(f"norm_type must be positive or inf, got {norm_type}")
    for _, gradient in pairs:
        if isinstance(gradient, DTensor) and any(
            isinstance(placement, Partial) for placement in gradient.placements
        ):
            raise NotImplementedError(
                "EP CPU-offloaded DTensor gradient clipping does not support Partial placements"
            )

    first_local = _local_gradient(pairs[0][1]) if pairs else torch.zeros(())
    accumulator = torch.zeros((), dtype=torch.float32, device=first_local.device)
    is_inf = math.isinf(norm_type)
    for parameter, gradient in pairs:
        owns_piece = (
            _owns_dtensor_local_piece(
                gradient,
                ep_local=id(parameter) in ep_local_parameters,
                global_mesh=global_mesh,
            )
            if isinstance(gradient, DTensor)
            else id(parameter) in ep_local_parameters
            and _is_ep_plain_owner(global_mesh)
            or id(parameter) not in ep_local_parameters
            and (
                not (dist.is_available() and dist.is_initialized())
                or dist.get_rank() == 0
            )
        )
        if not owns_piece:
            continue
        value = _local_gradient(gradient).detach().to(torch.float32).abs()
        if is_inf:
            accumulator = torch.maximum(
                accumulator, value.max() if value.numel() else value.new_zeros(())
            )
        else:
            accumulator = accumulator + value.pow(norm_type).sum()

    reduction = dist.ReduceOp.MAX if is_inf else dist.ReduceOp.SUM
    total = _all_reduce_scalar(accumulator, reduction)
    if not is_inf:
        total = total.pow(1.0 / norm_type)
    return total.to(first_local.device)


def clip_grad_norm_ep_local_shards_(
    parameters: Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    *,
    ep_local_parameters: set[int],
    global_mesh=None,
) -> torch.Tensor:
    """Clip an EP/FSDP model by one ownership-filtered global gradient norm."""
    parameters = list(parameters)
    total = get_grad_norm_ep_local_shards_(
        parameters,
        norm_type=norm_type,
        ep_local_parameters=ep_local_parameters,
        global_mesh=global_mesh,
    )
    coefficient = (float(max_norm) / (total + 1e-6)).clamp(max=1.0)
    for parameter in parameters:
        if parameter.grad is not None:
            local = _local_gradient(parameter.grad)
            local.detach().mul_(coefficient.to(device=local.device))
    return total


@dataclass(frozen=True)
class ExpertLayout:
    """Where the experts sit in a fused expert parameter.

    ``dim`` is the axis of the logical tensor (this rank's tensor under expert
    parallelism) that carries the experts, and ``num_experts`` how many experts that
    tensor holds. Position ``p`` along ``dim`` belongs to expert ``p // group``
    (expert-major, e.g. ``gate_up_proj`` with ``group=1`` or an expert LoRA A with
    ``group=r``), or to expert ``p % num_experts`` when ``interleaved`` (rank-major,
    e.g. an expert LoRA B ``[out, r * E]``).
    """

    dim: int
    num_experts: int
    group: int = 1
    interleaved: bool = False

    def expert_ids(self, offset: int, length: int, device) -> torch.Tensor:
        positions = torch.arange(offset, offset + length, device=device)
        if self.interleaved:
            return positions % self.num_experts
        return positions // self.group


def expert_parameter_layouts(model: torch.nn.Module) -> dict[int, ExpertLayout]:
    """Expert layouts of a model's fused expert parameters, keyed by ``id(parameter)``.

    Experts modules are found the way the expert-parallel plugin finds them (3D
    ``gate_up_proj`` / ``down_proj`` with the experts on dim 0), with or without
    expert parallelism; their per-expert biases and PEFT expert LoRA (``lora_A``
    ``[E * r, in]`` expert-major, ``lora_B`` ``[out, r * E]`` rank-major) are included.
    """
    from axolotl.integrations.expert_parallel.shard import (
        _detect_experts_modules,
        _is_param_wrapper,
        _real_experts_base,
    )

    layouts: dict[int, ExpertLayout] = {}
    counts: dict[int, int] = {}
    for _, module in _detect_experts_modules(model):
        count = int(
            module.num_local_experts
            if getattr(module, "_is_expert_parallel", False)
            and getattr(module, "num_local_experts", None)
            else module.gate_up_proj.shape[0]
        )
        counts[id(module)] = count
        for name in (
            "gate_up_proj",
            "down_proj",
            "gate_up_proj_bias",
            "down_proj_bias",
        ):
            parameter = getattr(module, name, None)
            if (
                isinstance(parameter, torch.nn.Parameter)
                and parameter.dim() >= 1
                and parameter.shape[0] == count
            ):
                layouts[id(parameter)] = ExpertLayout(0, count)
    for module in model.modules():
        if not _is_param_wrapper(module):
            continue
        base = _real_experts_base(module)
        lora_count = counts.get(id(base)) if base is not None else None
        if not lora_count:
            continue
        count = lora_count
        for adapter in getattr(module, "lora_A", {}).values():
            weight = adapter.weight
            if weight.dim() == 2 and weight.shape[0] % count == 0:
                layouts[id(weight)] = ExpertLayout(0, count, weight.shape[0] // count)
        for adapter in getattr(module, "lora_B", {}).values():
            weight = adapter.weight
            if weight.dim() == 2 and weight.shape[1] % count == 0:
                layouts[id(weight)] = ExpertLayout(1, count, interleaved=True)
    return layouts


@dataclass
class GradNorms:
    """Per-tensor (and per-expert) global gradient norms.

    ``norms[offsets[i] : offsets[i] + sizes[i]]`` belongs to ``parameters[i]``: one
    norm for an ordinary tensor, one per expert (of this rank's experts) for a fused
    expert tensor with a known :class:`ExpertLayout`.
    """

    parameters: list[torch.Tensor]
    norms: torch.Tensor
    offsets: list[int]
    sizes: list[int]
    layouts: list[ExpertLayout | None]


def _local_offset(layout_tensor, dim: int) -> int:
    if not isinstance(layout_tensor, DTensor):
        return 0
    from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

    _, offset = compute_local_shape_and_global_offset(
        layout_tensor.shape, layout_tensor.device_mesh, layout_tensor.placements
    )
    return int(offset[dim]) if len(offset) > dim else 0


def get_grad_norms_per_tensor_(
    parameters: Iterable[torch.Tensor],
    *,
    ep_local_parameters: set[int] | frozenset[int] = frozenset(),
    global_mesh=None,
    expert_layouts: dict[int, ExpertLayout] | None = None,
) -> GradNorms:
    """Return each gradient's global 2-norm, per expert for fused expert tensors.

    The norm of a DTensor gradient is that of the whole logical tensor, identical on
    every rank of its mesh: the local piece's sums of squares are all-reduced over
    the mesh axes with a ``Shard`` placement (``Replicate`` axes already hold the same
    values). Parameters are bucketed by layout and each bucket is reduced as one
    vector, so a step costs one collective per sharded axis of each distinct layout,
    never one per tensor. Plain tensors are whole (or replicated) and need no
    collective.

    A parameter in ``expert_layouts`` gets one norm per expert: its local piece's
    squares are summed per expert (by the experts' positions along the layout's
    ``dim``, wherever the data sharding cuts) and reduced like any other tensor.
    Under expert parallelism each rank's expert tensor holds whole, distinct experts,
    so these norms never need the ``ep`` axis; an expert tensor sharded over a mesh
    axis that spans ``ep`` of ``global_mesh`` raises. Any other expert-parallel
    tensor (``ep_local_parameters`` without a layout) is treated as one fused tensor
    over all EP ranks: its squares are also summed over the ``ep`` axis.

    A DTensor parameter that requires grad joins its bucket even when its gradient is
    ``None`` on this rank, so all ranks issue the same collectives; it is returned
    when it has a gradient on any rank. Local pieces may be CPU-offloaded; norms live
    on the device of the first local gradient.
    """
    expert_layouts = expert_layouts or {}
    entries = []
    for parameter in parameters:
        gradient = parameter.grad
        layout = (
            gradient
            if isinstance(gradient, DTensor)
            else parameter
            if isinstance(parameter, DTensor)
            else None
        )
        if gradient is None and not (layout is not None and parameter.requires_grad):
            continue
        if layout is not None and any(
            isinstance(placement, Partial) for placement in layout.placements
        ):
            raise NotImplementedError(
                "Per-tensor gradient clipping does not support Partial DTensor "
                "placements; reduce the gradients to Shard or Replicate first"
            )
        entries.append((parameter, gradient, layout, expert_layouts.get(id(parameter))))
    if not entries:
        return GradNorms([], torch.zeros(0), [], [], [])

    locals_ = [
        _local_gradient(gradient) if gradient is not None else None
        for _, gradient, _, _ in entries
    ]
    device = next(
        (local.device for local in locals_ if local is not None),
        next(
            (
                layout.to_local().device
                for _, _, layout, _ in entries
                if isinstance(layout, DTensor)
            ),
            torch.device("cpu"),
        ),
    )
    sizes = [experts.num_experts if experts else 1 for *_, experts in entries]
    offsets = [0]
    for size in sizes[:-1]:
        offsets.append(offsets[-1] + size)
    squares = torch.zeros(sum(sizes), dtype=torch.float32, device=device)
    present = torch.tensor(
        [local is not None for local in locals_], dtype=torch.float32, device=device
    )

    whole: dict[torch.device, list[int]] = {}
    for index, local in enumerate(locals_):
        if local is None:
            continue
        experts = entries[index][3]
        if experts is None:
            whole.setdefault(local.device, []).append(index)
            continue
        dims = [dim for dim in range(local.dim()) if dim != experts.dim]
        values = local.detach().to(torch.float32).square()
        values = values.sum(dim=dims) if dims else values
        ids = experts.expert_ids(
            _local_offset(entries[index][2], experts.dim),
            local.shape[experts.dim],
            device,
        )
        squares[offsets[index] : offsets[index] + sizes[index]].index_add_(
            0, ids, values.to(device)
        )
    for indices in whole.values():
        norms = torch._foreach_norm(
            [locals_[index].detach() for index in indices],  # type: ignore[union-attr]
            2,
            dtype=torch.float32,
        )
        squares[torch.tensor([offsets[index] for index in indices], device=device)] = (
            torch.stack(norms).to(device).square()
        )

    distributed = dist.is_available() and dist.is_initialized()
    has_ep = global_mesh is not None and "ep" in (global_mesh.mesh_dim_names or ())
    buckets: dict[tuple, list[int]] = {}
    for index, (parameter, _, layout, experts) in enumerate(entries):
        if not distributed:
            break
        fused_ep = has_ep and experts is None and id(parameter) in ep_local_parameters
        mesh = None
        axes: tuple[int, ...] = ()
        if isinstance(layout, DTensor):
            mesh = layout.device_mesh
            axes = tuple(
                axis
                for axis, placement in enumerate(layout.placements)
                if isinstance(placement, Shard)
            )
            if fused_ep and any(
                "ep" in _mesh_axis_global_names(mesh, axis, global_mesh)
                for axis in axes
            ):
                fused_ep = False  # its mesh already spans the ep axis
        if axes or fused_ep:
            key = (mesh, axes, fused_ep, experts is not None)
            buckets.setdefault(key, []).append(index)
    for (mesh, axes, fused_ep, per_expert), indices in buckets.items():
        if (
            per_expert
            and has_ep
            and any(
                "ep" in _mesh_axis_global_names(mesh, axis, global_mesh)
                for axis in axes
            )
        ):
            raise NotImplementedError(
                "an expert-parallel gradient is sharded over a mesh axis that "
                "spans the ep axis; its per-expert norms would mix experts"
            )
        positions = torch.cat(
            [
                torch.arange(offsets[index], offsets[index] + sizes[index])
                for index in indices
            ]
        ).to(device)
        flags = torch.tensor(indices, device=device)
        vector = torch.cat([squares[positions], present[flags]])
        groups = [mesh.get_group(axis) for axis in axes]
        if fused_ep:
            groups.append(global_mesh["ep"].get_group())
        for group in groups:
            vector = _all_reduce_scalar(vector, dist.ReduceOp.SUM, group)
        vector = vector.to(device)
        squares[positions] = vector[: len(positions)]
        present[flags] = vector[len(positions) :]

    keep = torch.nonzero(present > 0).flatten().tolist()
    kept_positions = torch.cat(
        [torch.arange(offsets[index], offsets[index] + sizes[index]) for index in keep]
        or [torch.zeros(0, dtype=torch.long)]
    ).to(device)
    kept_offsets = [0]
    for index in keep[:-1]:
        kept_offsets.append(kept_offsets[-1] + sizes[index])
    return GradNorms(
        [entries[index][0] for index in keep],
        squares[kept_positions].sqrt(),
        kept_offsets if keep else [],
        [sizes[index] for index in keep],
        [entries[index][3] for index in keep],
    )


def scale_grads_per_tensor_(result: GradNorms, coefficients: torch.Tensor) -> None:
    """Multiply each local gradient piece by its coefficient(s) in place.

    ``coefficients`` follows ``result.norms``; an expert tensor's slices are scaled by
    their own expert's coefficient.
    """
    by_device: dict[torch.device, tuple[list, list]] = {}
    for parameter, offset, size, experts in zip(
        result.parameters, result.offsets, result.sizes, result.layouts, strict=True
    ):
        if parameter.grad is None:
            continue
        local = _local_gradient(parameter.grad).detach()
        if experts is None:
            grads, coefs = by_device.setdefault(local.device, ([], []))
            grads.append(local)
            coefs.append(coefficients[offset])
            continue
        ids = experts.expert_ids(
            _local_offset(
                parameter.grad if isinstance(parameter.grad, DTensor) else parameter,
                experts.dim,
            ),
            local.shape[experts.dim],
            coefficients.device,
        )
        factor = coefficients[offset : offset + size][ids].to(local.device)
        shape = [1] * local.dim()
        shape[experts.dim] = -1
        local.mul_(factor.view(shape))
    for device, (grads, coefs) in by_device.items():
        values = torch.stack(coefs).to(device)
        torch._foreach_mul_(grads, list(values.unbind()))
