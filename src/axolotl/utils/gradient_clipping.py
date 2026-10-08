"""Gradient clipping that keeps CPU-offloaded DTensor shards local."""

from __future__ import annotations

import math
from collections.abc import Iterable

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


def get_grad_norms_per_tensor_(
    parameters: Iterable[torch.Tensor],
    *,
    ep_local_parameters: set[int] | frozenset[int] = frozenset(),
    global_mesh=None,
) -> tuple[list[torch.Tensor], torch.Tensor]:
    """Return each gradient's global 2-norm without modifying the gradients.

    The norm of a DTensor gradient is that of the whole logical tensor, identical on
    every rank of its mesh: the local piece's sum of squares is all-reduced over the
    mesh axes with a ``Shard`` placement (``Replicate`` axes already hold the same
    values). Parameters are bucketed by ``(mesh, sharded axes)`` and each bucket is
    reduced as one vector, so a step costs one collective per sharded axis of each
    distinct layout, never one per tensor. Plain tensors are whole (or replicated)
    and need no collective.

    Expert-parallel tensors (``ep_local_parameters``) hold different experts on
    different EP ranks, so their norm is the norm of this rank's slice. Their meshes
    exclude the ``ep`` axis; reducing over a mesh axis that spans the ``ep`` axis of
    ``global_mesh`` would mix experts and raises instead.

    A DTensor parameter that requires grad joins its bucket's collective even when
    its gradient is ``None`` on this rank, so all ranks issue the same collectives;
    it is returned when it has a gradient on any rank of its sharded axes.

    Returns ``(parameters, norms)``: the parameters with a gradient and a float32
    vector of their norms, in ``parameters`` order. Local pieces may be CPU-offloaded;
    norms live on the device of the first local gradient.
    """
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
        entries.append((parameter, gradient, layout))
    if not entries:
        return [], torch.zeros(0)

    locals_ = [
        _local_gradient(gradient) if gradient is not None else None
        for _, gradient, _ in entries
    ]
    device = next(
        (local.device for local in locals_ if local is not None),
        next(
            (
                layout.to_local().device
                for _, _, layout in entries
                if isinstance(layout, DTensor)
            ),
            torch.device("cpu"),
        ),
    )
    squares = torch.zeros(len(entries), dtype=torch.float32, device=device)
    present = torch.tensor(
        [local is not None for local in locals_], dtype=torch.float32, device=device
    )
    by_device: dict[torch.device, list[int]] = {}
    for index, local in enumerate(locals_):
        if local is not None:
            by_device.setdefault(local.device, []).append(index)
    for indices in by_device.values():
        norms = torch._foreach_norm(
            [locals_[index].detach() for index in indices],  # type: ignore[union-attr]
            2,
            dtype=torch.float32,
        )
        squares[torch.tensor(indices, device=device)] = (
            torch.stack(norms).to(device).square()
        )

    distributed = dist.is_available() and dist.is_initialized()
    buckets: dict[tuple, list[int]] = {}
    for index, (_, _, layout) in enumerate(entries):
        if not isinstance(layout, DTensor):
            continue
        axes = tuple(
            axis
            for axis, placement in enumerate(layout.placements)
            if isinstance(placement, Shard)
        )
        if axes and distributed:
            buckets.setdefault((layout.device_mesh, axes), []).append(index)
    ep_axis = (
        global_mesh is not None and "ep" in (global_mesh.mesh_dim_names or ())
    ) and bool(ep_local_parameters)
    for (mesh, axes), indices in buckets.items():
        if ep_axis and any(
            id(entries[index][0]) in ep_local_parameters for index in indices
        ):
            for axis in axes:
                if "ep" in _mesh_axis_global_names(mesh, axis, global_mesh):
                    raise NotImplementedError(
                        "an expert-parallel gradient is sharded over a mesh axis that "
                        "spans the ep axis; its per-tensor norm would mix experts"
                    )
        position = torch.tensor(indices, device=device)
        vector = torch.cat([squares[position], present[position]])
        for axis in axes:
            vector = _all_reduce_scalar(vector, dist.ReduceOp.SUM, mesh.get_group(axis))
        vector = vector.to(device)
        squares[position] = vector[: len(indices)]
        present[position] = vector[len(indices) :]

    keep = torch.nonzero(present > 0).flatten()
    return (
        [entries[index][0] for index in keep.tolist()],
        squares[keep].sqrt(),
    )


def scale_grads_per_tensor_(
    parameters: list[torch.Tensor], coefficients: torch.Tensor
) -> None:
    """Multiply each parameter's local gradient piece by its coefficient in place."""
    by_device: dict[torch.device, tuple[list, list]] = {}
    for parameter, coefficient in zip(parameters, coefficients.unbind(), strict=True):
        if parameter.grad is None:
            continue
        local = _local_gradient(parameter.grad).detach()
        grads, coefs = by_device.setdefault(local.device, ([], []))
        grads.append(local)
        coefs.append(coefficient)
    for device, (grads, coefs) in by_device.items():
        values = torch.stack(coefs).to(device)
        torch._foreach_mul_(grads, list(values.unbind()))
