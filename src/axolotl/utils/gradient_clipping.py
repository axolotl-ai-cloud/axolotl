"""Gradient clipping that keeps CPU-offloaded DTensor shards local."""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Partial, Shard


def _local_gradient(gradient):
    return gradient.to_local() if isinstance(gradient, DTensor) else gradient


def _all_reduce_scalar(
    value: torch.Tensor, op: dist.ReduceOp, group=None
) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    if dist.get_backend(group) != "nccl" or value.device.type != "cpu":
        dist.all_reduce(value, op=op, group=group)
        return value
    staged = value.to(torch.device("cuda", torch.cuda.current_device()))
    dist.all_reduce(staged, op=op, group=group)
    return staged.to(value.device)


def has_cpu_offloaded_dtensor_gradients(parameters: Iterable[torch.Tensor]) -> bool:
    return any(
        isinstance(parameter.grad, DTensor)
        and parameter.grad.to_local().device.type == "cpu"
        for parameter in parameters
        if parameter.grad is not None
    )


def clip_grad_norm_local_shards_(
    parameters: Iterable[torch.Tensor], max_norm: float, norm_type: float = 2.0
) -> torch.Tensor:
    """Clip by a global norm without DTensor collectives on CPU-offloaded shards."""
    parameters = list(parameters)
    gradients = [
        parameter.grad for parameter in parameters if parameter.grad is not None
    ]
    if not gradients:
        return torch.tensor(0.0)

    norm_type = float(norm_type)
    if norm_type <= 0 or math.isnan(norm_type):
        raise ValueError(f"norm_type must be positive or inf, got {norm_type}")
    for gradient in gradients:
        if isinstance(gradient, DTensor) and any(
            isinstance(placement, Partial) for placement in gradient.placements
        ):
            raise NotImplementedError(
                "CPU-offloaded DTensor gradient clipping does not support Partial placements"
            )
    first_local = _local_gradient(gradients[0])
    accumulator = torch.zeros((), dtype=torch.float32, device=first_local.device)
    is_inf = math.isinf(norm_type)
    reduction = dist.ReduceOp.MAX if is_inf else dist.ReduceOp.SUM
    for gradient in gradients:
        local = _local_gradient(gradient)
        value = local.detach().to(torch.float32).abs()
        if is_inf:
            contribution = value.max() if value.numel() else value.new_zeros(())
        else:
            contribution = value.pow(norm_type).sum()
        if (
            isinstance(gradient, DTensor)
            and dist.is_available()
            and dist.is_initialized()
        ):
            for axis, placement in enumerate(gradient.placements):
                if isinstance(placement, Shard):
                    contribution = _all_reduce_scalar(
                        contribution, reduction, gradient.device_mesh.get_group(axis)
                    )
        contribution = contribution.to(accumulator.device)
        if is_inf:
            accumulator = torch.maximum(accumulator, contribution)
        else:
            accumulator = accumulator + contribution

    total = accumulator
    if not is_inf:
        total = total.pow(1.0 / norm_type)
    coefficient = (float(max_norm) / (total + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        local = _local_gradient(gradient)
        local.detach().mul_(coefficient.to(device=local.device))
    return total.to(first_local.device)
