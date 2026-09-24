"""DDP preparation for frozen native TorchAO NVFP4 parameters."""

from __future__ import annotations


def prepare_native_nvfp4_components(model, device) -> tuple[str, ...]:
    """Validate and synchronize frozen native NVFP4 parameter components."""
    import torch
    import torch.distributed as dist

    if not dist.is_initialized() or dist.get_world_size() == 1:
        return ()
    weights = [
        (name, param)
        for name, param in model.named_parameters()
        if type(param).__name__ == "NVFP4Tensor"
    ]

    def semantics(value):
        if isinstance(value, dict):
            return tuple(sorted((str(key), repr(item)) for key, item in value.items()))
        return repr(value)

    components = ("qdata", "scale", "per_tensor_scale")
    errors = []
    local = []
    for name, param in weights:
        if param.requires_grad:
            errors.append(f"{name} is trainable")
        entry = [name, tuple(param.shape), param.block_size, str(param.dtype)]
        entry.append(semantics(getattr(param, "act_quant_kwargs", None)))
        for component in components:
            value = getattr(param, component, None)
            entry.append(
                None if value is None else (tuple(value.shape), str(value.dtype))
            )
        local.append(tuple(entry))
    gathered = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (local, errors))
    if any(other_errors for _, other_errors in gathered):
        raise ValueError("Native NVFP4 DDP requires frozen base weights on every rank")
    if any(entry != local for entry, _ in gathered):
        raise ValueError(
            "Native NVFP4 DDP requires identical component layouts on every rank"
        )
    if not weights:
        return ()
    with torch.no_grad():
        for _, param in weights:
            for component in components:
                value = getattr(param, component, None)
                if value is None:
                    continue
                temporary = value if value.device == device else value.to(device)
                broadcast_value = temporary.contiguous()
                dist.broadcast(broadcast_value.reshape(-1).view(torch.uint8), src=0)
                if broadcast_value is not value:
                    value.copy_(broadcast_value.to(value.device))
    return tuple(name for name, _ in weights)


def prepare_native_nvfp4_ddp(model, device) -> bool:
    """Synchronize frozen native NVFP4 components before DDP wraps the model."""
    weights = prepare_native_nvfp4_components(model, device)
    if not weights:
        return False
    ignored = set(getattr(model, "_ddp_params_and_buffers_to_ignore", ()))
    ignored.update(weights)
    model._ddp_params_and_buffers_to_ignore = ignored
    return True
