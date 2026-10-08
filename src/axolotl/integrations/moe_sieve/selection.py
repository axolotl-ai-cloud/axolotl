"""Routing profiles and deterministic per-layer expert selection."""

import math

import torch


def packed_experts(model):
    """Find packed routed-expert modules with the standard dispatch signature."""
    result = {}
    for name, module in model.named_modules():
        if not name or not hasattr(module, "num_experts"):
            continue
        parameters = dict(module.named_parameters(recurse=False))
        shapes = {
            key: list(param.shape)
            for key, param in parameters.items()
            if param.ndim == 3 and param.shape[0] == module.num_experts
        }
        if shapes:
            result[name] = (module, shapes)
    if not result:
        raise ValueError("MoE-Sieve found no packed expert parameters")
    return result


def select_experts(counts, fraction):
    """Select by descending count, breaking ties by expert ID."""
    if not 0 < fraction <= 1:
        raise ValueError("Expert fraction must be in (0, 1]")
    if not counts or any(type(count) is not int or count < 0 for count in counts):
        raise ValueError("Routing counts must be nonnegative integers")
    if not sum(counts):
        raise ValueError("No routed tokens observed for an expert layer")
    size = math.floor(fraction * len(counts))
    if not size:
        raise ValueError("Expert fraction selects zero experts")
    return sorted(sorted(range(len(counts)), key=lambda i: (-counts[i], i))[:size])


def validate_selection(model, selection):
    """Reject stale profiles before constructing any adapters."""
    modules = packed_experts(model)
    if set(selection) != set(modules):
        raise ValueError(
            "MoE-Sieve selection must cover exactly the model's packed expert modules"
        )
    for name, (module, shapes) in modules.items():
        spec = selection[name]
        ids = spec["selected_experts"]
        if (
            spec["parameter_shapes"] != shapes
            or spec["num_experts"] != module.num_experts
        ):
            raise ValueError(f"MoE-Sieve parameter layout mismatch for {name}")
        if (
            not ids
            or any(type(i) is not int or not 0 <= i < module.num_experts for i in ids)
            or len(set(ids)) != len(ids)
        ):
            raise ValueError(f"Invalid selected expert IDs for {name}")
    return modules


def profile_routing(model, batches, fraction=0.25):
    """Count dispatched IDs from packed-expert inputs on unsharded models."""
    modules = packed_experts(model)
    counts = {
        name: torch.zeros(module.num_experts, dtype=torch.long)
        for name, (module, _) in modules.items()
    }
    handles = []
    mask = None

    def make_hook(name):
        def count_routes(module, args, kwargs):
            candidates = [
                value
                for value in (*args, *kwargs.values())
                if isinstance(value, torch.Tensor)
                and value.dtype == torch.long
                and value.ndim == 2
            ]
            if len(candidates) != 1:
                raise ValueError(
                    f"Unsupported routing inputs for {name}: expected one 2D int64 expert-ID tensor"
                )
            ids = candidates[0].detach()
            if mask is not None:
                if mask.numel() != ids.shape[0]:
                    raise ValueError(
                        f"Routing token count does not match attention mask for {name}"
                    )
                ids = ids[mask.to(ids.device)]
            if ((ids < 0) | (ids > module.num_experts)).any():
                raise ValueError(f"Invalid routed expert IDs for {name}")
            ids = ids[ids < module.num_experts]
            counts[name].add_(
                torch.bincount(ids.flatten(), minlength=module.num_experts).cpu()
            )

        return count_routes

    training = {module: module.training for module in model.modules()}
    try:
        for name, (module, _) in modules.items():
            handles.append(
                module.register_forward_pre_hook(make_hook(name), with_kwargs=True)
            )
        model.eval()
        with torch.no_grad():
            for batch in batches:
                attention_mask = batch.get("attention_mask")
                mask = (
                    attention_mask.reshape(-1).bool()
                    if attention_mask is not None
                    else None
                )
                model(**{key: value for key, value in batch.items() if key != "labels"})
    finally:
        for handle in handles:
            handle.remove()
        for module, mode in training.items():
            module.training = mode
    return {
        name: {
            "num_experts": module.num_experts,
            "parameter_shapes": shapes,
            "counts": counts[name].tolist(),
            "selected_experts": select_experts(counts[name].tolist(), fraction),
        }
        for name, (module, shapes) in modules.items()
    }
