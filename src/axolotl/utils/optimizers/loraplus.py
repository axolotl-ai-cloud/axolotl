"""LoRA+ parameter-group helpers for optimizer factories."""

from collections.abc import Callable, Iterable
from typing import Any

from torch import Tensor

LoRAPlusEligibility = Callable[[str, Tensor, dict[str, Any]], bool]


def _lora_factor(name: str) -> str | None:
    components = name.split(".")
    if "lora_A" in components:
        return "A"
    if "lora_B" in components:
        return "B"
    return None


def apply_loraplus_lr_groups(
    param_groups: list[dict[str, Any]],
    named_parameters: Iterable[tuple[str, Tensor]],
    *,
    default_lr: float,
    loraplus_lr_ratio: float | None,
    eligible: LoRAPlusEligibility,
) -> list[dict[str, Any]]:
    """Split eligible 2D LoRA A/B factors while preserving each group's metadata.

    Factories retain ownership of routing and fallback policy through ``eligible``.
    Use this before optimizer construction; it does not repartition optimizer state.
    The input groups and their parameter lists are never mutated.
    """
    if loraplus_lr_ratio is None:
        return [{**group, "params": list(group["params"])} for group in param_groups]

    names_by_param_id = {}
    for name, param in named_parameters:
        names_by_param_id.setdefault(id(param), name)
    result = []
    for group in param_groups:
        if not group["params"]:
            result.append({**group, "params": []})
            continue
        regular_params = []
        lora_a_params = []
        lora_b_params = []
        for param in group["params"]:
            name = names_by_param_id.get(id(param))
            factor = _lora_factor(name) if name is not None else None
            if factor is None or param.ndim != 2 or not eligible(name, param, group):
                regular_params.append(param)
            elif factor == "A":
                lora_a_params.append(param)
            else:
                lora_b_params.append(param)

        if regular_params:
            result.append({**group, "params": regular_params})
        group_lr = group.get("lr", default_lr)
        if lora_a_params:
            result.append({**group, "params": lora_a_params, "lr": group_lr})
        if lora_b_params:
            lora_b_group = {
                **group,
                "params": lora_b_params,
                "lr": group_lr * loraplus_lr_ratio,
            }
            if "initial_lr" in group:
                lora_b_group["initial_lr"] = group["initial_lr"] * loraplus_lr_ratio
            result.append(lora_b_group)
    return result
