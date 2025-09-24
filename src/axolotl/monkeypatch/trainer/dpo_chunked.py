"""Monkeypatch helpers to reduce fp32 materialization during DPO training."""

from __future__ import annotations

from contextlib import contextmanager
from types import MethodType
from typing import Iterable

import torch
from trl import DPOTrainer

_PATCHED = False


def _iter_patch_targets(model) -> Iterable[torch.nn.Module]:
    current = model
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "module", None)


def _resolve_unwrapped_forward(module):
    forward = getattr(module, "forward", None)
    if forward is None:
        return None

    if hasattr(forward, "__wrapped__"):
        unwrapped = forward.__wrapped__
        return MethodType(unwrapped, module)

    original = getattr(module, "_original_forward", None)
    if original is None:
        return None

    func = original.__func__ if hasattr(original, "__func__") else original
    return MethodType(func, module)


@contextmanager
def _temporarily_disable_output_fp32(model):
    patched = []
    for target in _iter_patch_targets(model):
        replacement = _resolve_unwrapped_forward(target)
        if replacement is None:
            continue
        patched.append((target, target.forward, replacement))

    try:
        for module, _, replacement in patched:
            module.forward = replacement
        yield
    finally:
        for module, original_forward, _ in reversed(patched):
            module.forward = original_forward


def _cast_fp32_outputs(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not isinstance(output, dict):
        return output

    for key, value in output.items():
        if torch.is_tensor(value) and value.dtype in (torch.float16, torch.bfloat16):
            output[key] = value.float()
    return output


def patch_dpo_disable_output_fp32():
    """Patch TRL's DPOTrainer to skip Accelerate's convert_to_fp32 wrapper when requested."""
    global _PATCHED
    if _PATCHED:
        return

    original_concatenated_forward = DPOTrainer.concatenated_forward

    def patched_concatenated_forward(self, model, batch, is_ref_model: bool = False):
        if not getattr(self.args, "disable_output_fp32", False):
            return original_concatenated_forward(
                self, model, batch, is_ref_model=is_ref_model
            )

        with _temporarily_disable_output_fp32(model):
            result = original_concatenated_forward(
                self, model, batch, is_ref_model=is_ref_model
            )
        return _cast_fp32_outputs(result)

    DPOTrainer.concatenated_forward = patched_concatenated_forward
    _PATCHED = True
