"""Verify the epilogue sonicmoe will compute against the one the model declares.

`@use_experts_implementation` installs `_apply_gate` as the model's declared expert
epilogue, and every transformers experts backend calls it except `sonicmoe`, which
resolves the epilogue from `config.hidden_act` instead. A model whose epilogue that
string cannot describe (gpt_oss, deepseek_v4) runs silently-wrong math rather than
raising.

The check is numeric rather than a model allowlist: an allowlist rots as new
architectures land, and `hidden_act` is not a usable signal even in principle
(MiniMaxM3VLTextConfig overwrites the checkpoint's "swigluoai" with "silu" because
it has to be a real ACT2FN key).
"""

from __future__ import annotations

import torch

_PROBE_WIDTH = 16
# Per instance, not per class: `_apply_gate` closes over `self.limit` / `self.alpha`.
_VERDICT_ATTR = "_sonicmoe_epilogue_verdict"


def _describe(experts_module) -> str:
    cls = type(experts_module).__name__
    parts = []
    for attr in ("limit", "alpha", "swiglu_alpha", "swiglu_limit"):
        val = getattr(experts_module, attr, None)
        if val is not None:
            parts.append(f"{attr}={val}")
    return f"{cls}({', '.join(parts)})" if parts else cls


def _mismatch(experts_module, act: str, *, concat: bool, limit: float | None) -> bool:
    """True if ``(act, limit, concat)`` fails to reproduce ``_apply_gate``."""
    from .nvfp4 import gated_activation

    # Span past the clamp so a missing clamp cannot pass by luck.
    scale = 3.0 * (limit if limit is not None else 2.0)
    x = torch.randn(256, _PROBE_WIDTH, dtype=torch.float32) * scale
    try:
        want = experts_module._apply_gate(x)
        got = gated_activation(x, act, concat=concat, limit=limit)
    except Exception:  # noqa: BLE001
        # Either side failing means an epilogue that also fails in eager, or an activation
        # outside `gated_activation`'s superset of upstream's ACT_MAP. Both raise later
        # with a better message than this check could give.
        return False
    return want.shape != got.shape or not torch.allclose(
        want, got, atol=1e-5, rtol=1e-4
    )


def check_epilogue(
    experts_module, act: str, *, concat: bool, limit: float | None, path: str
) -> None:
    """Raise if ``path`` would not reproduce the module's declared ``_apply_gate``.

    Skipped under `torch.compile` (Dynamo folds the branch away): the probe is
    data-dependent and would break the single-graph guarantee. Real runs are covered
    eagerly by `check_model_epilogues` at plugin setup.
    """
    if torch.compiler.is_compiling():
        return
    # No declared epilogue means nothing to contradict.
    if not hasattr(experts_module, "_apply_gate"):
        return

    # `__func__` because attribute access rebuilds the bound method object each time.
    gate_fn = experts_module._apply_gate
    key = (getattr(gate_fn, "__func__", gate_fn), act, limit, concat, path)
    cached = getattr(experts_module, _VERDICT_ATTR, None)
    if cached is not None and cached[0] == key:
        if cached[1]:
            raise ValueError(cached[1])
        return

    reason = None
    if _mismatch(experts_module, act, concat=concat, limit=limit):
        from transformers.integrations.moe import _default_apply_gate

        computed = f"{act!r}" + (
            f" with clamp(+/-{limit})" if limit is not None else " with no clamp"
        )
        declares = (
            "The model declares a custom `_apply_gate`"
            if getattr(type(experts_module), "_apply_gate", None)
            is not _default_apply_gate
            else "The model's `_apply_gate` disagrees with the resolved activation"
        )
        reason = (
            f"sonicmoe would compute the wrong expert math for {_describe(experts_module)}. "
            f"{declares}, but the sonicmoe {path} path computes {computed}, and the "
            f"fused kernel has no activation that can express this epilogue.\n"
            "Use `expert_backend: scattermoe` (which implements these epilogues), or "
            "drop the kernel expert backend to fall back to the built-in "
            "implementation."
        )
    setattr(experts_module, _VERDICT_ATTR, (key, reason))
    if reason:
        raise ValueError(reason)


def check_model_epilogues(model) -> int:
    """Eagerly probe every experts module in ``model``. Returns the number checked.

    Runs at plugin setup so an unrepresentable epilogue fails before training starts
    rather than mid-forward.
    """
    from .nvfp4 import is_nvfp4_param, resolve_gated_activation

    checked = 0
    for module in model.modules():
        if not hasattr(module, "_apply_gate") or not hasattr(module, "gate_up_proj"):
            continue
        # `_apply_gate` is not the contract for non-gated experts (the backends call
        # `act_fn` directly), and the forward already rejects them by `has_gate`.
        if not getattr(module, "has_gate", True):
            continue
        nvfp4 = is_nvfp4_param(module.gate_up_proj)
        check_epilogue(
            module,
            resolve_gated_activation(module.config),
            concat=getattr(module, "is_concatenated", True),
            limit=getattr(module, "limit", None) if nvfp4 else None,
            path="NVFP4 grouped" if nvfp4 else "dense",
        )
        checked += 1
    return checked
