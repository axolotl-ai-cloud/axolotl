"""Route PEFT ``target_parameters`` LoRA on experts-interface MoEs to the fused kernel.

For models whose MoE is dispatched through transformers' ExpertsInterface
(``@use_experts_implementation`` — e.g. Gemma 4, DiffusionGemma), PEFT wraps the
*experts module* in a ``ParamWrapper`` chain. The decoder layer then calls
``experts(...)`` → ``ParamWrapper.forward`` → ``_activate_lora`` which materializes the
full merged weight ``base + delta`` before ScatterMoE runs. That defeats ScatterMoE's
fused LoRA kernel (which takes A/B separately, dequantizing only the active experts) and
fails outright on quantized bases (``aten.add`` on NVFP4/MXFP4).

This patches ``ParamWrapper.forward`` so that, when the wrapped base is a ScatterMoE
experts module, it walks the wrapper chain, hands the LoRA A/B to the base via
``_scattermoe_lora`` (in ScatterMoE layout), and calls the base forward directly —
bypassing the parametrization merge entirely. The wrappers stay in the module tree, so
PEFT save/load and optimizer tracking are unaffected.

This mirrors what ``HFScatterMoEGatedMLP`` already does for SparseMoeBlock models (walk
the wrapper, fuse the LoRA), bringing the experts-interface path to parity.
"""

from __future__ import annotations

# Implementations whose experts forward consumes ``module._scattermoe_lora`` (the fused
# per-row LoRA kernel). ``deep_ep_scattermoe`` is the EP composite: its ``_scattermoe_local``
# stage calls ``scattermoe_experts_forward_ep``, which reads the same attribute — so the
# fastpath must engage under EP too, else the LoRA falls back to PEFT's parametrize merge
# (which can't add a full-expert delta onto the EP-sharded weight).
_SCATTERMOE_IMPLS = frozenset({"scattermoe", "deep_ep_scattermoe"})


def _is_scattermoe_experts(module) -> bool:
    cfg = getattr(module, "config", None)
    impl = getattr(cfg, "_experts_implementation", None)
    return impl in _SCATTERMOE_IMPLS and hasattr(module, "gate_up_proj")


def patch_paramwrapper_fastpath() -> None:
    """Idempotently patch ``peft...ParamWrapper.forward`` for ScatterMoE experts."""
    try:
        from peft.tuners.lora.layer import ParamWrapper
    except (ImportError, AttributeError):
        return

    if getattr(ParamWrapper.forward, "_scattermoe_fastpath", False):
        return

    from .layers import _convert_smoe_lora
    from .parallel_linear_lora import get_lora_params_from_wrapper

    _orig_forward = ParamWrapper.forward

    def _fusable(wrapper) -> bool:
        # Fused kernel needs exactly one adapter on a raw (un-merged) base; else defer to PEFT.
        if getattr(wrapper, "disable_adapters", False) or getattr(
            wrapper, "merged", False
        ):
            return False
        return len(getattr(wrapper, "active_adapters", [])) == 1

    def forward(self, x, *args, **kwargs):
        # Mixed-batch inference (adapter_names) is PEFT's domain; defer to it.
        if kwargs.get("adapter_names") is not None:
            return _orig_forward(self, x, *args, **kwargs)

        # one wrapper per targeted parameter
        wrappers = {}
        base = self
        while hasattr(base, "base_layer") and hasattr(base, "lora_A"):
            name = getattr(base, "parameter_name", None)
            if name is not None:
                wrappers[name] = base
            base = base.base_layer

        if not (
            _is_scattermoe_experts(base) and all(_fusable(w) for w in wrappers.values())
        ):
            return _orig_forward(self, x, *args, **kwargs)

        num_experts = getattr(base, "num_experts", None)
        sm_lora = {}
        for name, wrapper in wrappers.items():
            lora_A, lora_B, scaling = get_lora_params_from_wrapper(wrapper)
            if lora_A is None or num_experts is None:
                continue
            # Under EP the base is sliced to E_local experts but the adapter stays a single global
            # tensor over E_global experts; take THIS rank's local-expert block and the true LoRA
            # rank so the fused kernel reads the right experts. No-op when not EP-sharded.
            from .experts import _ep_local_expert_lora

            lora_A, lora_B, num_experts_local, rank = _ep_local_expert_lora(
                lora_A, lora_B, base
            )
            # PEFT keeps LoRA fp32; cast to activation dtype (grads still route to the fp32 params).
            lora_A = lora_A.to(x.dtype)
            lora_B = lora_B.to(x.dtype)
            sm_lora[name] = _convert_smoe_lora(
                lora_A, lora_B, num_experts_local, rank, scaling
            )

        base._scattermoe_lora = sm_lora
        # Loading can create multiple ExpertsInterface registries; the experts forward binds one as
        # a closure default that may differ from the one register_scattermoe_experts() populated, so
        # register into the one THIS module dispatches through.
        if not getattr(base, "_scattermoe_iface_ok", False):
            _fn = type(base).forward
            _cells = dict(
                zip(_fn.__code__.co_freevars, _fn.__closure__ or (), strict=False)
            )
            _ei = _cells.get("experts_interface")
            if _ei is not None:
                _ei = _ei.cell_contents
                if "scattermoe" not in _ei:
                    from .experts import scattermoe_experts_forward

                    _ei.register("scattermoe", scattermoe_experts_forward)
            base._scattermoe_iface_ok = True
        try:
            return base(x, *args, **kwargs)
        finally:
            base._scattermoe_lora = None

    forward._scattermoe_fastpath = True  # type: ignore[attr-defined]
    ParamWrapper.forward = forward
