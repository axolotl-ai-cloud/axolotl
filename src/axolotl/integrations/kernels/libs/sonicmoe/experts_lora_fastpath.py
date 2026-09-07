"""Route PEFT ``target_parameters`` expert LoRA to the sonicmoe grouped path.

Same problem and fix shape as ``scattermoe_lora/experts_lora_fastpath.py``: when PEFT
targets ``experts.gate_up_proj``/``experts.down_proj`` via ``target_parameters``, the
experts module becomes a ``ParamWrapper`` chain whose forward materializes the merged
weight ``base + delta`` through ``_activate_lora`` before the experts interface runs.
That defeats the fused low-rank LoRA path and fails outright on quantized bases
(``aten.add`` on NVFP4Tensor is unimplemented). This patches ``ParamWrapper.forward``
so that, when the wrapped base is a sonicmoe experts module, the LoRA A/B/scaling are
handed to the base via ``_sonicmoe_lora`` and the base forward is called directly. The
wrappers stay in the module tree, so PEFT save/load and optimizer tracking are unaffected.
"""

from __future__ import annotations


def _is_sonicmoe_experts(module) -> bool:
    cfg = getattr(module, "config", None)
    impl = getattr(cfg, "_experts_implementation", None)
    return impl == "sonicmoe" and hasattr(module, "gate_up_proj")


def patch_paramwrapper_sonicmoe_fastpath() -> None:
    """Idempotently patch ``peft...ParamWrapper.forward`` for sonicmoe experts."""
    try:
        from peft.tuners.lora.layer import ParamWrapper
    except (ImportError, AttributeError):
        return

    if getattr(ParamWrapper.forward, "_sonicmoe_fastpath", False):
        return

    from .lora import get_lora_params_from_wrapper

    _orig_forward = ParamWrapper.forward

    def _fusable(wrapper) -> bool:
        # Fused path needs exactly one adapter on a raw (un-merged) base; else defer to PEFT.
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
            _is_sonicmoe_experts(base) and all(_fusable(w) for w in wrappers.values())
        ):
            return _orig_forward(self, x, *args, **kwargs)

        lora = {}
        for name, wrapper in wrappers.items():
            lora_A, lora_B, scaling = get_lora_params_from_wrapper(wrapper)
            if lora_A is None:
                continue
            # PEFT keeps LoRA fp32; cast to activation dtype (grads still route to the fp32 params).
            lora[name] = (lora_A.to(x.dtype), lora_B.to(x.dtype), scaling)

        base._sonicmoe_lora = lora
        # Loading can create multiple ExpertsInterface registries; the experts forward binds one as
        # a closure default that may differ from the one register_sonicmoe_experts() populated, so
        # register into the one THIS module dispatches through.
        if not getattr(base, "_sonicmoe_iface_ok", False):
            _fn = type(base).forward
            _cells = dict(
                zip(_fn.__code__.co_freevars, _fn.__closure__ or (), strict=False)
            )
            _ei = _cells.get("experts_interface")
            if _ei is not None:
                _ei = _ei.cell_contents
                if "sonicmoe" not in _ei:
                    from .experts import sonicmoe_experts_forward_with_lora

                    _ei.register("sonicmoe", sonicmoe_experts_forward_with_lora)
            base._sonicmoe_iface_ok = True
        try:
            return base(x, *args, **kwargs)
        finally:
            base._sonicmoe_lora = None

    forward._sonicmoe_fastpath = True  # type: ignore[attr-defined]
    ParamWrapper.forward = forward
