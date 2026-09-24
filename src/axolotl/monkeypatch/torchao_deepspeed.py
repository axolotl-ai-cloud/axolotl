"""DeepSpeed preparation for frozen native TorchAO NVFP4 parameters."""

from __future__ import annotations

from axolotl.monkeypatch.torchao_ddp import prepare_native_nvfp4_components


def _install_native_nvfp4_broadcast_filter(engine_class=None) -> None:
    if engine_class is None:
        from deepspeed.runtime.engine import DeepSpeedEngine

        engine_class = DeepSpeedEngine
    DeepSpeedEngine = engine_class

    if getattr(DeepSpeedEngine, "_axolotl_native_nvfp4_broadcast_filter", False):
        return
    original_broadcast_model = DeepSpeedEngine._broadcast_model

    def broadcast_model(self):
        names = getattr(self.module, "_axolotl_native_nvfp4_deepspeed_names", ())
        if not names:
            return original_broadcast_model(self)
        names = frozenset(names)
        native = {
            name: parameter
            for name, parameter in self.module.named_parameters()
            if name in names
        }
        if set(native) != names or any(
            type(parameter).__name__ != "NVFP4Tensor" or parameter.requires_grad
            for parameter in native.values()
        ):
            raise ValueError("Native NVFP4 DeepSpeed parameters must remain frozen")
        sentinel = object()
        previous = self.module.__dict__.get("named_parameters", sentinel)
        original_named_parameters = self.module.named_parameters

        def filtered_named_parameters(*args, **kwargs):
            return (
                (name, parameter)
                for name, parameter in original_named_parameters(*args, **kwargs)
                if name not in names
            )

        self.module.named_parameters = filtered_named_parameters
        try:
            return original_broadcast_model(self)
        finally:
            if previous is sentinel:
                del self.module.named_parameters
            else:
                self.module.named_parameters = previous

    if hasattr(DeepSpeedEngine, "module_state_dict"):
        original_module_state = DeepSpeedEngine.module_state_dict

        def module_state_dict(
            self,
            destination=None,
            prefix="",
            keep_vars=False,
            exclude_frozen_parameters=False,
        ):
            state = original_module_state(
                self,
                destination=destination,
                prefix=prefix,
                keep_vars=keep_vars,
                exclude_frozen_parameters=exclude_frozen_parameters,
            )
            if exclude_frozen_parameters and getattr(
                self.module, "_axolotl_native_nvfp4_deepspeed_names", ()
            ):
                for name, parameter in self.module.named_parameters(
                    remove_duplicate=False
                ):
                    if not parameter.requires_grad:
                        state.pop(prefix + name, None)
            return state

        DeepSpeedEngine.module_state_dict = module_state_dict

    if hasattr(DeepSpeedEngine, "load_module_state_dict"):
        original_load = DeepSpeedEngine.load_module_state_dict

        def load_module_state_dict(
            self, checkpoint, strict=True, custom_load_fn=None, fetch_z3_params=False
        ):
            native_names = getattr(
                self.module, "_axolotl_native_nvfp4_deepspeed_names", ()
            )
            if native_names:
                canonical = dict(self.module.named_parameters())
                native_ids = {id(canonical[name]) for name in native_names}
                native_aliases = {
                    name
                    for name, parameter in self.module.named_parameters(
                        remove_duplicate=False
                    )
                    if id(parameter) in native_ids
                }
                if native_aliases & checkpoint["module"].keys():
                    raise ValueError(
                        "Adapter checkpoint contains frozen native parameters"
                    )
                if fetch_z3_params:
                    raise ValueError(
                        "Native NVFP4 adapter restore does not support ZeRO-3"
                    )
                validate_native_nvfp4_adapter_state(self.module, checkpoint["module"])
                if custom_load_fn is None:
                    custom_load_fn = lambda src, dst: load_native_nvfp4_adapter_state(
                        dst, src
                    )
            return original_load(
                self,
                checkpoint,
                strict=strict,
                custom_load_fn=custom_load_fn,
                fetch_z3_params=fetch_z3_params,
            )

        DeepSpeedEngine.load_module_state_dict = load_module_state_dict
    DeepSpeedEngine._broadcast_model = broadcast_model
    DeepSpeedEngine._axolotl_native_nvfp4_broadcast_filter = True


def prepare_native_nvfp4_deepspeed(model, device, zero_stage: int) -> bool:
    """Prepare frozen native NVFP4 parameters before a ZeRO 1/2 engine is built."""
    if zero_stage not in (1, 2):
        return False
    names = prepare_native_nvfp4_components(model, device)
    if not names:
        return False
    model._axolotl_native_nvfp4_deepspeed_names = frozenset(names)
    _install_native_nvfp4_broadcast_filter()
    return True


def _required_adapter_state(model):
    parameters = dict(model.named_parameters(remove_duplicate=False))
    model_keys = set(model.state_dict())
    required = model_keys - parameters.keys()
    required.update(
        name for name, parameter in parameters.items() if parameter.requires_grad
    )
    return required, model_keys


def validate_native_nvfp4_adapter_state(model, state_dict) -> set[str]:
    """Reject missing trainables before loading an adapter checkpoint."""
    expected, model_keys = _required_adapter_state(model)
    saved = set(state_dict)
    missing = expected - saved
    unexpected = saved - model_keys
    if missing or unexpected:
        raise ValueError(
            "Native NVFP4 DeepSpeed adapter checkpoint keys do not match trainable parameters: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return expected


def load_native_nvfp4_adapter_state(model, state_dict) -> None:
    """Restore trainables and buffers while allowing a separately loaded frozen base."""
    expected = validate_native_nvfp4_adapter_state(model, state_dict)
    result = model.load_state_dict(state_dict, strict=False)
    missing_trainable = set(result.missing_keys) & expected
    if missing_trainable or result.unexpected_keys:
        raise ValueError(
            "Native NVFP4 DeepSpeed adapter checkpoint did not restore strictly"
        )
