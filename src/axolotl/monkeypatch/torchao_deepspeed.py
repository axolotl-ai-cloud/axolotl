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
