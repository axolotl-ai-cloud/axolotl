"""Compatibility for DeepSpeed debug references during native tensor moves."""

from contextlib import contextmanager
from functools import wraps

import torch


@contextmanager
def _suspend_native_debug_refs(model, names):
    finalizers = getattr(names, "_finalizers", None)
    suspended = []
    try:
        if isinstance(finalizers, dict):
            for parameter in model.parameters():
                if type(parameter).__name__ != "NVFP4Tensor":
                    continue
                finalizer = finalizers.get(id(parameter))
                if finalizer is None or not finalizer.alive:
                    continue
                name = names[parameter]
                # Tensor swapping rejects even DeepSpeed's diagnostic weak references.
                finalizer.detach()
                finalizers.pop(id(parameter))
                suspended.append((parameter, name))
        yield
    finally:
        for parameter, name in suspended:
            names[parameter] = name


def install_native_nvfp4_debug_compat(engine_class):
    original = getattr(engine_class, "_configure_distributed_model", None)
    if original is None or getattr(original, "_axolotl_native_debug_compat", False):
        return

    @wraps(original)
    def configure_distributed_model(self, model, *args, **kwargs):
        if not getattr(model, "_axolotl_native_nvfp4_deepspeed_names", ()):
            return original(self, model, *args, **kwargs)
        from deepspeed.utils import debug

        with _suspend_native_debug_refs(model, debug.param_names):
            return original(self, model, *args, **kwargs)

    configure_distributed_model._axolotl_native_debug_compat = True
    engine_class._configure_distributed_model = configure_distributed_model


def install_native_nvfp4_zero3_dtype_compat(offload_class):
    original = getattr(offload_class, "_convert_to_zero_parameters", None)
    if original is None or getattr(
        original, "_axolotl_native_nvfp4_zero3_dtype_compat", False
    ):
        return

    @wraps(original)
    def convert_to_zero_parameters(self, ds_config, module, mpu):
        if (
            getattr(module, "_axolotl_native_nvfp4_zero3_components", ())
            and self.dtype == torch.uint8
        ):
            dtype = next(
                (
                    parameter.dtype
                    for parameter in module.parameters()
                    if parameter.is_floating_point()
                ),
                None,
            )
            if dtype is not None:
                self.dtype = dtype
        return original(self, ds_config, module, mpu)

    convert_to_zero_parameters._axolotl_native_nvfp4_zero3_dtype_compat = True
    offload_class._convert_to_zero_parameters = convert_to_zero_parameters
