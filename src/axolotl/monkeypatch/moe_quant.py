"""
Loading-time quantization for MoE expert weights stored as 3D nn.Parameter tensors.

In transformers v5, MoE models store expert weights as fused 3D tensors that BnB
skips (only targets nn.Linear). This module patches weight loading to quantize them
on-the-fly (4-bit via bitsandbytes parametrize, 8-bit via custom int8 parametrization),
reducing peak VRAM from "all experts in bf16" to "one expert at a time."
"""

import contextlib
import functools
import os

import bitsandbytes as bnb
import torch
import torch.nn.utils.parametrize as P

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_moe_load_state = {
    "count": 0,
    "mode": "4bit",
    "quant_type": "nf4",
    "compress_statistics": True,
    "patched": False,
}


class Bnb8bitParametrization(torch.nn.Module):
    """Dequantizes int8 row-wise quantized data on access."""

    def __init__(self, row_stats: torch.Tensor):
        super().__init__()
        self.register_buffer("row_stats", row_stats)

    @torch.no_grad()
    def forward(self, quantized_param: torch.Tensor) -> torch.Tensor:
        orig_shape = quantized_param.shape
        if quantized_param.ndim > 2:
            quantized_param = quantized_param.reshape(-1, orig_shape[-1])
        result = bnb.functional.int8_vectorwise_dequant(quantized_param, self.row_stats)
        return result.reshape(orig_shape)


def _enable_parametrization_cache(module, inputs):
    P._cache_enabled += 1


def _disable_parametrization_cache(module, inputs, output):
    P._cache_enabled -= 1
    if not P._cache_enabled:
        P._cache = {}


def replace_parameter_8bit(module, param_name):
    """Replace a module parameter with an 8-bit quantized version using parametrization."""
    original_param = getattr(module, param_name)
    int8_data, row_stats, _ = bnb.functional.int8_vectorwise_quant(
        original_param.data.to(torch.float16)
    )

    setattr(module, param_name, torch.nn.Parameter(int8_data, requires_grad=False))
    del original_param

    P.register_parametrization(
        module, param_name, Bnb8bitParametrization(row_stats), unsafe=True
    )

    if not getattr(module, "_axolotl_8bit_hooks_registered", False):
        module.register_forward_pre_hook(_enable_parametrization_cache)
        module.register_forward_hook(_disable_parametrization_cache)
        module._axolotl_8bit_hooks_registered = True


def patch_moe_quantization_on_load(cfg):
    """Patch transformers' weight loading to quantize MoE expert params on-the-fly."""
    mode = "8bit" if getattr(cfg, "load_in_8bit", False) else "4bit"
    _moe_load_state["mode"] = mode
    _moe_load_state["count"] = 0

    if _moe_load_state["patched"]:
        LOG.debug("MoE loading-time quantization patch already active")
        return

    import transformers.core_model_loading
    import transformers.modeling_utils

    if mode == "4bit":
        from bitsandbytes.nn.parametrize import replace_parameter_4bit

        quant_type = getattr(cfg, "bnb_4bit_quant_type", None) or "nf4"
        compress_statistics = getattr(cfg, "bnb_4bit_use_double_quant", None)
        if compress_statistics is None:
            compress_statistics = True

        _moe_load_state["quant_type"] = quant_type
        _moe_load_state["compress_statistics"] = compress_statistics

    # Force sequential tensor loading so we can quantize-and-free one expert at a time.
    # Without this, transformers pre-fetches all bf16 expert tensors to GPU simultaneously.
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"

    transformers.modeling_utils.caching_allocator_warmup = lambda *_: None

    original_set_param = transformers.core_model_loading.set_param_for_module

    def _patched_set_param_for_module(model, target_name, param_value, *args, **kwargs):
        original_set_param(model, target_name, param_value, *args, **kwargs)

        if param_value.ndim >= 3 and param_value.is_cuda:
            mod_path, _, pname = target_name.rpartition(".")
            mod = model.get_submodule(mod_path) if mod_path else model
            if not isinstance(mod, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                if "expert" not in target_name.lower():
                    LOG.debug(
                        "Skipping non-expert 3D param: %s (shape=%s)",
                        target_name,
                        list(param_value.shape),
                    )
                    return

                if _moe_load_state["mode"] == "4bit":
                    replace_parameter_4bit(
                        mod,
                        pname,
                        compress_statistics=_moe_load_state["compress_statistics"],
                        quant_type=_moe_load_state["quant_type"],
                    )
                else:
                    replace_parameter_8bit(mod, pname)
                _moe_load_state["count"] += 1

                param_value.data = torch.empty(0, device="cpu")
                torch.cuda.empty_cache()

    transformers.core_model_loading.set_param_for_module = _patched_set_param_for_module
    _moe_load_state["patched"] = True


def get_moe_quantized_count():
    """Return the number of expert parameters quantized during loading."""
    return _moe_load_state["count"]


def patch_peft_target_parameters_matching():
    """Fix PEFT's _inject_parameters for suffix matching and portable adapter ordering.

    1. Expands short suffix targets (e.g. "mlp.experts.gate_up_proj") to full module
       paths so the parametrized branch can match them.

    2. Makes the parametrized branch iterate module.parametrizations in insertion order
       instead of PEFT's sorted(target_names), matching the standard branch. This ensures
       adapters saved during training load correctly with vanilla PEFT, vLLM, and other
       tools without requiring this patch.
    """
    if getattr(patch_peft_target_parameters_matching, "_axolotl_patched", False):
        return

    from contextlib import nullcontext

    from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
    from peft.utils.integrations import init_empty_weights
    from peft.utils.other import _get_submodules

    def _patched_inject_parameters(
        self, peft_config, model, adapter_name, low_cpu_mem_usage
    ):
        original_targets = list(peft_config.target_parameters)
        target_names_set = set(original_targets)

        for module_name, module in model.named_modules():
            if not hasattr(module, "parametrizations"):
                continue
            for target in original_targets:
                mod_path, _, param_name = target.rpartition(".")
                if (
                    module_name == mod_path or module_name.endswith("." + mod_path)
                ) and hasattr(module, param_name):
                    target_names_set.add(f"{module_name}.{param_name}")

        def strip_base_layer_from_name(mod_name):
            name = ".base_layer"
            while name in mod_name:
                prefix, _, suffix = mod_name.rpartition(name)
                mod_name = prefix + suffix
            return mod_name

        def create_and_replace_param(mod_name, key, param_name):
            parent, target, target_name = _get_submodules(model, mod_name)
            unwrapped_name = strip_base_layer_from_name(mod_name)
            unwrapped = model.get_submodule(unwrapped_name)
            if (
                isinstance(unwrapped, BaseTunerLayer)
                and unwrapped.__class__.__name__ != "ParamWrapper"
            ):
                raise ValueError(
                    f"Trying to wrap an `nn.Parameter` of layer '{unwrapped_name}' of type "
                    f"{type(target).__name__}, which is not a valid target. Make sure that "
                    "this layer is not also targeted with `target_modules`. For some models, "
                    "PEFT will do this automatically, try setting `target_modules=[]`."
                )
            self._check_target_module_compatiblity(peft_config, model, target_name)
            ctx = init_empty_weights if low_cpu_mem_usage else nullcontext
            with ctx():
                self._create_and_replace(
                    peft_config,
                    adapter_name,
                    target,
                    target_name,
                    parent,
                    current_key=key,
                    parameter_name=param_name.rpartition(".")[-1],
                )

        def _matches(key):
            return (key in target_names_set) or any(
                key.endswith(f".{t}") for t in target_names_set
            )

        for module_name, module in model.named_modules():
            if hasattr(module, "parametrizations"):
                for param_name in module.parametrizations:
                    key = f"{module_name}.{param_name}"
                    if _matches(key):
                        create_and_replace_param(module_name, key, param_name)
                        self.targeted_parameter_names.append(key)
            else:
                unwrapped_name = strip_base_layer_from_name(module_name)
                for param_name, _ in module.named_parameters(recurse=False):
                    key = f"{unwrapped_name}.{param_name}"
                    if _matches(key):
                        create_and_replace_param(module_name, key, param_name)
                        self.targeted_parameter_names.append(key)

    BaseTuner._inject_parameters = _patched_inject_parameters
    patch_peft_target_parameters_matching._axolotl_patched = True
    LOG.info("Patched PEFT _inject_parameters for parametrized module suffix matching")
