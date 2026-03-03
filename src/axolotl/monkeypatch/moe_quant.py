"""
Loading-time quantization for MoE expert weights stored as 3D nn.Parameter tensors.

In transformers v5, MoE models store expert weights as fused 3D tensors that BnB
skips (only targets nn.Linear). This module patches weight loading to quantize them
on-the-fly (4-bit via bitsandbytes parametrize, 8-bit via custom int8 parametrization),
reducing peak VRAM from "all experts in bf16" to "one expert at a time."
"""

import bitsandbytes as bnb
import torch
import torch.nn.utils.parametrize as P

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Module-level state for the loading-time quantization patch.
_moe_load_state = {
    "count": 0,
    "mode": "4bit",
    "quant_type": "nf4",
    "compress_statistics": True,
    "patched": False,
}


class Bnb8bitParametrization(torch.nn.Module):
    """Parametrization that dequantizes int8 row-wise quantized data on access."""

    def __init__(self, row_stats: torch.Tensor):
        super().__init__()
        self.register_buffer("row_stats", row_stats)

    @torch.no_grad()
    def forward(self, quantized_param: torch.Tensor) -> torch.Tensor:
        # Flatten 3D+ to 2D for BnB's dequant, then reshape back.
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

    # Cache dequantized values during forward to avoid redundant dequantization.
    if not getattr(module, "_axolotl_8bit_hooks_registered", False):
        module.register_forward_pre_hook(_enable_parametrization_cache)
        module.register_forward_hook(_disable_parametrization_cache)
        module._axolotl_8bit_hooks_registered = True


def patch_moe_quantization_on_load(cfg):
    """Patch transformers' weight loading to quantize MoE expert params on-the-fly.

    Wraps ``set_param_for_module`` so that 3D+ CUDA tensors with "expert" in their
    name are quantized (4-bit or 8-bit) as they're loaded, keeping peak VRAM low.
    """
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

    # Disable caching_allocator_warmup — it pre-allocates a huge tensor at bf16
    # size for all params, defeating our on-load quantization VRAM savings.
    def _noop_warmup(*args, **kwargs):
        pass

    transformers.modeling_utils.caching_allocator_warmup = _noop_warmup

    original_set_param = transformers.core_model_loading.set_param_for_module

    def _patched_set_param_for_module(model, target_name, param_value, *args, **kwargs):
        original_set_param(model, target_name, param_value, *args, **kwargs)

        # Quantize 3D+ expert params that BnB skipped (only on CUDA).
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

                # Release the bf16 tensor so CUDA memory is freed immediately.
                param_value.data = torch.empty(0, device="cpu")
                torch.cuda.empty_cache()

    transformers.core_model_loading.set_param_for_module = _patched_set_param_for_module
    _moe_load_state["patched"] = True


def get_moe_quantized_count():
    """Return the number of expert parameters quantized during loading."""
    return _moe_load_state["count"]


def patch_peft_target_parameters_matching():
    """Fix PEFT's _inject_parameters to use suffix matching for parametrized modules."""
    if getattr(patch_peft_target_parameters_matching, "_axolotl_patched", False):
        return
    from peft.tuners.tuners_utils import BaseTuner

    original_inject = BaseTuner._inject_parameters

    def _patched_inject_parameters(
        self, peft_config, model, adapter_name, low_cpu_mem_usage
    ):
        # Patch target_parameters to use full paths for parametrized modules
        original_targets = list(peft_config.target_parameters)
        expanded = set(original_targets)

        for module_name, module in model.named_modules():
            if not hasattr(module, "parametrizations"):
                continue
            for target in original_targets:
                mod_path, _, param_name = target.rpartition(".")
                if (
                    module_name == mod_path or module_name.endswith("." + mod_path)
                ) and hasattr(module, param_name):
                    expanded.add(f"{module_name}.{param_name}")

        peft_config.target_parameters = sorted(expanded)
        try:
            return original_inject(
                self, peft_config, model, adapter_name, low_cpu_mem_usage
            )
        finally:
            peft_config.target_parameters = original_targets

    BaseTuner._inject_parameters = _patched_inject_parameters
    patch_peft_target_parameters_matching._axolotl_patched = True
    LOG.info("Patched PEFT _inject_parameters for parametrized module suffix matching")
