"""
Loading-time quantization for MoE expert weights stored as 3D nn.Parameter tensors.

In transformers v5, many MoE models store expert weights as fused 3D nn.Parameter
tensors instead of individual nn.Linear modules. BnB 4-bit quantization only targets
nn.Linear, so these expert weights are loaded in full precision, causing high peak VRAM.

This module patches transformers' weight loading to quantize 3D expert parameters
on-the-fly as they're assigned to modules, using bitsandbytes.nn.parametrize.
replace_parameter_4bit (requires bitsandbytes >= 0.48.0). This reduces peak VRAM
from "all experts in bf16" to "one expert param in bf16 at a time."

PEFT's target_parameters / ParamWrapper can then apply LoRA on top of these quantized
params via stacked parametrizations.

Note: FSDP2 cpu ram efficient loading and Tensor Parallel (DTensor) compatibility
with parametrization is untested.
"""

import bitsandbytes as bnb
import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# Module-level state for the loading-time quantization patch.
_moe_load_state = {
    "count": 0,
    "quant_type": "nf4",
    "compress_statistics": True,
    "patched": False,
}


def patch_moe_quantization_on_load(cfg):
    """Patch transformers' weight loading to quantize 3D MoE expert params on-the-fly.

    Wraps ``transformers.core_model_loading.set_param_for_module`` so that after each
    parameter is assigned to its module, any 3D+ tensor on CUDA that BnB skipped
    (i.e. not inside a Linear4bit/Linear8bitLt) is immediately quantized via
    ``replace_parameter_4bit``. This keeps peak VRAM to one expert param in bf16
    at a time, instead of loading all experts in bf16 first.

    The patch stays active permanently — the ``ndim >= 3`` and ``is_cuda`` checks
    make it safe for non-MoE models (no false positives).

    Args:
        cfg: Axolotl DictDefault config. Reads bnb_4bit_quant_type and
             bnb_4bit_use_double_quant for quantization settings.
    """
    if _moe_load_state["patched"]:
        LOG.debug("MoE loading-time quantization patch already active")
        return

    import os

    import transformers.core_model_loading
    from bitsandbytes.nn.parametrize import replace_parameter_4bit

    # Disable transformers' async weight loading thread pool. Without this,
    # the ThreadPoolExecutor pre-fetches tensors to CUDA faster than the main
    # loop can quantize them, causing all expert weights to accumulate in bf16
    # on GPU — defeating the purpose of loading-time quantization.
    os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1"
    LOG.info("Disabled async weight loading (HF_DEACTIVATE_ASYNC_LOAD=1)")

    # Read quantization settings from config
    quant_type = getattr(cfg, "bnb_4bit_quant_type", None) or "nf4"
    compress_statistics = getattr(cfg, "bnb_4bit_use_double_quant", None)
    if compress_statistics is None:
        compress_statistics = True

    _moe_load_state["quant_type"] = quant_type
    _moe_load_state["compress_statistics"] = compress_statistics
    _moe_load_state["count"] = 0

    original_set_param = transformers.core_model_loading.set_param_for_module

    _first_call = [True]

    def _patched_set_param_for_module(model, target_name, param_value, *args, **kwargs):
        if _first_call[0]:
            LOG.info("MoE quant patch: set_param_for_module intercepted (first call)")
            _first_call[0] = False

        original_set_param(model, target_name, param_value, *args, **kwargs)

        # Quantize 3D+ expert params that BnB skipped (only on CUDA).
        if param_value.ndim >= 3:
            LOG.info(
                "MoE quant patch: 3D param %s shape=%s cuda=%s",
                target_name,
                param_value.shape,
                param_value.is_cuda,
            )
            if param_value.is_cuda:
                mod_path, _, pname = target_name.rpartition(".")
                mod = model.get_submodule(mod_path) if mod_path else model
                if not isinstance(mod, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
                    replace_parameter_4bit(
                        mod,
                        pname,
                        compress_statistics=_moe_load_state["compress_statistics"],
                        quant_type=_moe_load_state["quant_type"],
                    )
                    torch.cuda.empty_cache()
                    _moe_load_state["count"] += 1
                    LOG.info(
                        "Quantized 3D expert param: %s "
                        "(alloc=%.2f GiB, reserved=%.2f GiB)",
                        target_name,
                        torch.cuda.memory_allocated() / 1024**3,
                        torch.cuda.memory_reserved() / 1024**3,
                    )

    transformers.core_model_loading.set_param_for_module = _patched_set_param_for_module
    _moe_load_state["patched"] = True
    LOG.info(
        "Activated MoE loading-time quantization patch "
        "(quant_type=%s, compress_statistics=%s)",
        quant_type,
        compress_statistics,
    )


def get_moe_quantized_count():
    """Return the number of expert parameters quantized during loading."""
    return _moe_load_state["count"]


def patch_peft_target_parameters_matching():
    """Fix PEFT's _inject_parameters to use suffix matching for parametrized modules.

    PEFT's parametrized-module branch uses exact name match for target_parameters,
    but the standard branch uses endswith. This means suffix-style paths like
    "mlp.experts.gate_up_proj" fail to match parametrized modules whose full path
    is "model.layers.0.mlp.experts.gate_up_proj". This patch makes the parametrized
    branch consistent with the standard branch.
    """
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
    LOG.info("Patched PEFT _inject_parameters for parametrized module suffix matching")
