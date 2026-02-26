"""
Loading-time quantization for MoE expert weights stored as 3D nn.Parameter tensors.

In transformers v5, many MoE models store expert weights as fused 3D nn.Parameter
tensors instead of individual nn.Linear modules. BnB quantization only targets
nn.Linear, so these expert weights are loaded in full precision, causing high peak VRAM.

This module patches transformers' weight loading to quantize 3D expert parameters
on-the-fly as they're assigned to modules. For 4-bit, it uses
bitsandbytes.nn.parametrize.replace_parameter_4bit (requires bitsandbytes >= 0.48.0).
For 8-bit, it uses a custom parametrization built on bitsandbytes.functional's
int8_vectorwise_quant/dequant (row-wise absmax scaling). Both reduce peak VRAM from
"all experts in bf16" to "one expert param in bf16 at a time."

PEFT's target_parameters / ParamWrapper can then apply LoRA on top of these quantized
params via stacked parametrizations.
"""

import gc

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
    """Parametrization that dequantizes int8 row-wise quantized data on access.

    Mirrors ``Bnb4bitParametrization`` from ``bitsandbytes.nn.parametrize`` but for
    int8 row-wise (absmax) quantization.  Stores the per-row scales as a buffer and
    delegates to ``bitsandbytes.functional.int8_vectorwise_dequant`` which computes
    ``int8_data * row_stats * (1/127)``.
    """

    def __init__(self, row_stats: torch.Tensor):
        super().__init__()
        self.register_buffer("row_stats", row_stats)

    @torch.no_grad()
    def forward(self, quantized_param: torch.Tensor) -> torch.Tensor:
        return bnb.functional.int8_vectorwise_dequant(quantized_param, self.row_stats)


def _enable_parametrization_cache(module, inputs):
    P._cache_enabled += 1


def _disable_parametrization_cache(module, inputs, output):
    P._cache_enabled -= 1
    if not P._cache_enabled:
        P._cache = {}


def replace_parameter_8bit(module, param_name):
    """Replace a module parameter with an 8-bit quantized version using parametrization.

    Mirrors ``bitsandbytes.nn.parametrize.replace_parameter_4bit`` but for int8
    row-wise (absmax) quantization.  Uses ``int8_vectorwise_quant`` which supports
    N-D tensors natively (scales shape = ``prod(shape[:-1])``).

    Args:
        module: The module containing the parameter to quantize.
        param_name: Name of the parameter within the module.
    """
    original_param = getattr(module, param_name)
    int8_data, row_stats, _ = bnb.functional.int8_vectorwise_quant(
        original_param.data.to(torch.float16)
    )

    setattr(module, param_name, torch.nn.Parameter(int8_data, requires_grad=False))
    del original_param

    P.register_parametrization(
        module, param_name, Bnb8bitParametrization(row_stats), unsafe=True
    )

    # Register caching hooks (same pattern as BnB 4-bit). Caching avoids
    # redundant dequantization when the same param is accessed multiple times
    # in a single forward pass.
    module.register_forward_pre_hook(_enable_parametrization_cache)
    module.register_forward_hook(_disable_parametrization_cache)


def _8bit_state_dict_post_hook(
    module, state_dict, prefix, local_metadata, *, param_name
):
    """Placeholder for 8-bit state_dict serialization hook.

    For LoRA/QLoRA training, only adapter weights (lora_A/B) are saved — base model
    weights including quantized experts are not serialized. State dict hooks for 8-bit
    are therefore not needed for the primary use case. If full-model saving with 8-bit
    quantized expert params is needed, this hook should store int8 data + row_stats.
    """
    raise NotImplementedError(
        "State dict serialization for 8-bit quantized expert parameters is not yet "
        "implemented. This is not needed for LoRA/QLoRA training (only adapter weights "
        "are saved). If you need to save the full model with 8-bit quantized experts, "
        "please open an issue."
    )


def patch_moe_quantization_on_load(cfg):
    """Patch transformers' weight loading to quantize 3D MoE expert params on-the-fly.

    Wraps ``transformers.core_model_loading.set_param_for_module`` so that after each
    parameter is assigned to its module, any 3D+ tensor on CUDA that BnB skipped
    (i.e. not inside a Linear4bit/Linear8bitLt) is immediately quantized via
    ``replace_parameter_4bit`` (for 4-bit) or ``replace_parameter_8bit`` (for 8-bit).
    This keeps peak VRAM to one expert param in bf16 at a time, instead of loading
    all experts in bf16 first.

    The patch stays active permanently — the ``ndim >= 3`` and ``is_cuda`` checks
    make it safe for non-MoE models (no false positives).

    Args:
        cfg: Axolotl DictDefault config. For 4-bit, reads bnb_4bit_quant_type and
             bnb_4bit_use_double_quant. For 8-bit, no additional settings needed.
    """
    if _moe_load_state["patched"]:
        LOG.debug("MoE loading-time quantization patch already active")
        return

    import transformers.core_model_loading
    import transformers.modeling_utils

    # Determine quantization mode from config.
    if getattr(cfg, "load_in_8bit", False):
        mode = "8bit"
    else:
        mode = "4bit"

    _moe_load_state["mode"] = mode
    _moe_load_state["count"] = 0

    if mode == "4bit":
        from bitsandbytes.nn.parametrize import replace_parameter_4bit

        quant_type = getattr(cfg, "bnb_4bit_quant_type", None) or "nf4"
        compress_statistics = getattr(cfg, "bnb_4bit_use_double_quant", None)
        if compress_statistics is None:
            compress_statistics = True

        _moe_load_state["quant_type"] = quant_type
        _moe_load_state["compress_statistics"] = compress_statistics

    # Patch caching_allocator_warmup to be a no-op. This function pre-allocates
    # a single huge GPU tensor equal to the model's total param bytes to warm the
    # CUDA caching allocator. For MoE models, it calculates expert params at bf16
    # size (BnB doesn't know we'll quantize them), causing a ~50+ GiB reservation
    # that defeats loading-time quantization. Disabling it trades slightly slower
    # weight loading for dramatically lower peak VRAM.
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

                # Release the bf16 CUDA storage. After quantization, the
                # module holds the quantized parametrization — but the
                # caller's references (loop var + dict) keep the bf16 CUDA
                # storage alive past empty_cache(). Replacing .data frees
                # the CUDA memory immediately regardless of Python refcount.
                param_value.data = torch.empty(0, device="cpu")
                gc.collect()
                torch.cuda.empty_cache()

    transformers.core_model_loading.set_param_for_module = _patched_set_param_for_module
    _moe_load_state["patched"] = True


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
