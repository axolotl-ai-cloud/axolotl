"""Loading-time quantization for MoE expert weights stored as 3D nn.Parameter tensors."""

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
    # Module path → param names in definition order, captured before quantization.
    # Without this, alphabetical loading order would mismatch merge order.
    "expert_param_order": {},
}


class Bnb8bitParametrization(torch.nn.Module):
    """Dequantizes int8 row-wise quantized data on access."""

    def __init__(self, row_stats: torch.Tensor):
        super().__init__()
        self.register_buffer("row_stats", row_stats)

    @torch.no_grad()
    def forward(self, quantized_param: torch.Tensor) -> torch.Tensor:
        """Flatten 3D+ to 2D for BnB's dequant, then reshape back."""
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
    """Patch transformers' weight loading to quantize MoE expert params on-the-fly."""
    mode = "8bit" if getattr(cfg, "load_in_8bit", False) else "4bit"
    _moe_load_state["mode"] = mode
    _moe_load_state["count"] = 0
    _moe_load_state["expert_param_order"] = {}

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

                # Record definition order before parametrizations override it
                # with alphabetical order.
                if mod_path not in _moe_load_state["expert_param_order"]:
                    _moe_load_state["expert_param_order"][mod_path] = list(
                        mod._parameters.keys()
                    )

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
    """Fix PEFT's _inject_parameters for target_parameters on quantized MoE experts.

    1. Expands short suffixes to full module paths for parametrized modules.
    2. Iterates params in definition order (not alphabetical order) so saved
       adapters are compatible with standard PEFT, vLLM, etc.
    3. Skips ParametrizationList synthetic paths to prevent PEFT from mistakenly
       targeting quantized expert params via name-suffix matching.
    """
    if getattr(patch_peft_target_parameters_matching, "_axolotl_patched", False):
        return

    from contextlib import nullcontext

    from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer
    from peft.utils.integrations import init_empty_weights
    from peft.utils.other import _get_submodules

    # Mapping from unfused parameter names to their fused equivalents.
    # When a model stores fused weights (e.g. gate_up_proj) but the user
    # specifies unfused names (gate_proj, up_proj), we auto-expand so the
    # fused parameter is also targeted. The original unfused names are kept
    # in the set so that models that do NOT fuse still work.
    _UNFUSED_TO_FUSED: dict[str, str] = {
        "gate_proj": "gate_up_proj",
        "up_proj": "gate_up_proj",
    }

    def _patched_inject_parameters(
        self, peft_config, model, adapter_name, low_cpu_mem_usage
    ):
        original_targets = list(peft_config.target_parameters)
        expanded = set(original_targets)

        # Expand short suffixes to full paths for parametrized modules.
        for module_name, module in model.named_modules():
            if not hasattr(module, "parametrizations"):
                continue
            for target in original_targets:
                mod_path, _, param_name = target.rpartition(".")
                if not (
                    module_name == mod_path or module_name.endswith("." + mod_path)
                ):
                    continue

                if hasattr(module, param_name):
                    expanded.add(f"{module_name}.{param_name}")
                elif param_name in _UNFUSED_TO_FUSED:
                    # The model uses fused weights (e.g. gate_up_proj) but the
                    # user specified unfused names (gate_proj / up_proj).
                    fused_name = _UNFUSED_TO_FUSED[param_name]
                    if hasattr(module, fused_name):
                        if fused_name not in expanded:
                            LOG.warning(
                                "target_parameter '%s' not found on %s, "
                                "but fused equivalent '%s' exists — adding "
                                "it automatically.",
                                param_name,
                                module_name,
                                fused_name,
                            )
                        expanded.add(f"{module_name}.{fused_name}")
                    else:
                        LOG.warning(
                            "target_parameter '%s' not found on %s and no "
                            "fused equivalent exists either — skipping.",
                            param_name,
                            module_name,
                        )
                else:
                    LOG.warning(
                        "target_parameter '%s' not found on %s — skipping. "
                        "Check that the parameter name matches the model's "
                        "weight names.",
                        param_name,
                        module_name,
                    )

        target_names_set = expanded

        def strip_base_layer_from_name(module_name):
            name = ".base_layer"
            while name in module_name:
                prefix, _, suffix = module_name.rpartition(name)
                module_name = prefix + suffix
            return module_name

        def create_and_replace_param(module_name, key, param_name):
            parent, target, target_name = _get_submodules(model, module_name)
            unwrapped_module_name = strip_base_layer_from_name(module_name)
            unwrapped_module = model.get_submodule(unwrapped_module_name)
            if (
                isinstance(unwrapped_module, BaseTunerLayer)
                and unwrapped_module.__class__.__name__ != "ParamWrapper"
            ):
                raise ValueError(
                    f"Trying to wrap an `nn.Parameter` of layer "
                    f"'{unwrapped_module_name}' of type "
                    f"{type(target).__name__}, which is not a valid target. "
                    f"Make sure that this layer is not also targeted with "
                    f"`target_modules`."
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

        # Use definition order (not alphabetical order) for parametrized modules
        # so ParamWrapper nesting matches vanilla PEFT on a plain model.
        expert_param_order = _moe_load_state.get("expert_param_order", {})

        for module_name, module in model.named_modules():
            if hasattr(module, "parametrizations"):
                stored_order = expert_param_order.get(module_name)
                if stored_order is not None:
                    params_iter = [
                        p for p in stored_order if p in module.parametrizations
                    ]
                else:
                    # Fallback for paths that bypass model loading (e.g. unit tests).
                    params_iter = list(module.parametrizations.keys())
                for param_name in params_iter:
                    key = f"{module_name}.{param_name}"
                    if (key in target_names_set) or any(
                        key.endswith(f".{t}") for t in target_names_set
                    ):
                        create_and_replace_param(module_name, key, param_name)
                        self.targeted_parameter_names.append(key)
            else:
                unwrapped_module_name = strip_base_layer_from_name(module_name)
                for param_name, _ in module.named_parameters(recurse=False):
                    key = f"{unwrapped_module_name}.{param_name}"
                    if (key in target_names_set) or any(
                        key.endswith(f".{t}") for t in target_names_set
                    ):
                        create_and_replace_param(module_name, key, param_name)
                        self.targeted_parameter_names.append(key)

    BaseTuner._inject_parameters = _patched_inject_parameters

    # Skip ParametrizationList synthetic paths (e.g. "...parametrizations.up_proj")
    # so PEFT suffix-matching doesn't try to wrap quantized expert params in LoRA.
    # Previous MoE models (Mixtral, DeepSeek, etc.) stored experts as nn.Linear
    # modules, so PEFT's normal target_modules path worked fine. NemotronH uses
    # 3D nn.Parameter tensors via our quantize_moe_experts parametrization, which
    # exposes synthetic ".parametrizations.<name>" paths that PEFT's suffix match
    # would otherwise treat as target_modules candidates.
    _original_check = BaseTuner._check_target_module_exists

    @staticmethod
    def _patched_check_target_module_exists(config, key):
        if ".parametrizations." in key:
            return False
        return _original_check(config, key)

    BaseTuner._check_target_module_exists = _patched_check_target_module_exists

    patch_peft_target_parameters_matching._axolotl_patched = True
    LOG.info("Patched PEFT _inject_parameters for consistent ParamWrapper ordering")
