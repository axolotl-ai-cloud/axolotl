"""Gemma-4 NVFP4 kernel adapter.

Owns Gemma-4 specifics: detecting the NVFP4-modelopt checkpoint, registering the native NVFP4
expert WeightConverters, and applying the non-expert quantization policy (fp8 / nf4). The hybrid
global-attention mask and the large-head flash kernel are generic capabilities wired in the
loader/patch_manager and are intentionally NOT owned here.
"""

from __future__ import annotations

from axolotl.integrations.kernels.adapters import (
    ModelAdapter,
    modelopt_nvfp4_model_config,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def is_gemma4_nvfp4_modelopt(cfg) -> bool:
    """True iff the base model is a Gemma-4 NVFP4-modelopt checkpoint (quant_method=modelopt,
    quant_algo=NVFP4, model_type startswith gemma4). Any failure returns False."""
    model_config = modelopt_nvfp4_model_config(cfg)
    return model_config is not None and str(
        getattr(model_config, "model_type", "")
    ).startswith("gemma4")


def resolve_nonexpert_quantization(cfg) -> str | None:
    """Resolve the non-expert quantization policy to one of {None, 'fp8', 'nf4', 'nvfp4'}.

    Prefers the intent field ``nonexpert_quantization`` (none/bf16/fp8_blockwise/nf4/nvfp4); falls
    back to the legacy gemma4-specific flags. Returns None for no quantization (bf16 non-experts).
    'nvfp4' routes non-experts through the Marlin W4A16 kernel (same path as the experts); 'nf4'
    uses bitsandbytes.
    """
    policy = cfg.get("nonexpert_quantization")
    if policy:
        p = str(policy).lower()
        if p in ("none", "bf16"):
            return None
        if p in ("fp8", "fp8_blockwise"):
            return "fp8"
        if p == "nf4":
            return "nf4"
        if p == "nvfp4":
            return "nvfp4"
    if cfg.get("gemma4_fp8_nonexpert"):
        return "fp8"
    if cfg.get("gemma4_nf4_nonexpert"):
        return "nf4"
    return None


class Gemma4Adapter(ModelAdapter):
    name = "gemma4_nvfp4"

    def matches(self, cfg) -> bool:
        return bool(cfg.use_scattermoe) and is_gemma4_nvfp4_modelopt(cfg)

    def consumes_nonexpert_quantization(self, cfg) -> bool:
        return True

    def pre_model_load(self, cfg) -> None:
        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_weight_converter import (
            register_gemma4_nvfp4_converters,
        )

        register_gemma4_nvfp4_converters()
        LOG.info(
            "gemma4: registered NVFP4 expert WeightConverters (modelopt checkpoint)"
        )

    def pre_lora_load(self, cfg, model) -> None:
        policy = resolve_nonexpert_quantization(cfg)
        if policy == "fp8":
            from axolotl.integrations.kernels.libs.scattermoe_lora.gemma4_fp8_nonexpert import (
                quantize_gemma4_nonexpert_linears,
            )

            n = quantize_gemma4_nonexpert_linears(model)
            LOG.info(
                "gemma4: fp8-quantized %d non-expert linears (experts stay NVFP4)", n
            )
        elif policy == "nf4":
            from axolotl.integrations.kernels.libs.scattermoe_lora.gemma4_nf4_nonexpert import (
                quantize_gemma4_nonexpert_nf4,
            )

            n = quantize_gemma4_nonexpert_nf4(model)
            LOG.info(
                "gemma4: nf4-quantized %d non-expert linears (experts stay NVFP4)", n
            )
        elif policy == "nvfp4":
            from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_nonexpert import (
                quantize_gemma4_nonexpert_nvfp4,
            )

            n = quantize_gemma4_nonexpert_nvfp4(model)
            LOG.info(
                "gemma4: nvfp4-quantized %d non-expert linears via Marlin W4A16 "
                "(experts also NVFP4)",
                n,
            )
