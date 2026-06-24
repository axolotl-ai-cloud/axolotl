"""GLM-5.2 (``glm_moe_dsa``) NVFP4 kernel adapter.

GLM-5.2 is the DeepSeek-V3.2 sparse-MLA (DSA / Lightning-Indexer) lineage. A ``quant_method:
modelopt`` / ``quant_algo: NVFP4`` checkpoint is not a recognized transformers quantizer, so the
model loads as a bf16 skeleton; we then register ``WeightConverter``s so the NVFP4 tensors load
correctly. Rather than assume a fixed layout, the adapter inspects the checkpoint's safetensors
index (:func:`inspect_nvfp4_layout`) to discover what is actually quantized — the per-expert
projections (fused into 3D ``NVFP4Tensor`` for the scattermoe path) and any non-routed NVFP4
linears (dequantized to bf16) — and registers exactly those converters for THIS checkpoint.

The DSA attention / indexer / RoPE fused training kernels are a separate (Phase-1) concern and
are intentionally NOT owned here yet.
"""

from __future__ import annotations

from axolotl.integrations.kernels.adapters import ModelAdapter
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def is_glm_moe_dsa_nvfp4_modelopt(cfg) -> bool:
    """True iff the base model is a GLM-5.2 (``glm_moe_dsa``) NVFP4-modelopt checkpoint
    (``quant_method=modelopt``, ``quant_algo=NVFP4``). Any failure returns False."""
    try:
        from transformers import AutoConfig

        hf_cfg = AutoConfig.from_pretrained(cfg.base_model, trust_remote_code=True)
    except Exception:
        return False
    if str(getattr(hf_cfg, "model_type", "")) != "glm_moe_dsa":
        return False
    qcfg = getattr(hf_cfg, "quantization_config", None)
    if qcfg is None:
        return False
    if isinstance(qcfg, dict):
        return (
            qcfg.get("quant_method") == "modelopt" and qcfg.get("quant_algo") == "NVFP4"
        )
    return (
        getattr(qcfg, "quant_method", None) == "modelopt"
        and getattr(qcfg, "quant_algo", None) == "NVFP4"
    )


class GlmMoeDsaAdapter(ModelAdapter):
    name = "glm_moe_dsa_nvfp4"

    def matches(self, cfg) -> bool:
        return bool(cfg.use_scattermoe) and is_glm_moe_dsa_nvfp4_modelopt(cfg)

    def pre_model_load(self, cfg) -> None:
        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
            inspect_nvfp4_layout,
        )
        from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_weight_converter import (
            register_nvfp4_converters_for_layout,
        )

        layout = inspect_nvfp4_layout(cfg.base_model)
        LOG.info(
            "glm_moe_dsa: detected NVFP4 layout — routed experts: %s (projs=%s); "
            "non-routed NVFP4 linears: %s",
            layout["routed_present"],
            layout["routed_projs"],
            layout["nonrouted_suffixes"],
        )
        register_nvfp4_converters_for_layout("glm_moe_dsa", layout)
