"""Qwen3-MoE / Qwen3-Next NVFP4 kernel adapter.

A ``quant_method: modelopt`` / ``quant_algo: NVFP4`` Qwen3-MoE-family checkpoint (e.g.
nvidia/Qwen3-30B-A3B-NVFP4, nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4) is not a recognized
transformers quantizer, so the model loads as a bf16 skeleton; we then register
``WeightConverter``s so the NVFP4 tensors load correctly. Like the glm_moe_dsa adapter (this is
that adapter minus the DSA attention patching), the checkpoint's safetensors index is inspected
(:func:`inspect_nvfp4_layout`) to discover what is actually quantized: the per-expert
projections (fused into 3D ``NVFP4Tensor`` for the grouped MoE paths) and any non-routed NVFP4
linears (dequantized to bf16), and exactly those converters are registered for THIS checkpoint.
qwen3_next differs only outside the MoE block (Gated DeltaNet linear attention on 3/4 layers,
a shared expert, per-projection ``input_scale`` tensors, an ``mtp.*`` head the causal-LM class
never loads); the layout inspection handles all of that without model-specific code.
"""

from __future__ import annotations

from axolotl.integrations.kernels.adapters import modelopt_nvfp4_model_config
from axolotl.integrations.kernels.adapters.nvfp4_moe import Nvfp4MoeAdapter


def is_qwen3_moe_nvfp4_modelopt(cfg) -> bool:
    """True iff the base model is a ``qwen3_moe``/``qwen3_next`` NVFP4-modelopt checkpoint
    (``quant_method=modelopt``, ``quant_algo=NVFP4``). Any failure returns False."""
    model_config = modelopt_nvfp4_model_config(cfg)
    return model_config is not None and str(
        getattr(model_config, "model_type", "")
    ) in ("qwen3_moe", "qwen3_next")


class Qwen3MoeAdapter(Nvfp4MoeAdapter):
    name = "qwen3_moe"

    def matches(self, cfg) -> bool:
        return bool(
            cfg.use_scattermoe or cfg.use_sonicmoe
        ) and is_qwen3_moe_nvfp4_modelopt(cfg)
