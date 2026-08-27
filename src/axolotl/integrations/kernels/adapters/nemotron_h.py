"""Nemotron-3 (``nemotron_h``) NVFP4 kernel adapter.

Nemotron-3 NVFP4 checkpoints (e.g. nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) ship
``quant_method: modelopt`` with ``quant_algo: MIXED_PRECISION``: every routed expert
projection (per-expert non-gated ``up_proj``/``down_proj``) is NVFP4 group-16, while the
shared path — ``fc1/fc2_latent_proj``, shared experts, attention, mamba — is static FP8.
transformers recognizes neither, so the model loads as a bf16 skeleton and the shared
:class:`Nvfp4MoeAdapter` loader registers the converters this checkpoint's headers call
for: per-expert NVFP4 ``up/down`` fused into the 3D ``experts.up_proj``/``experts.down_proj``
(no gate to concatenate), and each FP8 linear dequantized to bf16. The nemotron_h built-in
``backbone.`` → ``model.`` renames are preserved by the registration.
"""

from __future__ import annotations

from axolotl.integrations.kernels.adapters import modelopt_quant_model_config
from axolotl.integrations.kernels.adapters.nvfp4_moe import Nvfp4MoeAdapter


def is_nemotron_h_nvfp4_modelopt(cfg) -> bool:
    """True iff the base model is a ``nemotron_h`` modelopt checkpoint with NVFP4 experts
    (``quant_algo`` NVFP4 or MIXED_PRECISION — Nemotron-3 ships the latter: FP8 shared path,
    NVFP4 routed experts). Any failure returns False."""
    model_config = modelopt_quant_model_config(cfg, algos=("NVFP4", "MIXED_PRECISION"))
    return (
        model_config is not None
        and str(getattr(model_config, "model_type", "")) == "nemotron_h"
    )


class NemotronHAdapter(Nvfp4MoeAdapter):
    name = "nemotron_h"

    def matches(self, cfg) -> bool:
        return bool(
            cfg.use_scattermoe or cfg.use_sonicmoe
        ) and is_nemotron_h_nvfp4_modelopt(cfg)
