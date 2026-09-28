"""Nemotron-3 (``nemotron_h``) NVFP4 kernel adapter.

Nemotron-3 ships modelopt ``MIXED_PRECISION``: NVFP4 group-16 routed experts (non-gated
``up``/``down``, no gate to concatenate), static-FP8 shared path. transformers recognizes
neither, so the model loads as a bf16 skeleton and :class:`Nvfp4MoeAdapter` registers the
converters that fill it.
"""

from __future__ import annotations

from axolotl.integrations.kernels.adapters import modelopt_quant_model_config
from axolotl.integrations.kernels.adapters.nvfp4_moe import Nvfp4MoeAdapter


def is_nemotron_h_nvfp4_modelopt(cfg) -> bool:
    """True iff the base model is a ``nemotron_h`` modelopt checkpoint; Nemotron-3 ships
    ``MIXED_PRECISION`` (FP8 shared path, NVFP4 routed experts). Never raises."""
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
