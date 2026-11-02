"""Mistral Large 4 (``mistral_large4``) NVFP4 kernel adapter.

The checkpoint is modelopt ``MIXED_PRECISION`` in Mistral-native naming: NVFP4 routed experts as
``w1/w3/w2`` and 128x128 block-FP8 attention / indexer ``wq_b`` / shared experts / dense layer 0.
transformers renames native keys before any converter runs, and the expert converters live in
the text model's (scoped) mapping, so the converters register there under the renamed names.
"""

from __future__ import annotations

import re

from axolotl.integrations.kernels.adapters import modelopt_quant_model_config
from axolotl.integrations.kernels.adapters.nvfp4_moe import (
    SUPPORTED_ROUTED_PROJS,
    Nvfp4MoeAdapter,
)


def is_mistral_large4_nvfp4_modelopt(cfg) -> bool:
    """True iff the base model is a ``mistral_large4`` modelopt checkpoint. Never raises."""
    model_config = modelopt_quant_model_config(cfg, algos=("NVFP4", "MIXED_PRECISION"))
    return (
        model_config is not None
        and str(getattr(model_config, "model_type", "")) == "mistral_large4"
    )


def _renamed_suffix(suffix: str, renamings: list) -> str:
    key = f"layers.0.{suffix}.weight"
    for renaming in renamings:
        key, _ = renaming.rename_source_key(key)
    return re.sub(r"^.*?layers\.\d+\.", "", key)[: -len(".weight")]


class MistralLarge4Adapter(Nvfp4MoeAdapter):
    name = "mistral_large4"
    supported_routed_projs = (SUPPORTED_ROUTED_PROJS, frozenset({"w1", "w2", "w3"}))

    def matches(self, cfg) -> bool:
        return bool(
            cfg.use_scattermoe or cfg.use_sonicmoe
        ) and is_mistral_large4_nvfp4_modelopt(cfg)

    def _conversion_model_type(self, cfg) -> str:
        return "mistral_large4_text"

    def _registration_layout(self, cfg, layout: dict) -> dict:
        from transformers.conversion_mapping import get_checkpoint_conversion_mapping
        from transformers.core_model_loading import WeightRenaming

        # root mapping first, then the text scope: the order transformers applies them in
        renamings = [
            t
            for model_type in ("mistral_large4", "mistral_large4_text")
            for t in get_checkpoint_conversion_mapping(model_type) or []
            if isinstance(t, WeightRenaming)
        ]
        layout = dict(layout)
        for key in ("nonrouted_suffixes", "fp8_suffixes"):
            layout[key] = sorted(
                {_renamed_suffix(s, renamings) for s in layout.get(key, [])}
            )
        return layout
