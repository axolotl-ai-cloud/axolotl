"""Mistral Large 4 (``mistral_large4``) kernel adapter.

Runs on transformers' built-in modeling (``MistralLarge4ForConditionalGeneration``, loaded as
image-text-to-text; sdpa / eager only), never remote code. Its MLA is *clean* — a single
``q_proj`` (``q_lora_rank=null``) and no gradientless fused indexer — so the attention
projections are safe to LoRA-adapt.

NVFP4 checkpoint: modelopt ``MIXED_PRECISION`` in Mistral-native naming: NVFP4 routed experts as
``w1/w3/w2`` and 128x128 block-FP8 attention / indexer ``wq_b`` / shared experts / dense layer 0.
transformers renames native keys before any converter runs, and the expert converters live in
the text model's (scoped) mapping, so the converters register there under the renamed names.

Block-FP8 (FineGrainedFP8) checkpoint: transformers loads ``FP8Linear`` / ``FP8Experts`` and their
backward already dequantizes for dX. ``moe_grouped_mode`` picks the scattermoe expert path
(``fp8_block`` keeps the experts native; ``nvfp4`` / ``mxfp8`` requant them at the plugin level).
Genuine block-FP8 experts are never reinterpreted as NVFP4.

For a LoRA run on a block-FP8 or bf16 checkpoint the adapter also swaps the MLA attention LoRA
layers for the fused quant-dispatching kernel (``libs/mistral_large4``) and keeps the RMSNorms in
their own fp32 FSDP2 shard group. Both stay off on the NVFP4 checkpoint unless
``mistral_large4_attn_lora_kernel: true``.
"""

from __future__ import annotations

import re

from axolotl.integrations.kernels.adapters import (
    load_base_model_config,
    modelopt_quant_model_config,
)
from axolotl.integrations.kernels.adapters.nvfp4_moe import (
    SUPPORTED_ROUTED_PROJS,
    Nvfp4MoeAdapter,
)
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

MISTRAL_LARGE4_FP32_SHARD_CLASSES = ("MistralLarge4RMSNorm",)


def is_mistral_large4_nvfp4_modelopt(cfg) -> bool:
    """True iff the base model is a ``mistral_large4`` modelopt checkpoint. Never raises."""
    model_config = modelopt_quant_model_config(cfg, algos=("NVFP4", "MIXED_PRECISION"))
    return (
        model_config is not None
        and str(getattr(model_config, "model_type", "")) == "mistral_large4"
    )


def _has_fp8_linear(model) -> bool:
    try:
        from transformers.integrations.finegrained_fp8 import FP8Linear
    except Exception:
        return False
    return any(isinstance(m, FP8Linear) for m in model.modules())


def _renamed_suffix(suffix: str, renamings: list) -> str:
    key = f"layers.0.{suffix}.weight"
    for renaming in renamings:
        key, _ = renaming.rename_source_key(key)
    return re.sub(r"^.*?layers\.\d+\.", "", key)[: -len(".weight")]


class MistralLarge4Adapter(Nvfp4MoeAdapter):
    name = "mistral_large4"
    supported_routed_projs = (SUPPORTED_ROUTED_PROJS, frozenset({"w1", "w2", "w3"}))

    @staticmethod
    def _loads_nvfp4(cfg) -> bool:
        return bool(
            cfg.use_scattermoe or cfg.use_sonicmoe
        ) and is_mistral_large4_nvfp4_modelopt(cfg)

    def matches(self, cfg) -> bool:
        # mistral_large4_text has no loadable model class, so only the top-level type can load
        model_config = load_base_model_config(cfg)
        return str(getattr(model_config, "model_type", "")) == "mistral_large4"

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

    def pre_model_load(self, cfg) -> None:
        model_config = load_base_model_config(cfg)
        if cfg.get("trust_remote_code") and getattr(model_config, "auto_map", None):
            raise ValueError(
                f"{cfg.base_model} ships remote modeling code (auto_map), but the Mistral Large 4 "
                "kernels target transformers' built-in modeling. Drop trust_remote_code."
            )
        if self._loads_nvfp4(cfg):
            super().pre_model_load(cfg)
        if not is_mistral_large4_nvfp4_modelopt(cfg):
            from axolotl.monkeypatch.accelerate.fsdp2_quantized import (
                register_fp32_shard_classes,
            )

            register_fp32_shard_classes(MISTRAL_LARGE4_FP32_SHARD_CLASSES)

    def pre_lora_load(self, cfg, model) -> None:
        # NOTE: grouped-expert requant (mxfp8 / nvfp4) is model-agnostic and runs at the plugin level
        # (requant_grouped_experts) BEFORE this hook, so the fp8-linear patch below sees only the
        # non-expert linears. moe_grouped_mode='fp8_block' keeps the block-fp8 experts native.
        #
        # A FineGrainedFP8 base can't backprop through transformers' fp8 GEMM (no autograd). If any
        # fp8 (non-expert) linear remains, install the dequant-autograd forward so LoRA works.
        if _has_fp8_linear(model):
            from axolotl.integrations.kernels.libs.mistral_large4.fp8_training import (
                patch_fp8_linear_training,
            )

            patch_fp8_linear_training()

    def post_model_load(self, cfg, model) -> None:
        super().post_model_load(cfg, model)
        self._patch_attn_lora(cfg, model)

    def _attn_lora_kernel_enabled(self, cfg) -> bool:
        if cfg.get("adapter") != "lora":
            return False
        flag = cfg.get("mistral_large4_attn_lora_kernel")
        if flag is not None:
            return bool(flag)
        return not is_mistral_large4_nvfp4_modelopt(cfg)

    def _patch_attn_lora(self, cfg, model) -> None:
        """Fused quant-dispatching LoRA for the MLA attention projections."""
        if not self._attn_lora_kernel_enabled(cfg):
            return
        from axolotl.integrations.kernels.libs.mistral_large4 import (
            patch_mistral_large4_attn_lora,
        )
        from axolotl.integrations.kernels.libs.mistral_large4.lora_attn import (
            MLA_PROJ_SUFFIXES,
        )

        # Optional: quantize the frozen attention bases to a low-bit format (kernel then runs the
        # 4-/8-bit fused path). Must run post-PEFT-wrap (here), before the kernel patch.
        attn_quant = cfg.get("mistral_large4_attn_quantization")
        if attn_quant and str(attn_quant).lower() != "none":
            from axolotl.integrations.kernels.libs.mistral_large4.lora_lowbit import (
                quantize_attn_bases,
            )

            quantize_attn_bases(model, str(attn_quant).lower(), MLA_PROJ_SUFFIXES)

        native_fp8 = cfg.get("mistral_large4_attn_native_fp8") is not False
        patch_mistral_large4_attn_lora(model, enabled=True, native_fp8=native_fp8)
