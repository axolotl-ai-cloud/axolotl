"""Mistral Large 4 (``mistral_large4``) kernel adapter.

Runs on transformers' built-in modeling (``MistralLarge4ForConditionalGeneration``, loaded as
image-text-to-text; sdpa / eager only), never remote code. Its MLA has a single ``q_proj``
(``q_lora_rank=null``), and its gradientless DSA indexer reads the hidden states rather than the
MLA projections, so all four attention projections are safe to LoRA-adapt.

NVFP4 checkpoint: modelopt ``MIXED_PRECISION`` in Mistral-native naming: NVFP4 routed experts as
``w1/w3/w2`` and 128x128 block-FP8 attention / indexer ``wq_b`` / shared experts / dense layer 0.
transformers renames native keys before any converter runs, and the expert converters live in
the text model's (scoped) mapping, so the converters register there under the renamed names.

Block-FP8 (FineGrainedFP8) checkpoint: transformers loads ``FP8Linear`` / ``FP8Experts`` and their
backward already dequantizes for dX. Under ``use_scattermoe`` the generic ``moe_grouped_mode: auto``
puts the block-FP8 experts on the fp8_block grouped path (NVFP4 experts on nvfp4), no requant.
Genuine block-FP8 experts are never reinterpreted as NVFP4.

For a LoRA run on a block-FP8 or bf16 checkpoint the adapter also swaps the MLA attention LoRA
layers for the fused quant-dispatching kernel (``libs/mistral_large4``), off on the NVFP4
checkpoint unless ``mistral_large4_attn_lora_kernel: true``. Off the NVFP4 checkpoint it also
registers the RMSNorms as fp32 FSDP2 shard classes, which only take effect once
``mistral_large4_attn_quantization`` puts torchao low-bit weights in the model.
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
        if cfg.get("adapter") in ("lora", "qlora") and cfg.get("lora_target_linear"):
            LOG.warning(
                "lora_target_linear: true also puts LoRA on the DSA indexer projections "
                "(self_attn.indexer.wq_b / wk / weights_proj), which run under no_grad, so those "
                "adapters never train. List the targets with lora_target_modules instead."
            )
        if self._attn_quantization(cfg):
            if not self._attn_lora_kernel_enabled(cfg):
                raise ValueError(
                    "mistral_large4_attn_quantization quantizes the bases of the fused attention "
                    "LoRA, which is off for this config: it needs adapter: lora, and defaults off "
                    "on the modelopt NVFP4 checkpoint and under an "
                    "activation_checkpointing.selective that saves matmul outputs. Set mistral_large4_attn_lora_kernel: true or drop "
                    "mistral_large4_attn_quantization."
                )
            for key, reason in (
                ("peft_use_dora", "the low-bit attention LoRA has no DoRA path"),
                (
                    "load_in_8bit",
                    "bitsandbytes attention bases cannot be requantized to a low-bit format",
                ),
                (
                    "load_in_4bit",
                    "bitsandbytes attention bases cannot be requantized to a low-bit format",
                ),
                (
                    "lora_merge_aware",
                    "the merge-aware forward cannot merge LoRA into a low-bit base",
                ),
            ):
                if cfg.get(key):
                    raise ValueError(
                        f"mistral_large4_attn_quantization does not support {key}: {reason}. "
                        "Drop one of them."
                    )
        conflict = self._selective_checkpointing_conflict(cfg)
        if (
            conflict
            and cfg.get("adapter") == "lora"
            and cfg.get("mistral_large4_attn_lora_kernel")
        ):
            raise ValueError(
                f"mistral_large4_attn_lora_kernel: true cannot be combined with {conflict}: "
                "the fused attention LoRA adds the adapter delta into the saved base matmul "
                "output in place. Set mistral_large4_attn_lora_kernel: false or save only "
                "attention."
            )
        if self._loads_nvfp4(cfg):
            super().pre_model_load(cfg)
        if not is_mistral_large4_nvfp4_modelopt(cfg):
            from axolotl.monkeypatch.accelerate.fsdp2_quantized import (
                register_fp32_shard_classes,
            )

            register_fp32_shard_classes(MISTRAL_LARGE4_FP32_SHARD_CLASSES)

    def post_model_load(self, cfg, model) -> None:
        super().post_model_load(cfg, model)
        self._patch_attn_lora(cfg, model)

    def _attn_lora_kernel_enabled(self, cfg) -> bool:
        if cfg.get("adapter") != "lora":
            return False
        flag = cfg.get("mistral_large4_attn_lora_kernel")
        if flag is not None:
            return bool(flag)
        return (
            not is_mistral_large4_nvfp4_modelopt(cfg)
            and self._selective_checkpointing_conflict(cfg) is None
        )

    @staticmethod
    def _selective_checkpointing_conflict(cfg) -> str | None:
        """What in ``activation_checkpointing.selective`` clashes with the fused attention LoRA,
        which adds the adapter delta into the base matmul output in place (the cached tensor
        SAC would save), or None."""
        checkpointing = cfg.get("activation_checkpointing")
        sac = (
            checkpointing.get("selective") if isinstance(checkpointing, dict) else None
        )
        if not isinstance(sac, dict):
            return None
        from axolotl.integrations.kernels.libs.mistral_large4.lora_attn import (
            MLA_PROJ_SUFFIXES,
        )
        from axolotl.utils.schemas.validation import (
            _sac_entry_covers,
            _sac_entry_is_layer_container,
        )

        matmul_ops = ("aten::mm", "aten::addmm", "aten::matmul", "aten::bmm")
        for spec in sac.get("save") or []:
            if spec != "attention" and any(spec in op for op in matmul_ops):
                return (
                    f"activation_checkpointing.selective.save entry {spec!r} saves "
                    "matmul outputs"
                )
        if sac.get("save_matmul_min_k"):
            return (
                "activation_checkpointing.selective.save_matmul_min_k saves matmul "
                "outputs"
            )
        covered = {suffix.rsplit(".", 1)[-1] for suffix in MLA_PROJ_SUFFIXES}
        covered.add("self_attn")
        for entry in sac.get("save_modules") or []:
            if _sac_entry_is_layer_container(entry) or any(
                _sac_entry_covers(entry, module) for module in covered
            ):
                return (
                    f"activation_checkpointing.selective.save_modules entry {entry!r} "
                    "saves attention matmul outputs"
                )
        return None

    @staticmethod
    def _attn_quantization(cfg) -> str | None:
        quant = cfg.get("mistral_large4_attn_quantization")
        return str(quant).lower() if quant and str(quant).lower() != "none" else None

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
        attn_quant = self._attn_quantization(cfg)
        if attn_quant:
            from axolotl.integrations.kernels.libs.mistral_large4.lora_lowbit import (
                quantize_attn_bases,
            )

            quantize_attn_bases(model, attn_quant, MLA_PROJ_SUFFIXES)

        native_fp8 = cfg.get("mistral_large4_attn_native_fp8") is not False
        patch_mistral_large4_attn_lora(model, enabled=True, native_fp8=native_fp8)
