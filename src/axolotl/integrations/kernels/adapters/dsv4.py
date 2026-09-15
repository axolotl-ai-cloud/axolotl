"""DeepSeek-V4 kernel adapter.

Owns everything DSV4-specific: the NVFP4/FP8 quantizer install, fused attention/RoPE/mHC
kernel patching, the post-PEFT fp32->compute dtype policy (keeping the mHC/keep_in_fp32 modules
in fp32), the fused clamped-SwiGLU shared-expert MLP LoRA patch, and the fp8 attention LoRA
patch. Also registers the mHC module class names that FSDP2's quantized-mixed-dtype path shards
in their own fp32 group.
"""

from __future__ import annotations

import torch

from axolotl.integrations.kernels.adapters import ModelAdapter
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

# mHC modules kept in fp32 and sharded separately by the FSDP2 quantized path.
DSV4_FP32_SHARD_CLASSES = ("DeepseekV4HyperConnection", "DeepseekV4HyperHead")


def _maybe_truncate_layers(model) -> None:
    """Debug-only (``DSV4_TRUNCATE_LAYERS=N``): truncate the decoder stack to the first N layers
    before PEFT so a small slice of the model fits a couple of GPUs for fast local bring-up /
    memory iteration. The first layers span all attention types (sliding/CSA/HCA). No-op unless the
    env var is set. Keeps HF's ``len(layer_types) == num_hidden_layers`` validators happy by
    truncating per-layer config lists (length ``orig`` -> n; ``orig+1`` e.g. compress_ratios -> n+1)."""
    import os

    trunc = os.environ.get("DSV4_TRUNCATE_LAYERS")
    if not trunc:
        return
    import gc

    import torch.nn as nn

    n = int(trunc)
    for mod in model.modules():
        layers = getattr(mod, "layers", None)
        if (
            isinstance(layers, nn.ModuleList)
            and len(layers) > n
            and "DecoderLayer" in type(layers[0]).__name__
        ):
            orig = len(layers)
            mod.layers = layers[:n]
            for cfg_obj in (model.config, getattr(mod, "config", None)):
                if cfg_obj is None:
                    continue
                if hasattr(cfg_obj, "num_hidden_layers"):
                    cfg_obj.num_hidden_layers = n
                for k, v in list(vars(cfg_obj).items()):
                    if isinstance(v, list) and len(v) in (orig, orig + 1):
                        setattr(cfg_obj, k, v[: n + (len(v) - orig)])
            gc.collect()
            torch.cuda.empty_cache()
            LOG.warning("DSV4_TRUNCATE_LAYERS=%d: truncated decoder stack (debug)", n)
            break


class DSV4Adapter(ModelAdapter):
    name = "deepseek_v4"

    def matches(self, cfg) -> bool:
        return bool(cfg.use_dsv4_kernels)

    def pre_model_load(self, cfg) -> None:
        # NVFP4 MoE checkpoints declare `quant_method: fp8` but transformers' finegrained-FP8
        # quantizer has no NVFP4 expert path. Install an NVFP4-aware subclass so experts load as
        # NVFP4Tensor for the scattermoe fused path. Only relevant with the scattermoe expert path.
        if cfg.use_scattermoe:
            from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_fp8_quantizer import (
                configure_nonexpert_mode,
                install_nvfp4_fp8_quantizer,
            )

            configure_nonexpert_mode(cfg.get("dsv4_fp8_nonexpert_mode"))
            install_nvfp4_fp8_quantizer()

        # Fused Triton training kernels for DSV4 (attention / RoPE / mHC).
        from axolotl.integrations.kernels.libs.dsv4 import patch_deepseek_v4_kernels

        patch_deepseek_v4_kernels()
        # Make the FSDP2 quantized path keep these mHC modules in their own fp32 shard group.
        from axolotl.monkeypatch.accelerate.fsdp2_quantized import (
            register_fp32_shard_classes,
        )

        register_fp32_shard_classes(DSV4_FP32_SHARD_CLASSES)

    def pre_lora_load(self, cfg, model) -> None:
        # MIXED_PRECISION checkpoints load NVFP4 experts as plain packed uint8 with dropped scales;
        # rebuild them as NVFP4Tensor so scattermoe dequantizes correctly.
        if not cfg.use_scattermoe:
            return
        _maybe_truncate_layers(model)
        has_packed_experts = any(
            isinstance(getattr(m, "gate_up_proj", None), torch.Tensor)
            and m.gate_up_proj.dtype == torch.uint8
            and m.gate_up_proj.ndim == 3
            for m in model.modules()
        )
        if has_packed_experts:
            from axolotl.integrations.kernels.libs.scattermoe_lora.nvfp4_moe_loading import (
                attach_nvfp4_expert_scales,
            )

            attach_nvfp4_expert_scales(model, cfg.base_model)

    def post_model_load(self, cfg, model) -> None:
        self._apply_fp32_dtype_policy(cfg, model)
        self._patch_shared_mlp_lora(cfg, model)
        self._patch_attn_fp8_lora(cfg, model)

    # --- dtype policy -------------------------------------------------------
    @staticmethod
    def _apply_fp32_dtype_policy(cfg, model) -> None:
        """Cast residual fp32 params (PEFT-upcast LoRA) to the compute dtype while keeping the
        model's ``_keep_in_fp32_modules[_strict]`` params (mHC/norms/sinks) in fp32. The fused
        DSV4 kernels demote fp32 activations at their input boundary, so this gives one consistent
        compute dtype. ``dsv4_bf16_all: true`` reverts to a blanket cast including keep_in_fp32."""
        dt = cfg.torch_dtype or torch.bfloat16
        keep_all = bool(cfg.get("dsv4_bf16_all"))
        keep_patterns: list[str] = []
        if not keep_all:
            seen: set[str] = set()
            for m in model.modules():
                for attr in ("_keep_in_fp32_modules_strict", "_keep_in_fp32_modules"):
                    for pat in getattr(m, attr, None) or ():
                        if pat not in seen:
                            seen.add(pat)
                            keep_patterns.append(pat)
        n = kept = 0
        for name, p in model.named_parameters():
            if p.dtype != torch.float32:
                continue
            if keep_patterns and any(pat in name for pat in keep_patterns):
                kept += 1
                continue
            p.data = p.data.to(dt)
            n += 1
        if n:
            LOG.info(
                "dsv4: cast %d residual fp32 params to %s for fused kernels (kept %d keep_in_fp32 fp32)",
                n,
                dt,
                kept,
            )

    # --- fused LoRA kernel swaps -------------------------------------------
    @staticmethod
    def _patch_shared_mlp_lora(cfg, model) -> None:
        """Swap DSV4 shared-expert MLPs for the fused clamped-SwiGLU LoRA kernel.

        Gated by ``dsv4_shared_mlp_lora_kernel`` (NOT the generic ``lora_mlp_kernel``, which the
        MoE-kernel validator force-disables). The validator translates a legacy
        ``lora_mlp_kernel: true`` on a DSV4 run into this flag (see KernelsArgs.disable_mlp_kernel).
        """
        if cfg.get("adapter") != "lora" or not cfg.get("dsv4_shared_mlp_lora_kernel"):
            return
        from axolotl.integrations.kernels.libs.dsv4.lora_mlp import (
            patch_dsv4_shared_mlp_lora,
        )

        n = patch_dsv4_shared_mlp_lora(model)
        LOG.info(
            "dsv4: patched %d shared-expert MLPs with fused clamped-SwiGLU LoRA", n
        )

    @staticmethod
    def _patch_attn_fp8_lora(cfg, model) -> None:
        """Native blockwise-fp8 fused LoRA for the large attention projections (q_b/o_b).
        No-op unless ``dsv4_fp8_lora_kernel``."""
        if cfg.get("adapter") != "lora":
            return
        from axolotl.integrations.kernels.libs.dsv4.lora_fp8 import (
            patch_dsv4_attn_fp8_lora,
        )

        patch_dsv4_attn_fp8_lora(model, enabled=bool(cfg.get("dsv4_fp8_lora_kernel")))
