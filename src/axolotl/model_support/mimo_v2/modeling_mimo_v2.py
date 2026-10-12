"""MiMo-V2 text decoder on transformers' native MiMo-V2-Flash modules.

The published remote code routes experts through an inference-only ``noaux_tc``
gate and an ``nn.ModuleList`` of experts; the native modules train and expose
fused experts to the MoE kernels. Vision, audio and MTP weights are not loaded.
"""

import torch
from transformers import core_model_loading
from transformers.core_model_loading import Concatenate, ConversionOps, _IdentityOp
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
from transformers.models.mimo_v2_flash.modeling_mimo_v2_flash import (
    MiMoV2FlashForCausalLM,
    MiMoV2FlashModel,
)

from .configuration_mimo_v2 import MiMoV2Config

_TEXT_ONLY_UNEXPECTED = [
    r"^model\.mtp\.",
    r"^visual\.",
    r"^audio_encoder\.",
    r"^speech_embeddings\.",
]


class MiMoV2Model(MiMoV2FlashModel):
    config_class = MiMoV2Config
    _keys_to_ignore_on_load_unexpected = _TEXT_ONLY_UNEXPECTED


class MiMoV2ForCausalLM(MiMoV2FlashForCausalLM):
    config_class = MiMoV2Config
    _keys_to_ignore_on_load_unexpected = _TEXT_ONLY_UNEXPECTED

    def save_pretrained(self, *args, save_original_format=False, **kwargs):
        # dequantization cannot be reversed; the bf16 native layout reloads as-is
        return super().save_pretrained(
            *args, save_original_format=save_original_format, **kwargs
        )


class MiMoV2Dequantize(Fp8Dequantize):
    """FP8 128x128-block weights and uint8-packed MXFP4 experts to the param dtype."""

    def __init__(self):
        super().__init__(hf_quantizer=None)

    def _dequantize_one(self, quantized, scales, output_dtype=None):
        # without a quantizer the loader casts to the model dtype first; the packed
        # bytes, e8m0 exponents and FP8 values survive that cast exactly
        rows, cols = quantized.shape
        if scales.shape[-1] * 16 == cols:
            quantized = quantized.to(torch.uint8).view(torch.int8)
            scales = scales.to(torch.uint8)
        else:
            # FP8 scale grids are padded to a multiple of 4 rows
            scales = scales[: rows // 128, : cols // 128]
        return super()._dequantize_one(quantized, scales, output_dtype)

    @property
    def reverse_op(self) -> ConversionOps:
        return _IdentityOp()


class SplitQKV(ConversionOps):
    """Split a fused ``qkv_proj`` whose k/v widths differ between full and SWA layers."""

    @torch.no_grad
    def convert(self, input_dict, source_patterns, target_patterns, **kwargs):
        (qkv,) = input_dict.values()
        qkv = qkv[0] if isinstance(qkv, list) else qkv
        config = kwargs["config"]
        q = config.num_attention_heads * config.head_dim
        kv = qkv.shape[0] - q
        k = kv * config.head_dim // (config.head_dim + config.v_head_dim)
        return dict(zip(target_patterns, qkv.split([q, k, kv - k]), strict=True))

    @property
    def reverse_op(self) -> ConversionOps:
        return Concatenate(dim=0)


# the many-to-many guard only admits allow-listed ops; dequant + split is one
if SplitQKV not in core_model_loading._INTERNAL_MANY_TO_MANY_CONVERSIONS:
    core_model_loading._INTERNAL_MANY_TO_MANY_CONVERSIONS += (SplitQKV,)
