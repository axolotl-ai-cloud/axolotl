"""Compatibility shim: tie DiffusionGemma's encoder experts to quantized decoder experts.

DiffusionGemma ties the encoder text layers (including the MoE experts) to the
decoder layers. When ``quantize_moe_experts`` replaces a decoder expert weight with
a bitsandbytes 4-bit *parametrized* parameter during loading, transformers'
``tie_weights()`` can no longer resolve the tied source: ``get_parameter_or_buffer``
only checks plain parameters/buffers, not ``module.parametrizations``. The result is
``AttributeError: ...experts.gate_up_proj is neither a parameter, buffer, nor extra
state`` at load time.

This patch makes ``get_parameter_or_buffer`` fall back to the parametrization's
underlying tensor, so the tied encoder experts share the quantized decoder weights.
"""

from __future__ import annotations

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_tie_weights_for_quantized_experts():
    """Idempotently patch ``PreTrainedModel.get_parameter_or_buffer``."""
    from transformers.modeling_utils import PreTrainedModel

    if getattr(PreTrainedModel.get_parameter_or_buffer, "_diffusion_gemma_patched", False):
        return

    original = PreTrainedModel.get_parameter_or_buffer

    def _patched(self, target: str):
        try:
            return original(self, target)
        except AttributeError:
            mod_path, _, param_name = target.rpartition(".")
            try:
                module = self.get_submodule(mod_path) if mod_path else self
            except AttributeError:
                raise
            parametrizations = getattr(module, "parametrizations", None)
            if parametrizations is not None and param_name in parametrizations:
                # The underlying (quantized) tensor lives at `.original`.
                return parametrizations[param_name].original
            raise

    _patched._diffusion_gemma_patched = True
    PreTrainedModel.get_parameter_or_buffer = _patched
    LOG.info("DiffusionGemma: patched get_parameter_or_buffer for tied quantized experts")


def _to_fp4(weight, fmt: str):
    """Quantize a 3D ``[E, d1, d2]`` expert weight to a frozen torchao FP4 tensor."""
    import torch

    if fmt == "nvfp4":
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        return NVFP4Tensor.to_nvfp4(weight.to(torch.bfloat16).contiguous(), block_size=16)
    if fmt == "mxfp4":
        from torchao.prototype.mx_formats.mx_tensor import MXTensor

        return MXTensor.to_mx(
            weight.to(torch.bfloat16).contiguous(),
            elem_dtype=torch.float4_e2m1fn_x2,
            block_size=32,
        )
    raise ValueError(f"frozen_fp4_experts must be 'nvfp4' or 'mxfp4', got {fmt!r}")


def quantize_experts_to_fp4(model, fmt: str) -> int:
    """Quantize DiffusionGemma's fused MoE experts to frozen torchao FP4 in place.

    ScatterMoE consumes torchao MXFP4/NVFP4 expert tensors directly (selective
    dequant in the kernel), so this realizes a genuinely 4-bit-frozen expert base
    without a bitsandbytes round-trip. The encoder experts are tied to the decoder
    experts, so each quantized parameter is shared back to the matching encoder layer.
    """
    import gc

    import torch
    from torch import nn

    decoder_layers = model.model.decoder.layers
    encoder_layers = model.model.encoder.language_model.layers
    count = 0
    for i, dec_layer in enumerate(decoder_layers):
        experts = dec_layer.experts
        enc_experts = encoder_layers[i].experts
        for name in ("gate_up_proj", "down_proj"):
            weight = getattr(experts, name)
            quant = nn.Parameter(_to_fp4(weight.data, fmt), requires_grad=False)
            if name in experts._parameters:
                del experts._parameters[name]
            setattr(experts, name, quant)
            # Re-tie the encoder expert (same object) to the quantized parameter.
            if name in enc_experts._parameters:
                del enc_experts._parameters[name]
            setattr(enc_experts, name, quant)
            # Drop the bf16 source so its GPU memory is released immediately.
            del weight
            count += 1
        gc.collect()
        torch.cuda.empty_cache()
    LOG.info(f"DiffusionGemma: quantized {count} expert tensors to frozen {fmt}")
    return count
