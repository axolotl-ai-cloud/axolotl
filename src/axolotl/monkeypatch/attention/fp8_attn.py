"""FP8 low-precision attention via torchao.

Requires:
  - PyTorch >= 2.11.0
  - SM90+ (Hopper/Blackwell) GPU
  - flash-attn package with FA3 support
  - torchao >= 0.17.0

Uses per-head FP8 quantized attention with automatic RoPE fusion under torch.compile.
The torchao patch replaces F.scaled_dot_product_attention, so the model must use
HF's "sdpa" attention implementation for the patch to intercept attention calls.
"""

import logging

import torch

LOG = logging.getLogger(__name__)


def patch_fp8_attention(model: torch.nn.Module) -> torch.nn.Module:
    """Apply FP8 low-precision attention to a model.

    Must be called after model loading and before torch.compile.
    KV caching should be disabled (config.use_cache = False).
    """
    from torchao.prototype.attention import apply_low_precision_attention

    LOG.info("Applying FP8 low-precision attention (torchao)")
    return apply_low_precision_attention(model)
