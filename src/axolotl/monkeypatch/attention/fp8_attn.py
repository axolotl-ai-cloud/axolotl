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

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def patch_fp8_attention(model: torch.nn.Module) -> torch.nn.Module:
    """Apply FP8 low-precision attention to a model.

    Must be called after model loading and before torch.compile.
    KV caching should be disabled (config.use_cache = False).

    FP8 attention with a training backward pass is Hopper-only here: torchao's
    only backend is FP8_FA3 (SM 9.x + flash_attn FA3), and Transformer Engine's
    cuDNN fused/flash attention hard-disables FP8 on sm_120 (consumer Blackwell).
    Fail fast with that context rather than surface torchao's opaque
    "No compatible backend for SMxx".
    """
    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        if major == 12:
            raise RuntimeError(
                "attn_implementation: fp8 is unavailable on consumer Blackwell "
                "(sm_120): no FP8 attention kernel with a backward pass runs here "
                "(torchao FP8_FA3 is Hopper-only; TE disables FP8 fused/flash "
                "attention on sm_120). Use flash_attention_2 (bf16 attention) — "
                "it pairs with NVFP4 FP4-GEMM training and is the fastest "
                "varlen-packing attention on sm_120."
            )

    from torchao.prototype.attention import apply_low_precision_attention

    LOG.info("Applying FP8 low-precision attention (torchao)")
    return apply_low_precision_attention(model)
