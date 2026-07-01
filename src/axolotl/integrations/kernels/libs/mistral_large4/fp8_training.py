"""Make a FineGrainedFP8 base trainable through PEFT.

``transformers``' ``FP8Linear.forward`` dispatches to ``fp8_linear`` (a fused Triton/DeepGEMM
blockwise-fp8 GEMM) which has **no autograd formula** — fine for inference, but any fp8 linear on
the autograd path (a LoRA base, the dense layer-0 MLP, any untargeted projection) then raises on
``.backward()``. LoRA training freezes the fp8 base but still needs ``dX`` to flow through it.

This patches ``FP8Linear.forward`` to an autograd-friendly path: dequantize the frozen fp8 weight
to the activation dtype (transient, one weight at a time) and run a plain ``F.linear`` so autograd
computes ``dX`` (the weight is frozen, so no weight grad is produced). The targeted attention
projections still use the faster native-fp8 fused-LoRA kernel (they bypass ``FP8Linear.forward``);
this only rescues the linears that would otherwise break the backward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_PATCHED_ATTR = "_ml4_fp8_training_forward"


def _dequant_forward(self, x: torch.Tensor) -> torch.Tensor:
    # High-precision weights already loaded (dequantize=True) -> plain path.
    if self.weight.element_size() > 1:
        return F.linear(x, self.weight, self.bias)
    from transformers.integrations.finegrained_fp8 import to_local

    from axolotl.kernels.quantize import dequantize_fp8

    w = dequantize_fp8(to_local(self.weight), to_local(self.weight_scale_inv), x.dtype)
    return F.linear(x, w, self.bias)


def patch_fp8_linear_training() -> bool:
    """Swap ``FP8Linear.forward`` for the dequant-autograd forward (idempotent).

    Returns True if the patch is now in place. Class-level, so it covers every FP8Linear instance
    (they all need a trainable backward). No-op if transformers has no FP8Linear.
    """
    try:
        from transformers.integrations import finegrained_fp8 as fp8mod
    except Exception:
        return False
    cls = getattr(fp8mod, "FP8Linear", None)
    if cls is None:
        return False
    if getattr(cls, _PATCHED_ATTR, False):
        return True
    cls.forward = _dequant_forward
    setattr(cls, _PATCHED_ATTR, True)
    LOG.info(
        "mistral_large4: patched FP8Linear.forward for trainable backward (dequant-autograd)"
    )
    return True
