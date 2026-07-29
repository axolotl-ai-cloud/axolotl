# Why: Qwen3VLVisionPatchEmbed's Conv3d has kernel_size == stride and no
# padding, so it is exactly a linear projection over flattened patches. The
# GEMM path (cuBLAS) is ~11x faster than cuDNN's conv3d for this shape and,
# unlike Conv3d, never dispatches to slow_conv_dilated3d on torch builds
# without cuDNN (measured 41 s vs 1.4 ms per 3 MP image). Weights stay in the
# Conv3d module, so state_dicts are unchanged in both directions.

import torch
import torch.nn.functional as F

from axolotl.utils.logging import get_logger

logger = get_logger(__name__)


def _linear_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    proj = self.proj
    in_features = (
        self.in_channels * self.temporal_patch_size * self.patch_size * self.patch_size
    )
    hidden_states = hidden_states.reshape(-1, in_features).to(dtype=proj.weight.dtype)
    return F.linear(
        hidden_states, proj.weight.reshape(proj.weight.shape[0], -1), proj.bias
    )


def patch_qwen3_vl_patch_embed_linear() -> None:
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLVisionPatchEmbed,
    )

    if getattr(Qwen3VLVisionPatchEmbed, "_axolotl_patch_embed_linear", False):
        return

    Qwen3VLVisionPatchEmbed.forward = _linear_forward
    Qwen3VLVisionPatchEmbed._axolotl_patch_embed_linear = True
    logger.info(
        "Patched Qwen3VLVisionPatchEmbed.forward to the equivalent patchify GEMM"
    )
