"""Liger non-loss kernels for GLM-4 MoE Lite."""

import torch


def liger_interleaved_rotary_pos_emb(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1
):
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

    # GLM rotates even/odd pairs but returns split-half rotary outputs.
    q = torch.cat((q[..., 0::2], q[..., 1::2]), dim=-1)
    k = torch.cat((k[..., 0::2], k[..., 1::2]), dim=-1)
    return liger_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)


def apply_liger_glm4_moe_lite(rope=False, rms_norm=False, glu_activation=False):
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
    from transformers.models.glm4_moe_lite import modeling_glm4_moe_lite

    if rope:
        modeling_glm4_moe_lite.apply_rotary_pos_emb = liger_rotary_pos_emb
        modeling_glm4_moe_lite.apply_rotary_pos_emb_interleave = (
            liger_interleaved_rotary_pos_emb
        )
    if rms_norm:
        modeling_glm4_moe_lite.Glm4MoeLiteRMSNorm = LigerRMSNorm
    if glu_activation:
        # Keep the constructor's intermediate_size override for shared experts.
        modeling_glm4_moe_lite.Glm4MoeLiteMLP.forward = LigerSwiGLUMLP.forward
