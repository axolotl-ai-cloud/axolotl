# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""End-to-end correctness: our DSA-kernel attention path == HuggingFace eager GlmMoeDsaAttention.

The patched ``_glm_dsa_attention_forward`` replaces the eager forward with the absorbed-MLA path
(indexer top-k -> absorbed query -> sparse gather/dense over the compressed latent -> value
projection). This asserts it numerically matches the stock eager module on identical weights/inputs,
covering both the dense (topk >= seq) and sparse (topk < seq) regimes, with the stock and the fused
indexer. Combined with test_mla_absorb_gather (gather == dense), the full eager->dense->gather chain
is verified.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

pytest.importorskip(
    "transformers.models.glm_moe_dsa.modeling_glm_moe_dsa",
    reason="glm_moe_dsa required",
)
from transformers.models.glm_moe_dsa.configuration_glm_moe_dsa import (  # noqa: E402
    GlmMoeDsaConfig,
)
from transformers.models.glm_moe_dsa.modeling_glm_moe_dsa import (  # noqa: E402
    GlmMoeDsaAttention,
    GlmMoeDsaRotaryEmbedding,
)

from axolotl.integrations.kernels.libs.glm_dsa.patch import (  # noqa: E402
    patch_glm_moe_dsa_attention,
)

DEV = "cuda"
DT = torch.bfloat16


def _build(index_topk, seed=0):
    torch.manual_seed(seed)
    cfg = GlmMoeDsaConfig(
        hidden_size=512,
        num_attention_heads=2,
        num_key_value_heads=2,
        q_lora_rank=256,
        kv_lora_rank=512,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        index_topk=index_topk,
        index_head_dim=128,
        index_n_heads=2,
        num_hidden_layers=1,
    )
    cfg._attn_implementation = "eager"
    attn = GlmMoeDsaAttention(cfg, layer_idx=0).to(DEV, DT).eval()
    rope = GlmMoeDsaRotaryEmbedding(cfg).to(DEV)
    return cfg, attn, rope


@pytest.mark.parametrize("S,index_topk", [(64, 16), (48, 128)], ids=["sparse", "dense"])
@pytest.mark.parametrize(
    "use_fused_indexer", [False, True], ids=["eager_idx", "fused_idx"]
)
def test_dsa_attention_matches_hf_eager(S, index_topk, use_fused_indexer):
    cfg, attn, rope = _build(index_topk)
    B = 1
    hidden = torch.randn(B, S, cfg.hidden_size, device=DEV, dtype=DT)
    position_ids = torch.arange(S, device=DEV).unsqueeze(0)
    cos, sin = rope(hidden, position_ids)
    pe = (cos, sin)

    import torch.nn as nn

    with torch.no_grad():
        out_eager = attn(hidden, pe, None, position_ids=position_ids)[0]
        wrapper = nn.Module()
        wrapper.add_module("a", attn)
        assert (
            patch_glm_moe_dsa_attention(wrapper, use_fused_indexer=use_fused_indexer)
            == 1
        )
        out_kernel = attn(hidden, pe, None, position_ids=position_ids)[0]

    assert torch.isfinite(out_kernel).all()
    err = (out_eager.float() - out_kernel.float()).abs().max().item()
    scale = out_eager.float().abs().max().item() + 1e-6
    assert err / scale < 5e-2, f"rel err {err / scale:.4g} (abs {err:.4g})"
