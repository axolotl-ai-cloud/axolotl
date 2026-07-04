"""GLM-5.2 (glm_moe_dsa) DSA fused training kernels.

DeepSeek-V3.2 sparse-MLA (DSA / Lightning-Indexer) attention for GLM-5.2: MLA weight absorption +
head-batched sparse-gather flash attention (fwd+bwd), a fused Lightning-Indexer scorer + top-k,
length-aware dense/sparse dispatch, and a context-parallel path that all-gathers only the 576-wide
compressed KV. See ``patch.patch_glm_moe_dsa_attention``.
"""

from .patch import (
    fused_indexer_topk,
    keep_router_fp32,
    patch_glm_moe_dsa_attention,
)

__all__ = [
    "patch_glm_moe_dsa_attention",
    "keep_router_fp32",
    "fused_indexer_topk",
]
