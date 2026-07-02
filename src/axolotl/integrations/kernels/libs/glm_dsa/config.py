"""Real GLM-5.2 (glm_moe_dsa) shapes for kernel development + validation.

Kernels are validated at the REAL model dims (head_dim 256, index_head_dim 128, top-k 2048, ...)
with a truncated layer count / per-op isolation — shrinking the dims would hide shape-specific
behavior (SMEM pressure at head_dim 256, the 128-wide indexer dot, the 2048 top-k gather). These
mirror ``GlmMoeDsaConfig`` defaults == the published GLM-5.2 checkpoint.
"""

from __future__ import annotations

# --- attention (DeepSeek-V3.2 MLA) ---
HIDDEN = 6144
N_HEADS = 64
N_KV_HEADS = 64
Q_LORA_RANK = 2048
KV_LORA_RANK = 512
QK_NOPE_HEAD_DIM = 192
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM  # 256
V_HEAD_DIM = 256
ATTN_SCALE = QK_HEAD_DIM**-0.5
ROPE_THETA = 8_000_000.0

# --- DSA Lightning indexer ---
INDEX_N_HEADS = 32
INDEX_HEAD_DIM = 128
INDEX_TOPK = 2048
INDEX_SOFTMAX_SCALE = INDEX_HEAD_DIM**-0.5

# --- MoE (not exercised by the attention/indexer/rope kernels) ---
N_ROUTED_EXPERTS = 256
N_SHARED_EXPERTS = 1
NUM_EXPERTS_PER_TOK = 8
MOE_INTERMEDIATE = 2048

NUM_HIDDEN_LAYERS = 78
FIRST_K_DENSE_REPLACE = 3
INDEX_TOPK_FREQ = 4


def probe_shapes(seq: int = 4096, batch: int = 1) -> dict:
    """A single (batch, seq) probe at real dims for kernel tests/benches."""
    return {
        "B": batch,
        "S": seq,
        "H": N_HEADS,
        "D": QK_HEAD_DIM,
        "Dv": V_HEAD_DIM,
        "IDX_H": INDEX_N_HEADS,
        "IDX_D": INDEX_HEAD_DIM,
        "TOPK": min(INDEX_TOPK, seq),
        "scale": ATTN_SCALE,
        "idx_scale": INDEX_SOFTMAX_SCALE,
    }
