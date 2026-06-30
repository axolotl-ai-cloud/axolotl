# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Regression tests for the GLM-DSA absorbed-MLA sparse gather kernel.

These guard two subtle, hard-won fixes:

1. **Leading-all-masked-block softmax NaN.** The online-softmax running max ``m_i`` starts at
   ``-inf``; a query whose *first* ``BN`` block of selected keys is entirely causally/doc-masked
   keeps ``m_new == -inf``, so ``alpha = exp(m_i - m_new) = exp(-inf + inf) = NaN`` — *even though
   the query has valid keys in later blocks*. This poisons the whole output. It surfaces at long
   context (``topk`` >> a short query's causal positions) and is arch-independent. The kernel guards
   it with ``m_safe = where(m_new == -inf, 0, m_new)`` (+ an ``l_i == 0`` division guard for
   fully-masked rows). The reference is the dense path, which is always finite.

2. **Var-len / document-aware masking under sample packing** (``seq_k[idx] == seq_q[s]``): the gather
   must match the dense per-document mask so packing can use the sparse path.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

from axolotl.integrations.kernels.libs.glm_dsa.attention_mla_absorb import (  # noqa: E402
    mla_absorb_attn,
)
from axolotl.integrations.kernels.libs.glm_dsa.config import (  # noqa: E402
    KV_LORA_RANK,
    QK_ROPE_HEAD_DIM,
)
from axolotl.integrations.kernels.libs.glm_dsa.dispatch import (  # noqa: E402
    dense_masked_out_latent,
)

DEV = "cuda"
DQK = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576 (512 + 64)


def _causal_topk_future_first(B, S, Skv, TOPK, device, seed=0):
    """Per query ``s`` select ``TOPK`` keys, ordering FUTURE (causally-invalid, ``> s``) positions
    FIRST so the leading topk block(s) are entirely masked — the exact trigger for the running-max
    NaN — while still including valid causal keys (and ``s`` itself) in later slots."""
    idx = torch.empty(B, S, TOPK, dtype=torch.int32, device=device)
    for s in range(S):
        future = torch.arange(s + 1, Skv, device=device)
        causal = torch.arange(0, s + 1, device=device)
        pool = torch.cat([future, causal])  # future FIRST -> leading masked
        if pool.numel() >= TOPK:
            row = pool[:TOPK].clone()
        else:
            pad = causal[-1:].repeat(TOPK - pool.numel())
            row = torch.cat([pool, pad])
        row[-1] = s  # guarantee self (valid) present
        idx[:, s, :] = row.to(torch.int32)
    return idx


@pytest.mark.parametrize("S,Skv,TOPK", [(64, 64, 48), (96, 96, 80)])
def test_gather_leading_masked_block_is_finite_and_matches_dense(S, Skv, TOPK):
    """The whole point: with leading masked blocks the gather must NOT NaN and must equal dense."""
    torch.manual_seed(0)
    B, H = 1, 16
    q = torch.randn(B, H, S, DQK, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, Skv, DQK, device=DEV, dtype=torch.bfloat16)
    idx = _causal_topk_future_first(B, S, Skv, TOPK, DEV)
    scale = 1.0 / (DQK**0.5)

    qg = q.clone().requires_grad_(True)
    kg = k.clone().requires_grad_(True)
    og = mla_absorb_attn(qg, kg, idx, scale, 0)
    og.float().pow(2).mean().backward()

    assert torch.isfinite(og).all(), (
        "gather forward produced NaN/Inf on leading-masked-block input"
    )
    assert torch.isfinite(qg.grad).all(), "gather dq produced NaN/Inf"
    assert torch.isfinite(kg.grad).all(), "gather dk produced NaN/Inf"

    qd = q.clone().requires_grad_(True)
    kd = k.clone().requires_grad_(True)
    od = dense_masked_out_latent(qd, kd, idx, scale, 0)
    od.float().pow(2).mean().backward()

    # bf16 tensor-core tolerance
    assert (og.float() - od.float()).abs().max().item() < 5e-2
    assert (qg.grad.float() - qd.grad.float()).abs().max().item() < 5e-2


def test_gather_fully_masked_query_is_finite():
    """A query whose every selected key is in the future (zero valid causal keys) must yield a
    finite (zero) row via the ``l_i == 0`` guard, not 0/0 = NaN."""
    torch.manual_seed(1)
    B, H, S, Skv, TOPK = 1, 8, 32, 32, 16
    q = torch.randn(B, H, S, DQK, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, Skv, DQK, device=DEV, dtype=torch.bfloat16)
    # causal-valid for everyone...
    idx = torch.zeros(B, S, TOPK, dtype=torch.int32, device=DEV)
    for s in range(S):
        idx[:, s, :] = torch.randint(0, s + 1, (TOPK,), device=DEV, dtype=torch.int32)
    # ...except query 5, whose every selected key is strictly future (invalid)
    idx[:, 5, :] = torch.arange(6, 6 + TOPK, device=DEV, dtype=torch.int32).clamp_(
        max=Skv - 1
    )
    out = mla_absorb_attn(q, k, idx, 1.0 / (DQK**0.5), 0)
    assert torch.isfinite(out).all(), (
        "fully-masked query produced NaN/Inf (0/0 not guarded)"
    )


def test_gather_docmask_matches_dense_under_packing():
    """Doc-aware (var-len) gather must match the dense per-document mask: a query only attends keys
    in its own document (``seq_k[idx] == seq_q[s]``).

    NB: the dense reference de-duplicates selected keys (a boolean ``scatter_``) while the gather
    sums each slot independently, so they only agree when ``idx`` has DISTINCT keys per query. We
    build distinct causal keys and assert the exact match on queries with enough causal positions to
    fill ``TOPK`` without padding; every query is still asserted finite.
    """
    torch.manual_seed(2)
    B, H, S, Skv, TOPK = 1, 16, 64, 64, 16
    q = torch.randn(B, H, S, DQK, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, Skv, DQK, device=DEV, dtype=torch.bfloat16)
    seq = torch.zeros(B, S, dtype=torch.int32, device=DEV)
    seq[:, S // 2 :] = 1  # two documents
    seq_q, seq_k = seq, seq.clone()
    # distinct causal topk (crosses the doc boundary so masking matters); short queries cycle their
    # available causal keys (creates dups -> excluded from the exact comparison via ``no_dup``).
    idx = torch.empty(B, S, TOPK, dtype=torch.int32, device=DEV)
    no_dup = torch.zeros(S, dtype=torch.bool, device=DEV)
    for s in range(S):
        causal = torch.randperm(s + 1, device=DEV).to(torch.int32)
        if causal.numel() >= TOPK:
            idx[:, s, :] = causal[:TOPK]
            no_dup[s] = True
        else:
            reps = (TOPK + causal.numel() - 1) // causal.numel()
            idx[:, s, :] = causal.repeat(reps)[:TOPK]
    scale = 1.0 / (DQK**0.5)

    og = mla_absorb_attn(q.clone(), k.clone(), idx, scale, 0, seq_q, seq_k)
    od = dense_masked_out_latent(q.clone(), k.clone(), idx, scale, 0, seq_q, seq_k)
    assert torch.isfinite(og).all()
    # exact agreement on the distinct-key (no-dup) queries only
    masked = (og.float() - od.float())[:, :, no_dup, :].abs()
    assert torch.isfinite(masked).all()
    assert masked.max().item() < 5e-2
    # doc mask must actually change the result vs causal-only (no seq)
    on = mla_absorb_attn(q.clone(), k.clone(), idx, scale, 0)
    assert (og.float() - on.float())[:, :, no_dup, :].abs().max().item() > 1e-2
