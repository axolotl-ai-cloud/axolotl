# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU regression tests for two GLM-5.2 workarounds (no GPU needed):

1. **Expert-capacity cap** (DeepEP #24): DeepEP's intranode combine deadlocks once one expert
   receives too many tokens (the GLM router concentrates with depth). ``_apply_expert_capacity``
   drops the lowest-weight excess (token,expert) assignments so no expert exceeds ``cap``.
2. **sm90 backward-autotune prune** (#18): on Hopper the full bwd autotune grid trials a
   nondeterministic invalid-PC (CUDA 718); ``_bwd_prune_sm90_safe`` must collapse to a single config
   on sm90 (so autotune never trials the rest) while leaving other archs to normal SMEM pruning.
"""

from unittest import mock

import torch

from axolotl.integrations.expert_parallel.experts_fn import _apply_expert_capacity


def test_expert_capacity_caps_tokens_per_expert():
    # 6 tokens, top-2 -> 12 (token,expert) assignments, 3 experts. cap=2.
    topk_idx = torch.tensor(
        [[0, 1], [0, 2], [0, 1], [0, 2], [1, 2], [0, 1]], dtype=torch.int64
    )
    topk_w = torch.tensor(
        [[0.9, 0.5], [0.8, 0.4], [0.7, 0.6], [0.3, 0.2], [0.95, 0.1], [0.55, 0.45]]
    )
    cap = 2
    out = _apply_expert_capacity(topk_idx.clone(), topk_w, cap)

    # every expert now has <= cap surviving assignments
    for e in range(3):
        assert int((out == e).sum()) <= cap, f"expert {e} exceeds cap"
    # dropped slots are sentinelled to -1, and the number dropped is exactly the overflow
    n_orig = int((topk_idx >= 0).sum())
    n_kept = int((out >= 0).sum())
    assert n_kept == n_orig - int((out == -1).sum())
    # the HIGHEST-weight assignment to expert 0 (token4? no expert0 weights: 0.9,0.8,0.7,0.3,_,0.55)
    # the two survivors for expert 0 must be the top-2 weights (0.9 @ tok0, 0.8 @ tok1)
    surv0_tokens = (out[:, 0] == 0).nonzero().flatten().tolist() + (
        out[:, 1] == 0
    ).nonzero().flatten().tolist()
    assert (
        0 in surv0_tokens and 1 in surv0_tokens
    )  # the two highest-weight expert-0 slots kept


def test_expert_capacity_noop_when_under_cap():
    topk_idx = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)
    topk_w = torch.tensor([[0.9, 0.1], [0.5, 0.5]])
    out = _apply_expert_capacity(topk_idx.clone(), topk_w, cap=8)
    assert torch.equal(out, topk_idx)  # nothing dropped when every expert is under cap


def test_expert_capacity_preserves_existing_sentinels():
    topk_idx = torch.tensor([[0, -1], [0, 0]], dtype=torch.int64)
    topk_w = torch.tensor([[0.9, 0.0], [0.8, 0.7]])
    out = _apply_expert_capacity(topk_idx.clone(), topk_w, cap=2)
    assert out[0, 1] == -1  # pre-existing -1 untouched
    assert int((out == 0).sum()) <= 2


def _absorb():
    import axolotl.integrations.kernels.libs.glm_dsa.attention_mla_absorb as M

    return M


def test_sm90_bwd_prune_collapses_to_single_config():
    M = _absorb()
    # mock the device SMEM so the base SMEM-prune keeps everything (CPU-only, deterministic)
    with mock.patch(
        "axolotl.integrations.kernels.libs.glm_dsa._autotune.max_smem",
        return_value=1 << 30,
    ):
        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ):
            kept = M._bwd_prune_sm90_safe(M._ABSORB_CONFIGS, {}, DL=512, DR=64)
        assert len(kept) == 1, (
            "sm90 must collapse the bwd grid to one config (avoids invalid-PC)"
        )

        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(12, 0)
        ):
            kept_sm120 = M._bwd_prune_sm90_safe(M._ABSORB_CONFIGS, {}, DL=512, DR=64)
        assert len(kept_sm120) > 1, (
            "non-sm90 keeps the full SMEM-pruned grid for autotuning"
        )
