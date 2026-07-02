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

import os
from types import SimpleNamespace
from unittest import mock

import pytest
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


def test_gather_supported_sm90_sm120_and_disable_override():
    """The sparse gather is enabled on sm90 (Hopper) and sm120; unlisted archs fall back to dense;
    ``GLM_DSA_DISABLE_GATHER`` forces dense everywhere."""
    from axolotl.integrations.kernels.libs.glm_dsa import dispatch as D

    dev = torch.device("cuda:0")
    for cap, expected in [
        ((9, 0), True),
        ((12, 0), True),
        ((8, 0), False),
        ((10, 0), False),
    ]:
        D._GATHER_OK.clear()
        with mock.patch.object(torch.cuda, "get_device_capability", return_value=cap):
            assert D._gather_supported(dev) is expected, f"cap {cap}"
    D._GATHER_OK.clear()
    with mock.patch.dict(os.environ, {"GLM_DSA_DISABLE_GATHER": "1"}):
        with mock.patch.object(
            torch.cuda, "get_device_capability", return_value=(9, 0)
        ):
            assert D._gather_supported(dev) is False  # override wins


def test_mla_attn_routes_to_gather_under_packing_and_forwards_seq():
    """Above the crossover the sparse gather is used EVEN under sample packing (it is doc-aware), and
    ``seq_q``/``seq_k`` are forwarded to it; when the gather is unsupported, dense is used."""
    from axolotl.integrations.kernels.libs.glm_dsa import dispatch as D

    qa = torch.zeros(1, 4, 8, 576)
    ks = torch.zeros(1, 8, 576)
    idx = torch.zeros(1, 8, 4, dtype=torch.int64)
    seq = torch.zeros(1, 8, dtype=torch.int64)

    with (
        mock.patch.object(D, "mla_absorb_attn", return_value="GATHER") as mg,
        mock.patch.object(D, "dense_masked_out_latent", return_value="DENSE"),
        mock.patch.object(D, "_gather_supported", return_value=True),
        mock.patch.object(D, "calibrate_crossover", return_value=0),
    ):
        out = D.mla_attn(qa, ks, idx, 1.0, seq_q=seq, seq_k=seq)
        assert out == "GATHER"  # packing no longer forces dense
        # seq_q / seq_k forwarded as the last two positional args
        assert mg.call_args.args[-2] is seq and mg.call_args.args[-1] is seq

    with (
        mock.patch.object(D, "dense_masked_out_latent", return_value="DENSE"),
        mock.patch.object(D, "_gather_supported", return_value=False),
        mock.patch.object(D, "calibrate_crossover", return_value=0),
    ):
        assert D.mla_attn(qa, ks, idx, 1.0, seq_q=seq, seq_k=seq) == "DENSE"


def test_deep_ep_forward_applies_token_capacity():
    """The capacity cap must actually be applied to the routing before the dispatch layout is built
    (regression: the cap function existed but was never called)."""
    from axolotl.integrations.expert_parallel import experts_fn as E

    captured = {}

    class _FakeBuf:
        def get_dispatch_layout(self, topk_idx, e_global):
            captured["topk"] = topk_idx.clone()
            raise RuntimeError("stop")  # short-circuit the rest of the forward

    ntok, K, e_global = 64, 2, 8
    topk = torch.zeros(
        ntok, K, dtype=torch.int64
    )  # column 0 -> expert 0 for ALL tokens
    topk[:, 1] = 1
    w = torch.rand(ntok, K)
    self_mod = SimpleNamespace(num_experts=e_global, num_experts_global=e_global)

    E.set_token_capacity(4)
    try:
        with (
            mock.patch.object(E, "get_buffer", return_value=_FakeBuf()),
            mock.patch.object(E, "_get_valid_token_mask", return_value=None),
        ):
            with pytest.raises(RuntimeError, match="stop"):
                E._deep_ep_forward(
                    self_mod, torch.zeros(ntok, 16), topk, w, kernel_name="eager"
                )
    finally:
        E.set_token_capacity(None)

    # expert 0 was requested by all 64 tokens; the cap must hold it to <= 4
    assert int((captured["topk"] == 0).sum()) <= 4


def test_mla_attn_skips_calibration_when_gather_unsupported():
    """On an unsupported GPU mla_attn must NOT call calibrate_crossover (it would compile/run the
    gather Triton path we are avoiding)."""
    from axolotl.integrations.kernels.libs.glm_dsa import dispatch as D

    qa = torch.zeros(1, 4, 8, 576)
    ks = torch.zeros(1, 8, 576)
    idx = torch.zeros(1, 8, 4, dtype=torch.int64)
    calib = mock.MagicMock(return_value=0)
    with (
        mock.patch.object(D, "dense_masked_out_latent", return_value="DENSE"),
        mock.patch.object(D, "_gather_supported", return_value=False),
        mock.patch.object(D, "calibrate_crossover", calib),
    ):
        assert D.mla_attn(qa, ks, idx, 1.0) == "DENSE"
    calib.assert_not_called()


def test_cp_doc_ids_must_be_global_not_per_chunk():
    """F2: under context parallelism the per-document ids must come from the GLOBAL position_ids, not
    each rank's local chunk. Otherwise a document crossing a CP boundary gets different (and colliding)
    ids per rank, which masks valid remote keys / allows cross-document attention."""
    from axolotl.integrations.kernels.libs.glm_dsa.patch import (
        _seq_idx_from_position_ids,
    )

    # doc B (positions 0,1,2,3) spans the cp=2 boundary at index 4 (S=4 queries per rank)
    global_pos = torch.tensor([[0, 1, 0, 1, 2, 3, 0, 1]])
    S = 4
    global_ids = _seq_idx_from_position_ids(global_pos)
    assert global_ids.tolist() == [[1, 1, 2, 2, 2, 2, 3, 3]]

    # NEW (number docs on the global ids, then slice per rank): the boundary doc shares ONE id
    r0, r1 = global_ids[:, 0:S], global_ids[:, S : 2 * S]
    assert (
        r0[0, -1].item() == r1[0, 0].item()
    )  # doc B: rank0's last query == rank1's first query

    # OLD (per-local-chunk cumsum): the same doc gets different ids across ranks, and ids collide
    old_r0 = _seq_idx_from_position_ids(global_pos[:, 0:S])
    old_r1 = _seq_idx_from_position_ids(global_pos[:, S : 2 * S])
    assert (
        old_r0[0, -1].item() != old_r1[0, 0].item()
    )  # doc B inconsistent across ranks (the bug)
    assert (
        old_r1[0, 2].item() == old_r0[0, 0].item()
    )  # doc C (rank1) collides with doc A (rank0)
