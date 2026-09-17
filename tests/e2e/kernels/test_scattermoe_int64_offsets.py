# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Regression: expert stacks over 2^31 elements overflowed the i32 ``E_idx * stride``
pointer math in the scattermoe Triton kernels — an illegal memory access in the dW
kernel and silent out-of-bounds reads/writes in the forward (observed corrupting
neighboring allocations at Nemotron-3-Ultra / Kimi-K3-scale expert shapes).

The weight here is [40, 4096, 16384] = 2.68e9 elements (~5.4GB bf16); the last
expert's base offset (39 x 67.1M = 2.62e9) wraps negative in i32, so correctness
is checked on the LAST experts.
"""

import pytest
import torch

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
    pytest.mark.skipif(
        torch.cuda.is_available()
        and torch.cuda.get_device_properties(0).total_memory < 24e9,
        reason="needs ~16GB free VRAM",
    ),
]

E, K_IN, N_OUT, T = 40, 4096, 16384, 64


def _setup():
    from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
        flatten_sort_count,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(T, K_IN, device="cuda", dtype=torch.bfloat16, generator=g)
    w = (
        torch.randn(E, K_IN, N_OUT, device="cuda", dtype=torch.bfloat16, generator=g)
        * 0.02
    )
    # route half the tokens to the LAST expert (offsets past 2^31)
    idx = torch.randint(0, E, (T, 1), device="cuda", generator=g)
    idx[::2] = E - 1
    sei, ssi, eo = flatten_sort_count(idx, E)
    return x, w, idx, sei, ssi, eo


def test_forward_last_expert_matches_reference():
    from axolotl.integrations.kernels.libs.scattermoe_lora.kernels.ops import (
        scatter2scatter,
    )

    x, w, idx, sei, ssi, eo = _setup()
    y = scatter2scatter(x, w, sei, ssi, k=1)
    for e in (E - 1, E - 2):
        rows = (idx[:, 0] == e).nonzero(as_tuple=True)[0]
        if not rows.numel():
            continue
        ref = x[rows].float() @ w[e].float()
        got = y[rows].float()
        rel = (got - ref).abs().max() / ref.abs().max().clamp(min=1e-6)
        assert rel.item() < 5e-2, f"expert {e}: rel={rel.item():.4e}"
    assert torch.isfinite(y).all()


def test_group_bwd_w_last_expert_matches_reference():
    from axolotl.integrations.kernels.libs.scattermoe_lora.kernels.ops import (
        group_bwd_W,
    )

    x, w, idx, sei, ssi, eo = _setup()
    del w
    # grouped rows sorted by expert; dW stack is the same >2^31-element shape
    order = torch.argsort(idx[:, 0])
    xg = x[order].contiguous()
    dyg = torch.randn(
        T,
        N_OUT,
        device="cuda",
        dtype=torch.bfloat16,
        generator=torch.Generator(device="cuda").manual_seed(1),
    )
    counts = torch.bincount(idx[:, 0], minlength=E)
    offsets = counts.cumsum(0)
    dw, _ = group_bwd_W(dyg, xg, offsets, E)
    assert tuple(dw.shape) == (E, K_IN, N_OUT)
    start = int(offsets[E - 2])
    ref = xg[start:].float().T @ dyg[start:].float()
    got = dw[E - 1].float()
    rel = (got - ref).abs().max() / ref.abs().max().clamp(min=1e-6)
    assert rel.item() < 5e-2, f"dW last expert: rel={rel.item():.4e}"
    assert torch.isfinite(dw).all()
