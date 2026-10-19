# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""GPU parity of SonicMoE EP local-pair compaction against the tokens x top_k grid.

One Qwen3-MoE experts module is treated as an expert-parallel shard (``num_experts_global``
> ``num_experts``), fed an EP-style receive grid (local ids, sentinel ``num_experts`` with
weight 0), and run three ways: transformers' grid path (``sonicmoe_ep_compact_routing`` off),
the compacted path (on), and a pure-torch fp32 reference. Forward output, d_hidden_states,
d_router_weights and the LoRA grads must agree to bf16 tolerance. One expert
gets no tokens and one token has no local expert; that token's output and grad must be
exactly zero on both kernel paths.
"""

import importlib.util

import pytest
import torch
import torch.nn.functional as F

E_LOCAL, TOP_K, NUM_TOKENS, RANK, SCALING = 8, 4, 300, 8, 2.0
IDLE_EXPERT, NO_LOCAL_TOKEN = 7, 5


def _is_hopper_or_newer() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 9


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA GPU"),
    pytest.mark.skipif(
        not _is_hopper_or_newer(),
        reason="SonicMoE requires Hopper (sm_90) or Blackwell (sm_100+)",
    ),
    pytest.mark.skipif(
        importlib.util.find_spec("kernels") is None,
        reason="HF `kernels` package not installed",
    ),
]


@pytest.fixture(scope="module")
def experts_module():
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.integrations.sonicmoe import is_sonicmoe_loadable

    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        register_sonicmoe_experts,
    )

    if not is_sonicmoe_loadable():
        pytest.skip("sonic-moe kernel not loadable")
    register_sonicmoe_experts()
    config = AutoConfig.for_model("qwen3_moe")
    config.hidden_size = 256
    config.intermediate_size = 512
    config.moe_intermediate_size = 128
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 64
    config.num_hidden_layers = 1
    config.num_experts = E_LOCAL
    config.num_experts_per_tok = TOP_K
    config.vocab_size = 128
    config._experts_implementation = "sonicmoe"
    model = AutoModelForCausalLM.from_config(config).cuda().bfloat16()
    experts = model.model.layers[0].mlp.experts
    experts.num_experts_global = 4 * E_LOCAL
    experts.train()
    return experts


@pytest.fixture(autouse=True)
def restore_padding():
    from axolotl.integrations.kernels.libs.sonicmoe import experts

    yield
    experts.configure_ep_compact_routing()
    experts.configure_ep_row_padding()


def _receive_grid():
    gen = torch.Generator(device="cuda").manual_seed(0)
    idx = torch.randint(
        0, E_LOCAL - 1, (NUM_TOKENS, TOP_K), device="cuda", generator=gen
    )
    remote = torch.rand(NUM_TOKENS, TOP_K, device="cuda", generator=gen) < 0.7
    remote[NO_LOCAL_TOKEN] = True
    idx = idx.masked_fill(remote, E_LOCAL)
    weights = torch.rand(NUM_TOKENS, TOP_K, device="cuda", generator=gen)
    weights = weights.masked_fill(remote, 0).to(torch.bfloat16)
    x = torch.randn(NUM_TOKENS, 256, device="cuda", generator=gen).to(torch.bfloat16)
    return x, idx, weights


def _lora(experts):
    gen = torch.Generator(device="cuda").manual_seed(1)
    w1, w2 = experts.gate_up_proj, experts.down_proj

    def pair(weight):
        e, dim1, dim2 = weight.shape
        a = torch.randn(RANK * e, dim2, device="cuda", generator=gen) * 0.02
        b = torch.randn(dim1, RANK * e, device="cuda", generator=gen) * 0.02
        return a.to(weight.dtype), b.to(weight.dtype)

    return pair(w1), pair(w2)


def _run(experts, compact, x, idx, weights, lora):
    from axolotl.integrations.kernels.libs.sonicmoe import experts as sonic

    sonic.configure_ep_row_padding("highwater", granule=256)
    sonic.configure_ep_compact_routing(compact, granule=64)
    x = x.clone().requires_grad_(True)
    weights = weights.clone().requires_grad_(True)
    (a1, b1), (a2, b2) = (
        tuple(t.clone().requires_grad_(True) for t in p) for p in lora
    )
    experts._sonicmoe_lora = {
        "gate_up_proj": (a1, b1, SCALING),
        "down_proj": (a2, b2, SCALING),
    }
    try:
        out = sonic._sonicmoe_ep_forward(experts, x, idx, weights)
    finally:
        del experts._sonicmoe_lora
    # base expert weights are frozen and enter W_eff detached on the LoRA path
    params = (x, weights, a1, b1, a2, b2)
    grads = torch.autograd.grad(out.float().square().sum(), params)
    return out.detach(), grads


def _reference(experts, x, idx, weights, lora):
    """fp32 torch: W_eff = W + s * B @ A per expert, then the grid semantics."""
    x = x.float().clone().requires_grad_(True)
    weights = weights.float().clone().requires_grad_(True)
    w1 = experts.gate_up_proj.detach().float()
    w2 = experts.down_proj.detach().float()
    (a1, b1), (a2, b2) = (
        tuple(t.float().clone().requires_grad_(True) for t in p) for p in lora
    )

    def eff(w, a, b):
        e, dim1, dim2 = w.shape
        return w + SCALING * torch.einsum(
            "ore,eri->eoi", b.reshape(dim1, RANK, e), a.reshape(e, RANK, dim2)
        )

    w1_eff, w2_eff = eff(w1, a1, b1), eff(w2, a2, b2)
    out = torch.zeros_like(x)
    for e in range(E_LOCAL):
        rows, slots = (idx == e).nonzero(as_tuple=True)
        if rows.numel() == 0:
            continue
        gate, up = F.linear(x[rows], w1_eff[e]).chunk(2, dim=-1)
        y = F.linear(F.silu(gate) * up, w2_eff[e])
        out = out.index_add(0, rows, y * weights[rows, slots, None])
    params = (x, weights, a1, b1, a2, b2)
    grads = torch.autograd.grad(out.square().sum(), params)
    return out.detach(), grads


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6)).item()


NAMES = (
    "d_hidden",
    "d_router_weights",
    "d_lora_A1",
    "d_lora_B1",
    "d_lora_A2",
    "d_lora_B2",
)


def test_compacted_routing_matches_grid_and_reference(experts_module):
    x, idx, weights = _receive_grid()
    assert not (idx == IDLE_EXPERT).any()
    lora = _lora(experts_module)
    out_grid, g_grid = _run(experts_module, False, x, idx, weights, lora)
    out_pairs, g_pairs = _run(experts_module, True, x, idx, weights, lora)
    out_ref, g_ref = _reference(experts_module, x, idx, weights, lora)

    assert _rel(out_pairs, out_ref) < 2e-2
    assert _rel(out_pairs, out_grid) < 2e-2
    for name, gp, gg, gr in zip(NAMES, g_pairs, g_grid, g_ref, strict=True):
        assert _rel(gp, gr) < 3e-2, name
        assert _rel(gp, gg) < 3e-2, name

    for out, grads in ((out_grid, g_grid), (out_pairs, g_pairs)):
        assert torch.equal(out[NO_LOCAL_TOKEN], torch.zeros_like(out[NO_LOCAL_TOKEN]))
        d_hidden = grads[0]
        assert torch.equal(
            d_hidden[NO_LOCAL_TOKEN], torch.zeros_like(d_hidden[NO_LOCAL_TOKEN])
        )
        # lora_A rows are expert-major (E, r, in); lora_B columns are r-major, expert innermost
        for d_a, d_b in ((grads[2], grads[3]), (grads[4], grads[5])):
            assert not d_a[IDLE_EXPERT * RANK : (IDLE_EXPERT + 1) * RANK].any()
            assert not d_b.reshape(d_b.shape[0], RANK, -1)[:, :, IDLE_EXPERT].any()
        assert not grads[1][idx == E_LOCAL].any()
