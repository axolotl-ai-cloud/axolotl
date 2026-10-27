# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""GPU parity of the SonicMoE LoRA split path against materialized ``W + s * B @ A``.

Three layers: each kernel primitive (``SonicOps``) against its plain-torch twin (``TorchOps``),
including the Triton GLU backward; a Qwen3-MoE experts module run through the real
``_sonicmoe_forward`` / EP entry points with ``sonicmoe_lora_split`` on and off and against an
fp32 reference, with sentinel pairs, an idle expert and a token with no local expert; and one
Beam EP=8 shape that must use less peak memory than the materialized path.
"""

import importlib.util
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


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

E_LOCAL, TOP_K, NUM_TOKENS, HIDDEN, INTER, RANK, SCALING = 8, 4, 300, 512, 256, 16, 2.0
IDLE_EXPERT, NO_LOCAL_TOKEN = 7, 5


@pytest.fixture(scope="module")
def sonic_ops():
    from transformers.integrations.sonicmoe import is_sonicmoe_loadable

    from axolotl.integrations.kernels.libs.sonicmoe.lora_split import SonicOps
    from axolotl.integrations.kernels.plugin import _shim_nvvm_mbarrier_arrive

    if not is_sonicmoe_loadable():
        pytest.skip("sonic-moe kernel not loadable")
    _shim_nvvm_mbarrier_arrive()
    return SonicOps()


@pytest.fixture(autouse=True)
def restore_flags():
    from axolotl.integrations.kernels.libs.sonicmoe import experts, lora_split

    yield
    lora_split.configure_lora_split(False)
    experts.configure_ep_compact_routing()
    experts.configure_ep_row_padding()


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6)).item()


def _pairs(num_tokens, num_experts, top_k, gen, sentinel_rate=0.3):
    idx = torch.randint(
        0, num_experts - 1, (num_tokens, top_k), device="cuda", generator=gen
    )
    sentinel = (
        torch.rand(num_tokens, top_k, device="cuda", generator=gen) < sentinel_rate
    )
    sentinel[NO_LOCAL_TOKEN] = True
    idx = idx.masked_fill(sentinel, num_experts)
    weights = torch.rand(num_tokens, top_k, device="cuda", generator=gen)
    weights = weights.masked_fill(sentinel, 0)
    tokens = (
        torch.arange(num_tokens, device="cuda", dtype=torch.int32)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    return tokens, idx.reshape(-1).int(), weights.reshape(-1)


def test_triton_glu_backward_matches_reference():
    from axolotl.integrations.kernels.libs.sonicmoe.glu_bwd_triton import glu_bwd_add_
    from axolotl.integrations.kernels.libs.sonicmoe.lora_split import TorchOps

    gen = torch.Generator(device="cuda").manual_seed(0)
    num_slots, inter = 777, 320
    cu = torch.tensor([0, 100, 100, 640], dtype=torch.int32, device="cuda")
    for activation in ("swiglu", "geglu"):
        h = (
            torch.randn(num_slots, 2 * inter, device="cuda", generator=gen) * 3
        ).bfloat16()
        dh = torch.randn(num_slots, 2 * inter, device="cuda", generator=gen).bfloat16()
        d = torch.randn(num_slots, inter, device="cuda", generator=gen).bfloat16()
        want = dh.clone()
        TorchOps().glu_bwd_add_(want, h, d, cu, activation)
        got = dh.clone()
        glu_bwd_add_(got, h, d, cu, activation)
        assert torch.equal(got[640:], dh[640:]), activation
        torch.testing.assert_close(got, want, rtol=1.6e-2, atol=1e-2)


def test_sonic_primitives_match_torch_ops(sonic_ops):
    from axolotl.integrations.kernels.libs.sonicmoe.lora_split import TorchOps

    ref = TorchOps()
    gen = torch.Generator(device="cuda").manual_seed(1)
    tokens, expert_ids, scores = _pairs(NUM_TOKENS, E_LOCAL, TOP_K, gen)
    meta = sonic_ops.routing_metadata(tokens, expert_ids, NUM_TOKENS, E_LOCAL)
    want_meta = ref.routing_metadata(tokens, expert_ids, NUM_TOKENS, E_LOCAL)
    n_valid = int(want_meta.cu_seqlens[-1])
    assert torch.equal(meta.cu_seqlens, want_meta.cu_seqlens)
    assert torch.equal(meta.token_offset, want_meta.token_offset)
    assert torch.equal(meta.s_reverse_scatter_idx, want_meta.s_reverse_scatter_idx)
    assert torch.equal(meta.s_scatter_idx[:n_valid], want_meta.s_scatter_idx[:n_valid])
    assert torch.equal(meta.x_gather_idx[:n_valid], want_meta.x_gather_idx[:n_valid])
    cu, xg = meta.cu_seqlens, meta.x_gather_idx
    num_slots = tokens.numel()

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=gen) * scale).bfloat16()

    def both(fn, *shape_out):
        outs = []
        for ops in (sonic_ops, ref):
            out = torch.empty(*shape_out, dtype=torch.bfloat16, device="cuda")
            outs.append(fn(ops, out))
        return outs

    x = rand(NUM_TOKENS, HIDDEN)
    p = rand(E_LOCAL, HIDDEN, RANK, scale=0.05)
    got, want = both(lambda o, out: o.gemm(x, p, out, cu, xg), num_slots, RANK)
    assert _rel(got[:n_valid], want[:n_valid]) < 1e-2

    u = rand(num_slots, RANK)
    q = rand(E_LOCAL, RANK, 2 * INTER)
    c_src = rand(num_slots, 2 * INTER)
    got, want = both(
        lambda o, out: o.gemm(u, q, out.copy_(c_src), cu, alpha=SCALING, c=out),
        num_slots,
        2 * INTER,
    )
    assert _rel(got[:n_valid], want[:n_valid]) < 1e-2

    b = rand(num_slots, RANK)
    got, want = both(
        lambda o, out: o.gemm_k(x.t(), b, out, cu, xg, alpha=SCALING),
        E_LOCAL,
        HIDDEN,
        RANK,
    )
    assert _rel(got, want) < 1e-2
    assert not got[IDLE_EXPERT].any()

    w1 = rand(E_LOCAL, 2 * INTER, HIDDEN, scale=HIDDEN**-0.5).transpose(1, 2)
    c1 = rand(num_slots, 2 * INTER)
    outs = []
    for ops in (sonic_ops, ref):
        h = torch.empty(num_slots, 2 * INTER, dtype=torch.bfloat16, device="cuda")
        a = torch.empty(num_slots, INTER, dtype=torch.bfloat16, device="cuda")
        ops.gemm_gated(x, w1, h.copy_(c1), h, a, cu, xg, "swiglu", True)
        outs.append((h, a))
    for got, want in zip(*outs, strict=True):
        assert _rel(got[:n_valid], want[:n_valid]) < 1e-2

    h = outs[1][0]
    w2 = rand(E_LOCAL, HIDDEN, INTER, scale=INTER**-0.5)
    dout = rand(NUM_TOKENS, HIDDEN)
    sigma = scores[meta.s_scatter_idx.long()].float()
    outs = []
    for ops in (sonic_ops, ref):
        dh = torch.empty_like(h)
        a_prime = torch.empty(num_slots, INTER, dtype=torch.bfloat16, device="cuda")
        colvec = ops.gemm_dgated(dout, w2, h, dh, a_prime, sigma, cu, xg, "swiglu")
        outs.append((dh, a_prime, colvec))
    for got, want in zip(*outs, strict=True):
        assert _rel(got[:n_valid], want[:n_valid]) < 1e-2

    rev, tok = meta.s_reverse_scatter_idx, meta.token_offset
    y = rand(num_slots, HIDDEN)
    got = sonic_ops.router_forward(y, scores.float(), rev, tok, NUM_TOKENS, E_LOCAL)
    want = ref.router_forward(y, scores.float(), rev, tok, NUM_TOKENS, E_LOCAL)
    assert _rel(got, want) < 1e-2
    got = sonic_ops.token_reduce(y, rev, tok, NUM_TOKENS, E_LOCAL)
    want = ref.token_reduce(y, rev, tok, NUM_TOKENS, E_LOCAL)
    assert _rel(got, want) < 1e-2
    src = torch.randn(num_slots, device="cuda", generator=gen)
    assert torch.equal(
        sonic_ops.gather_valid_rows(src, rev), ref.gather_valid_rows(src, rev)
    )


@pytest.fixture(scope="module")
def experts_module():
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.integrations.sonicmoe import is_sonicmoe_loadable

    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        register_sonicmoe_experts,
    )
    from axolotl.integrations.kernels.plugin import _shim_nvvm_mbarrier_arrive

    if not is_sonicmoe_loadable():
        pytest.skip("sonic-moe kernel not loadable")
    _shim_nvvm_mbarrier_arrive()
    register_sonicmoe_experts()
    config = AutoConfig.for_model("qwen3_moe")
    config.hidden_size = HIDDEN
    config.intermediate_size = 1024
    config.moe_intermediate_size = INTER
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
    experts.requires_grad_(False)
    experts.train()
    return experts


def _receive_grid():
    gen = torch.Generator(device="cuda").manual_seed(0)
    idx = torch.randint(
        0, E_LOCAL - 1, (NUM_TOKENS, TOP_K), device="cuda", generator=gen
    )
    remote = torch.rand(NUM_TOKENS, TOP_K, device="cuda", generator=gen) < 0.5
    remote[NO_LOCAL_TOKEN] = True
    idx = idx.masked_fill(remote, E_LOCAL)
    weights = torch.rand(NUM_TOKENS, TOP_K, device="cuda", generator=gen)
    weights = weights.masked_fill(remote, 0).to(torch.bfloat16)
    x = torch.randn(NUM_TOKENS, HIDDEN, device="cuda", generator=gen).to(torch.bfloat16)
    return x, idx, weights


def _lora(experts):
    gen = torch.Generator(device="cuda").manual_seed(1)

    def pair(weight):
        e, dim1, dim2 = weight.shape
        a = torch.randn(RANK * e, dim2, device="cuda", generator=gen) * 0.02
        b = torch.randn(dim1, RANK * e, device="cuda", generator=gen) * 0.02
        return a.to(weight.dtype), b.to(weight.dtype)

    return pair(experts.gate_up_proj), pair(experts.down_proj)


def _run(experts, x, idx, weights, lora, *, split, ep, compact=False):
    from axolotl.integrations.kernels.libs.sonicmoe import experts as sonic, lora_split

    lora_split.configure_lora_split(split)
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
    calls = []
    original = sonic.sonicmoe_lora_split_forward

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    sonic.sonicmoe_lora_split_forward = spy
    try:
        if ep:
            experts.num_experts_global = 4 * E_LOCAL
            out = sonic._sonicmoe_ep_forward(experts, x, idx, weights)
        else:
            experts.num_experts_global = E_LOCAL
            out = sonic._sonicmoe_forward(experts, x, idx, weights)
    finally:
        sonic.sonicmoe_lora_split_forward = original
        del experts._sonicmoe_lora
    assert bool(calls) == split
    params = (x, weights, a1, b1, a2, b2)
    grads = torch.autograd.grad(out.float().square().sum(), params)
    return out.detach(), grads


def _reference(experts, x, idx, weights, lora):
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


NAMES = (
    "d_hidden",
    "d_router_weights",
    "d_lora_A1",
    "d_lora_B1",
    "d_lora_A2",
    "d_lora_B2",
)


@pytest.mark.parametrize(
    ("ep", "compact"), [(False, False), (True, False), (True, True)]
)
def test_split_matches_materialized_and_reference(experts_module, ep, compact):
    x, idx, weights = _receive_grid()
    if not ep:
        idx = idx.masked_fill(idx == E_LOCAL, 0)
    assert not (idx == IDLE_EXPERT).any()
    lora = _lora(experts_module)
    kwargs = {"ep": ep, "compact": compact}
    out_mat, g_mat = _run(experts_module, x, idx, weights, lora, split=False, **kwargs)
    out_split, g_split = _run(
        experts_module, x, idx, weights, lora, split=True, **kwargs
    )
    out_ref, g_ref = _reference(experts_module, x, idx, weights, lora)

    assert _rel(out_split, out_ref) < 2e-2
    assert _rel(out_split, out_mat) < 2e-2
    for name, gs, gm, gr in zip(NAMES, g_split, g_mat, g_ref, strict=True):
        assert torch.isfinite(gs).all(), name
        assert _rel(gs, gr) < 3e-2, name
        assert _rel(gs, gm) < 3e-2, name

    d_hidden = g_split[0]
    if ep:
        assert not out_split[NO_LOCAL_TOKEN].any()
        assert not d_hidden[NO_LOCAL_TOKEN].any()
        assert not g_split[1][idx == E_LOCAL].any()
    for d_a, d_b in ((g_split[2], g_split[3]), (g_split[4], g_split[5])):
        assert not d_a[IDLE_EXPERT * RANK : (IDLE_EXPERT + 1) * RANK].any()
        assert not d_b.reshape(d_b.shape[0], RANK, -1)[:, :, IDLE_EXPERT].any()


def _beam_case(num_pairs, gen, num_experts=32, hidden=5120, inter=2560, rank=64):
    """A Beam EP=8 rank: 32 local experts, compacted pairs, 4 local experts per token."""
    per_token = 4
    num_tokens = num_pairs // per_token
    w1 = (
        torch.randn(num_experts, 2 * inter, hidden, device="cuda", generator=gen)
        * hidden**-0.5
    ).bfloat16()
    w2 = (
        torch.randn(num_experts, hidden, inter, device="cuda", generator=gen)
        * inter**-0.5
    ).bfloat16()

    def pair(weight):
        e, dim1, dim2 = weight.shape
        a = torch.randn(rank * e, dim2, device="cuda", generator=gen) * 0.01
        b = torch.randn(dim1, rank * e, device="cuda", generator=gen) * 0.01
        return a.bfloat16().requires_grad_(True), b.bfloat16().requires_grad_(True)

    tokens = (
        torch.arange(num_tokens, device="cuda", dtype=torch.int32)
        .repeat_interleave(per_token)
        .contiguous()
    )
    expert_ids = torch.rand(num_tokens, num_experts, device="cuda", generator=gen)
    expert_ids = expert_ids.argsort(dim=1)[:, :per_token].reshape(-1).int()
    scores = torch.rand(num_pairs, device="cuda", generator=gen).bfloat16()
    x = torch.randn(num_tokens, hidden, device="cuda", generator=gen).bfloat16()
    facade = SimpleNamespace(
        has_post_expert_norm=False,
        config=SimpleNamespace(hidden_act="silu"),
        is_transposed=False,
        is_concatenated=True,
        gate_up_proj=w1,
        down_proj=w2,
        gate_up_proj_bias=None,
        down_proj_bias=None,
        has_bias=False,
        num_experts=num_experts,
    )
    return facade, x, (tokens, expert_ids, scores), pair(w1), pair(w2)


def run_materialized(facade, x, pairs, lora1, lora2, scaling=2.0):
    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        _sonicmoe_pairs_forward,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.lora import MoELoRAMaterialize

    merged = SimpleNamespace(**vars(facade))
    merged.gate_up_proj = MoELoRAMaterialize.apply(facade.gate_up_proj, *lora1, scaling)
    merged.down_proj = MoELoRAMaterialize.apply(facade.down_proj, *lora2, scaling)
    return _sonicmoe_pairs_forward(merged, x, *pairs)


def run_split(facade, x, pairs, lora1, lora2, scaling=2.0):
    from axolotl.integrations.kernels.libs.sonicmoe.lora_split import (
        sonicmoe_lora_split_forward,
    )

    return sonicmoe_lora_split_forward(
        facade, x, *pairs, (*lora1, scaling), (*lora2, scaling)
    )


def _step_peak_and_ms(fn, facade, x, pairs, lora1, lora2, iters=3):
    leaves = (x, pairs[2], *lora1, *lora2)

    def step():
        out = fn(facade, x, pairs, lora1, lora2)
        torch.autograd.grad(out.float().square().mean(), leaves)

    step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    start = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - start) / iters * 1e3
    return torch.cuda.max_memory_allocated() - base, ms


@pytest.mark.slow
def test_beam_shape_uses_less_peak_memory(sonic_ops):
    if torch.cuda.get_device_properties(0).total_memory < 100 * 2**30:
        pytest.skip("needs a >=100 GiB GPU for the Beam EP=8 shape")
    gen = torch.Generator(device="cuda").manual_seed(0)
    facade, x, pairs, lora1, lora2 = _beam_case(131072, gen)
    x.requires_grad_(True)
    scores = pairs[2].clone().requires_grad_(True)
    pairs = (pairs[0], pairs[1], scores)
    mat_peak, mat_ms = _step_peak_and_ms(
        run_materialized, facade, x, pairs, lora1, lora2
    )
    split_peak, split_ms = _step_peak_and_ms(run_split, facade, x, pairs, lora1, lora2)
    print(
        f"\nmaterialized: {mat_peak / 2**30:.2f} GiB {mat_ms:.1f} ms | "
        f"split: {split_peak / 2**30:.2f} GiB {split_ms:.1f} ms"
    )
    assert split_peak < mat_peak - 2 * 2**30
