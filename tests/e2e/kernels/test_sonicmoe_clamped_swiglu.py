# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""GPU parity of SonicMoE's clamped-SwiGLU experts (DeepSeek-V4, GLM-5) against the eager experts.

Two layers: the quack primitives with the runtime-registered ``swiglu_limit10`` epilogue and the
Triton GLU backward against their plain-torch twins (``TorchOps``); and a DeepSeek-V4-Flash-shaped
experts layer (H=4096, I=2048, top-k 6, and GLM-5.3-Flash's top-k 8) through the real
``_sonicmoe_forward`` for full fine-tuning, materialized LoRA and the LoRA split, against the
module's eager loop in fp32. Expert weights are scaled so the clamp is active on a few percent of
gate/up entries; bf16 rounding at the bound flips some of those, so each error is held to that of
the eager module run in bf16 rather than to a fixed bf16 tolerance alone.
"""

import importlib.util
import math

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
    # sm90+ only: the B200 lane is the only single-GPU lane newer than L40S
    pytest.mark.gpu_blackwell,
    pytest.mark.usefixtures("sonicmoe_setup"),
]

LIMIT = 10.0
ACT = "swiglu_limit10"


@pytest.fixture(scope="module")
def sonic_ops():
    from transformers.integrations.sonicmoe import is_sonicmoe_loadable

    from axolotl.integrations.kernels.libs.sonicmoe.lora_split import SonicOps

    if not is_sonicmoe_loadable():
        pytest.skip("sonic-moe kernel not loadable")
    return SonicOps()


@pytest.fixture(autouse=True)
def restore_flags():
    from axolotl.integrations.kernels.libs.sonicmoe import lora_split

    yield
    lora_split.configure_lora_split(False)


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6)).item()


def test_triton_glu_backward_clamped_matches_reference():
    from axolotl.integrations.kernels.libs.sonicmoe.glu_bwd_triton import glu_bwd_add_
    from axolotl.integrations.kernels.libs.sonicmoe.lora_split import TorchOps

    gen = torch.Generator(device="cuda").manual_seed(0)
    num_slots, inter = 777, 320
    cu = torch.tensor([0, 100, 100, 640], dtype=torch.int32, device="cuda")
    h = (torch.randn(num_slots, 2 * inter, device="cuda", generator=gen) * 8).bfloat16()
    # the bound itself and the next bf16 values past it, on gate and up lanes
    edge = torch.tensor([LIMIT, -LIMIT, 10.0625, -10.0625], device="cuda")
    h[:4, :8] = edge.repeat_interleave(2).bfloat16()
    dh = torch.randn(num_slots, 2 * inter, device="cuda", generator=gen).bfloat16()
    d = torch.randn(num_slots, inter, device="cuda", generator=gen).bfloat16()
    want = dh.clone()
    TorchOps().glu_bwd_add_(want, h, d, cu, ACT)
    got = dh.clone()
    glu_bwd_add_(got, h, d, cu, ACT)
    assert torch.equal(got[640:], dh[640:])
    torch.testing.assert_close(got, want, rtol=1.6e-2, atol=1e-2)
    gate, up = h[:640, ::2].float(), h[:640, 1::2].float()
    assert torch.equal(got[:640, ::2][gate >= LIMIT], dh[:640, ::2][gate >= LIMIT])
    outside = up.abs() >= LIMIT
    assert torch.equal(got[:640, 1::2][outside], dh[:640, 1::2][outside])


def _pairs(num_tokens, num_experts, top_k, gen):
    idx = torch.rand(num_tokens, num_experts, device="cuda", generator=gen)
    idx = idx.argsort(dim=1)[:, :top_k]
    weights = torch.rand(num_tokens, top_k, device="cuda", generator=gen)
    tokens = (
        torch.arange(num_tokens, device="cuda", dtype=torch.int32)
        .unsqueeze(1)
        .expand(-1, top_k)
        .reshape(-1)
    )
    return tokens, idx.reshape(-1).int(), weights.reshape(-1)


def test_clamped_epilogue_primitives_match_torch_ops(sonic_ops):
    from axolotl.integrations.kernels.libs.sonicmoe.lora_split import TorchOps

    ref = TorchOps()
    sonic_ops.prepare_activation(ACT)
    assert ACT in sonic_ops.activation.gate_fn_map
    num_tokens, num_experts, top_k, hidden, inter = 300, 8, 4, 512, 256
    gen = torch.Generator(device="cuda").manual_seed(1)
    tokens, expert_ids, scores = _pairs(num_tokens, num_experts, top_k, gen)
    meta = sonic_ops.routing_metadata(tokens, expert_ids, num_tokens, num_experts)
    cu, xg = meta.cu_seqlens, meta.x_gather_idx
    num_slots = tokens.numel()

    def rand(*shape, scale=1.0):
        return (torch.randn(*shape, device="cuda", generator=gen) * scale).bfloat16()

    x = rand(num_tokens, hidden)
    # gate/up preactivations ~ N(0, 36): the clamp is active on a few percent of them
    w1 = rand(num_experts, 2 * inter, hidden, scale=6 * hidden**-0.5).transpose(1, 2)
    outs = []
    for ops, activation in ((sonic_ops, ACT), (ref, ACT), (ref, "swiglu")):
        h = torch.empty(num_slots, 2 * inter, dtype=torch.bfloat16, device="cuda")
        a = torch.empty(num_slots, inter, dtype=torch.bfloat16, device="cuda")
        ops.gemm_gated(x, w1, None, h, a, cu, xg, activation, True)
        outs.append((h, a))
    (h_got, a_got), (h_want, a_want), (_, a_unclamped) = outs
    assert _rel(h_got, h_want) < 1e-2
    assert _rel(a_got, a_want) < 1e-2
    assert _rel(a_unclamped, a_want) > 0.05

    w2 = rand(num_experts, hidden, inter, scale=inter**-0.5)
    dout = rand(num_tokens, hidden)
    sigma = scores[meta.s_scatter_idx.long()].float()
    outs = []
    for ops in (sonic_ops, ref):
        dh = torch.empty_like(h_want)
        a_prime = torch.empty(num_slots, inter, dtype=torch.bfloat16, device="cuda")
        colvec = ops.gemm_dgated(dout, w2, h_want, dh, a_prime, sigma, cu, xg, ACT)
        outs.append((dh, a_prime, colvec))
    for got, want in zip(*outs, strict=True):
        assert _rel(got, want) < 1e-2
    dh = outs[1][0]
    gate = h_want[:, ::2].float()
    assert not dh[:, ::2][gate >= LIMIT].any()

    outs = []
    for ops in (sonic_ops, ref):
        dw1 = ops.gemm_k(x.t(), dh, torch.empty_like(w1), cu, xg, concat_out=True)
        outs.append(dw1)
    assert _rel(*outs) < 1e-2


E_LOCAL, HIDDEN, INTER, NUM_TOKENS, RANK, SCALING = 32, 4096, 2048, 2048, 16, 2.0
# gate/up preactivations ~ N(0, 25): the clamp at 10 is active on ~2% of gates, ~5% of ups
WEIGHT_SCALE = 5.0
MODELS = [
    ("deepseek_v4", "DeepseekV4Experts", 6),
    ("glm5_next", "Glm5NextTextExperts", 8),
]


def _experts(model_type, cls_name, top_k):
    from transformers import AutoConfig

    config = AutoConfig.for_model(model_type)
    config = getattr(config, "text_config", config)
    config.hidden_size = HIDDEN
    config.moe_intermediate_size = INTER
    config.num_local_experts = E_LOCAL
    config.num_experts_per_tok = top_k
    config.swiglu_limit = LIMIT
    config._experts_implementation = None
    module = importlib.import_module(
        f"transformers.models.{model_type}.modeling_{model_type}"
    )
    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.device("cuda"):
        experts = getattr(module, cls_name)(config)
    for weight, scale in (
        (experts.gate_up_proj, WEIGHT_SCALE / math.sqrt(HIDDEN)),
        (experts.down_proj, 1 / math.sqrt(INTER)),
    ):
        weight.data = torch.randn(weight.shape, device="cuda", generator=gen) * scale
    return experts.bfloat16().requires_grad_(False)


def _eager(experts, x, idx, weights, w1, w2, gate_fn=None):
    gate_fn = gate_fn or experts._apply_gate
    out = torch.zeros_like(x)
    for e in range(w1.shape[0]):
        tokens, slots = (idx == e).nonzero(as_tuple=True)
        if tokens.numel() == 0:
            continue
        y = F.linear(gate_fn(F.linear(x[tokens], w1[e])), w2[e])
        out = out.index_add(0, tokens, y * weights[tokens, slots, None])
    return out


def _plain_swiglu(gate_up):
    gate, up = gate_up.chunk(2, dim=-1)
    return F.silu(gate) * up


NAMES = {
    "full": ("out", "d_hidden", "d_router", "d_gate_up_proj", "d_down_proj"),
    "lora": ("out", "d_hidden", "d_router", "d_A1", "d_B1", "d_A2", "d_B2"),
}


@pytest.mark.parametrize(("model_type", "cls_name", "top_k"), MODELS)
@pytest.mark.parametrize("mode", ["full", "materialized", "split"])
def test_flash_layer_matches_eager_experts(model_type, cls_name, top_k, mode):
    from axolotl.integrations.kernels.libs.sonicmoe import experts as sonic, lora_split
    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        register_sonicmoe_experts,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.lora import MoELoRAMaterialize

    register_sonicmoe_experts()
    experts = _experts(model_type, cls_name, top_k)
    gen = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(NUM_TOKENS, HIDDEN, device="cuda", generator=gen).bfloat16()
    logits = torch.rand(NUM_TOKENS, E_LOCAL, device="cuda", generator=gen)
    idx = logits.argsort(dim=1)[:, :top_k]
    weights = torch.rand(NUM_TOKENS, top_k, device="cuda", generator=gen).bfloat16()
    full = mode == "full"
    lora = None
    if not full:
        lora = [
            (torch.randn(shape, device="cuda", generator=gen) * 0.02).bfloat16()
            for shape in (
                (RANK * E_LOCAL, HIDDEN),
                (2 * INTER, RANK * E_LOCAL),
                (RANK * E_LOCAL, INTER),
                (HIDDEN, RANK * E_LOCAL),
            )
        ]

    def leaves(dtype):
        base = [
            experts.gate_up_proj.detach().to(dtype),
            experts.down_proj.detach().to(dtype),
        ]
        extra = base if full else [t.to(dtype) for t in lora]
        return [
            t.clone().requires_grad_(True)
            for t in (x.to(dtype), weights.to(dtype), *extra)
        ]

    def eager(dtype, gate_fn=None):
        xx, ww, *params = leaves(dtype)
        if full:
            w1, w2 = params
        else:
            w1 = MoELoRAMaterialize.apply(
                experts.gate_up_proj.detach().to(dtype), *params[:2], SCALING
            )
            w2 = MoELoRAMaterialize.apply(
                experts.down_proj.detach().to(dtype), *params[2:], SCALING
            )
        out = _eager(experts, xx, idx, ww, w1, w2, gate_fn)
        grads = torch.autograd.grad(out.float().square().sum(), [xx, ww, *params])
        return [out.detach(), *grads]

    def quack():
        xx, ww, *params = leaves(torch.bfloat16)
        lora_split.configure_lora_split(mode == "split")
        if full:
            experts.gate_up_proj, experts.down_proj = (
                torch.nn.Parameter(p) for p in params
            )
        else:
            experts._sonicmoe_lora = {
                "gate_up_proj": (*params[:2], SCALING),
                "down_proj": (*params[2:], SCALING),
            }
        try:
            out = sonic._sonicmoe_forward(experts, xx, idx, ww)
        finally:
            experts.__dict__.pop("_sonicmoe_lora", None)
        params = [experts.gate_up_proj, experts.down_proj] if full else params
        grads = torch.autograd.grad(out.float().square().sum(), [xx, ww, *params])
        return [out.detach(), *grads]

    want = eager(torch.float32)
    unclamped = eager(torch.float32, _plain_swiglu)
    yardstick = eager(torch.bfloat16)
    got = quack()

    assert _rel(unclamped[0], want[0]) > 0.05
    names = NAMES["full" if full else "lora"]
    for name, g, w, y in zip(names, got, want, yardstick, strict=True):
        assert torch.isfinite(g).all(), name
        err, eager_err = _rel(g, w), _rel(y, w)
        print(f"{model_type} {mode} {name}: quack {err:.2e} eager bf16 {eager_err:.2e}")
        assert err < max(2e-2, 1.5 * eager_err), name
