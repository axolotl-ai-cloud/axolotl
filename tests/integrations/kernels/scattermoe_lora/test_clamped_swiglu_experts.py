# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""ScatterMoE on clamped-SwiGLU experts: DeepSeek-V4 (``limit``, ``act_fn``) and GLM-5.3-Flash
(``swiglu_limit``, ``F.silu`` inline, no ``act_fn``).

CPU: the dense gated epilogue is the experts' own ``_apply_gate``, and the fused paths resolve the
activation and clamp without ``act_fn``. CUDA: the ScatterMoE forward, full fine-tune and fused
LoRA, against the module's eager loop in fp32 with the clamp active on a few percent of gate/up
entries; bf16 rounding at the bound flips some of those, so each error is held to that of the
eager module in bf16.
"""

import importlib
import math
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton")

LIMIT = 10.0
MODELS = [
    ("deepseek_v4", "DeepseekV4Experts"),
    ("glm5_next", "Glm5NextTextExperts"),
]


def _experts(model_type, cls_name, *, num_experts, hidden, inter, top_k):
    from transformers import AutoConfig

    config = AutoConfig.for_model(model_type)
    config = getattr(config, "text_config", config)
    config.hidden_size = hidden
    config.moe_intermediate_size = inter
    config.num_local_experts = num_experts
    config.num_experts_per_tok = top_k
    config.swiglu_limit = LIMIT
    config._experts_implementation = None
    module = importlib.import_module(
        f"transformers.models.{model_type}.modeling_{model_type}"
    )
    return getattr(module, cls_name)(config)


@pytest.mark.parametrize(("model_type", "cls_name"), MODELS)
def test_fused_paths_resolve_activation_and_clamp(model_type, cls_name):
    from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
        _detect_act_type,
        _expert_limit,
    )

    with torch.device("meta"):
        experts = _experts(
            model_type, cls_name, num_experts=4, hidden=16, inter=8, top_k=2
        )
    assert _detect_act_type(experts, default=None) == "silu"
    assert _expert_limit(experts) == LIMIT


def test_activation_falls_back_to_the_config():
    from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
        _detect_act_type,
    )

    geglu = SimpleNamespace(
        config=SimpleNamespace(hidden_activation="gelu_pytorch_tanh")
    )
    assert _detect_act_type(geglu) == "gelu_tanh"
    unknown = SimpleNamespace(config=SimpleNamespace(hidden_act="quadratic"))
    assert _detect_act_type(unknown, default=None) is None
    assert _detect_act_type(SimpleNamespace(), default=None) is None


@pytest.mark.parametrize(("model_type", "cls_name"), MODELS)
def test_gated_epilogue_is_the_declared_apply_gate(model_type, cls_name):
    from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
        _gated_epilogue,
    )

    experts = _experts(model_type, cls_name, num_experts=4, hidden=16, inter=8, top_k=2)
    gen = torch.Generator().manual_seed(0)
    gates_h = torch.randn(32, 16, generator=gen, dtype=torch.float64) * 3 * LIMIT
    want = experts._apply_gate(gates_h)
    assert torch.equal(_gated_epilogue(experts, gates_h), want)
    gate, up = gates_h.chunk(2, dim=-1)
    assert not torch.allclose(F.silu(gate) * up, want)


def test_gated_epilogue_without_apply_gate_clamps_swiglu_limit():
    from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
        _gated_epilogue,
    )

    module = SimpleNamespace(act_fn=F.silu, swiglu_limit=LIMIT)
    gen = torch.Generator().manual_seed(1)
    gates_h = torch.randn(32, 16, generator=gen, dtype=torch.float64) * 3 * LIMIT
    gate, up = gates_h.chunk(2, dim=-1)
    want = F.silu(gate.clamp(max=LIMIT)) * up.clamp(-LIMIT, LIMIT)
    torch.testing.assert_close(_gated_epilogue(module, gates_h), want)


E, HIDDEN, INTER, TOP_K, NUM_TOKENS, RANK, SCALING = 8, 512, 256, 4, 512, 8, 2.0
# gate/up preactivations ~ N(0, 25): the clamp at 10 is active on ~2% of gates, ~5% of ups
WEIGHT_SCALE = 5.0


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp_min(1e-6)).item()


def _eager(experts, x, idx, weights, w1, w2, gate_fn):
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


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize(("model_type", "cls_name"), MODELS)
@pytest.mark.parametrize("mode", ["full", "lora"])
def test_scattermoe_matches_eager_experts(model_type, cls_name, mode):
    from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
        scattermoe_experts_forward,
    )
    from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
        _convert_smoe_lora,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.lora import MoELoRAMaterialize

    gen = torch.Generator(device="cuda").manual_seed(0)
    with torch.device("cuda"):
        experts = _experts(
            model_type,
            cls_name,
            num_experts=E,
            hidden=HIDDEN,
            inter=INTER,
            top_k=TOP_K,
        )
    for weight, scale in (
        (experts.gate_up_proj, WEIGHT_SCALE / math.sqrt(HIDDEN)),
        (experts.down_proj, 1 / math.sqrt(INTER)),
    ):
        weight.data = torch.randn(weight.shape, device="cuda", generator=gen) * scale
    experts = experts.bfloat16().requires_grad_(False)
    base = (experts.gate_up_proj.detach(), experts.down_proj.detach())

    x = torch.randn(NUM_TOKENS, HIDDEN, device="cuda", generator=gen).bfloat16()
    idx = torch.rand(NUM_TOKENS, E, device="cuda", generator=gen).argsort(dim=1)
    idx = idx[:, :TOP_K]
    weights = torch.rand(NUM_TOKENS, TOP_K, device="cuda", generator=gen).bfloat16()
    full = mode == "full"
    lora = [
        (torch.randn(shape, device="cuda", generator=gen) * 0.02).bfloat16()
        for shape in (
            (RANK * E, HIDDEN),
            (2 * INTER, RANK * E),
            (RANK * E, INTER),
            (HIDDEN, RANK * E),
        )
    ]

    def leaves(dtype):
        extra = base if full else lora
        return [t.to(dtype).clone().requires_grad_(True) for t in (x, weights, *extra)]

    def eager(dtype, gate_fn):
        xx, ww, *params = leaves(dtype)
        if full:
            w1, w2 = params
        else:
            w1 = MoELoRAMaterialize.apply(base[0].to(dtype), *params[:2], SCALING)
            w2 = MoELoRAMaterialize.apply(base[1].to(dtype), *params[2:], SCALING)
        out = _eager(experts, xx, idx, ww, w1, w2, gate_fn)
        grads = torch.autograd.grad(out.float().square().sum(), [xx, ww, *params])
        return [out.detach(), *grads]

    def scattermoe():
        xx, ww, *params = leaves(torch.bfloat16)
        if full:
            experts.gate_up_proj, experts.down_proj = (
                torch.nn.Parameter(p) for p in params
            )
        else:
            experts._scattermoe_lora = {
                "gate_up_proj": _convert_smoe_lora(*params[:2], E, RANK, SCALING),
                "down_proj": _convert_smoe_lora(*params[2:], E, RANK, SCALING),
            }
        try:
            out = scattermoe_experts_forward(experts, xx, idx, ww)
        finally:
            experts.__dict__.pop("_scattermoe_lora", None)
        params = [experts.gate_up_proj, experts.down_proj] if full else params
        grads = torch.autograd.grad(out.float().square().sum(), [xx, ww, *params])
        return [out.detach(), *grads]

    want = eager(torch.float32, experts._apply_gate)
    unclamped = eager(torch.float32, _plain_swiglu)
    yardstick = eager(torch.bfloat16, experts._apply_gate)
    got = scattermoe()

    assert _rel(unclamped[0], want[0]) > 0.05
    names = ["out", "d_hidden", "d_router"]
    names += (
        ["d_gate_up_proj", "d_down_proj"] if full else ["d_A1", "d_B1", "d_A2", "d_B2"]
    )
    for name, g, w, y in zip(names, got, want, yardstick, strict=True):
        assert torch.isfinite(g).all(), name
        err, eager_err = _rel(g, w), _rel(y, w)
        print(
            f"{model_type} {mode} {name}: scattermoe {err:.2e} eager bf16 {eager_err:.2e}"
        )
        assert err < max(2e-2, 1.5 * eager_err), name
