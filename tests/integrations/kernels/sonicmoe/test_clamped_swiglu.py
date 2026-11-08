# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""CPU checks for SonicMoE's clamped-SwiGLU experts (DeepSeek-V4 ``limit``, GLM-5 ``swiglu_limit``).

The reference is the eager experts module's own ``_apply_gate``. The quack-path autograd
(``lora_split.LoRASplitExperts`` on ``TorchOps``) runs in float64, so its output and every
gradient must match the eager module to float64 precision for full fine-tuning (trainable base
weights), materialized LoRA and the LoRA split, with the clamp active on a large share of
gate/up entries and sentinel pairs, an idle expert and a token with no local expert in the
routing.
"""

import importlib
import math

import pytest
import torch
import torch.nn.functional as F
from transformers import AutoConfig

from axolotl.integrations.kernels.libs.sonicmoe import experts as sonic
from axolotl.integrations.kernels.libs.sonicmoe.epilogue import (
    declared_limit,
    dense_limit,
)
from axolotl.integrations.kernels.libs.sonicmoe.lora import MoELoRAMaterialize
from axolotl.integrations.kernels.libs.sonicmoe.lora_split import (
    TorchOps,
    glu_backward,
    glu_epilogue,
    glu_forward,
    glu_limit,
    sonicmoe_lora_split_forward,
)

E, K, H, INTER, T, RANK = 5, 3, 16, 8, 14, 3
IDLE_EXPERT, NO_LOCAL_TOKEN = 3, 6
LIMIT = 1.0
S1, S2 = 0.75, 1.5

MODELS = [
    ("deepseek_v4", "DeepseekV4Experts"),
    ("glm5_next", "Glm5NextTextExperts"),
]


def _experts(model_type, cls_name, gen, limit=LIMIT):
    config = AutoConfig.for_model(model_type)
    config = getattr(config, "text_config", config)
    config.hidden_size = H
    config.moe_intermediate_size = INTER
    config.num_local_experts = E
    config.num_experts_per_tok = K
    config.swiglu_limit = limit
    config._experts_implementation = None
    module = importlib.import_module(
        f"transformers.models.{model_type}.modeling_{model_type}"
    )
    experts = getattr(module, cls_name)(config)
    experts.gate_up_proj = torch.nn.Parameter(
        torch.randn(E, 2 * INTER, H, generator=gen, dtype=torch.float64) / math.sqrt(H),
        requires_grad=False,
    )
    experts.down_proj = torch.nn.Parameter(
        torch.randn(E, H, INTER, generator=gen, dtype=torch.float64) / math.sqrt(INTER),
        requires_grad=False,
    )
    return experts


def _routing(gen):
    idx = torch.randint(0, E - 1, (T, K), generator=gen)
    idx = torch.where(idx >= IDLE_EXPERT, idx + 1, idx)
    sentinel = torch.rand(T, K, generator=gen) < 0.3
    sentinel[NO_LOCAL_TOKEN] = True
    idx = idx.masked_fill(sentinel, E)
    weights = torch.rand(T, K, generator=gen, dtype=torch.float64) + 0.1
    x = torch.randn(T, H, generator=gen, dtype=torch.float64)
    return x, idx, weights


def _eager(experts, x, idx, weights, w1, w2):
    """The eager experts loop with ``w1`` / ``w2`` in place of the module's own weights."""
    out = torch.zeros_like(x)
    for e in range(E):
        tokens, slots = (idx == e).nonzero(as_tuple=True)
        if tokens.numel() == 0:
            continue
        a = experts._apply_gate(F.linear(x[tokens], w1[e]))
        y = F.linear(a, w2[e]) * weights[tokens, slots, None]
        out = out.index_add(0, tokens, y)
    return out


def _lora_pair(weight, gen):
    _, dim1, dim2 = weight.shape
    a = torch.randn(RANK * E, dim2, generator=gen, dtype=torch.float64) * 0.3
    b = torch.randn(dim1, RANK * E, generator=gen, dtype=torch.float64) * 0.3
    return a.requires_grad_(True), b.requires_grad_(True)


def test_activation_names_round_trip_the_limit():
    assert glu_epilogue("silu") == "swiglu"
    assert glu_epilogue("gelu") == "geglu"
    assert glu_epilogue("silu", 10.0) == "swiglu_limit10"
    assert glu_epilogue("silu", 7.5) == "swiglu_limit7.5"
    assert glu_limit("swiglu_limit10") == 10.0
    assert glu_limit("swiglu_limit7.5") == 7.5
    assert glu_limit("swiglu") is None
    assert glu_epilogue("gelu", 10.0) is None
    assert glu_epilogue("relu", 10.0) is None


@pytest.mark.parametrize(("model_type", "cls_name"), MODELS)
def test_glu_matches_apply_gate_at_the_clamp_boundaries(model_type, cls_name):
    gen = torch.Generator().manual_seed(0)
    experts = _experts(model_type, cls_name, gen)
    activation = glu_epilogue("silu", dense_limit(experts, "silu"))
    assert glu_limit(activation) == LIMIT

    inf = torch.tensor(math.inf, dtype=torch.float64)
    bound = torch.tensor(LIMIT, dtype=torch.float64)
    above, below = torch.nextafter(bound, inf), torch.nextafter(-bound, -inf)
    gate = torch.randn(32, INTER, generator=gen, dtype=torch.float64) * 3 * LIMIT
    up = torch.randn(32, INTER, generator=gen, dtype=torch.float64) * 3 * LIMIT
    gate[0, :3] = torch.stack([bound, above, -bound])
    up[0, :4] = torch.stack([bound, above, -bound, below])
    gate_up = torch.cat([gate, up], dim=-1).requires_grad_(True)
    want = experts._apply_gate(gate_up)
    d = torch.randn(want.shape, generator=gen, dtype=torch.float64)
    (want_grad,) = torch.autograd.grad(want, gate_up, d)

    torch.testing.assert_close(glu_forward(gate, up, activation), want.detach())
    dgate, dup = glu_backward(gate, up, d, activation)
    torch.testing.assert_close(torch.cat([dgate, dup], dim=-1), want_grad)
    # torch.clamp's convention: no gradient at the bound or past it
    assert dgate[0, 0] == 0 and dgate[0, 1] == 0 and dgate[0, 2] != 0
    assert (dup[0, :4] == 0).all()
    assert (dgate[gate >= LIMIT] == 0).all() and (dup[up.abs() >= LIMIT] == 0).all()
    assert (dgate[gate < LIMIT] != 0).all() and (dup[up.abs() < LIMIT] != 0).all()


def test_torch_ops_glu_bwd_add_applies_the_clamp():
    gen = torch.Generator().manual_seed(1)
    activation = glu_epilogue("silu", LIMIT)
    h = torch.randn(20, 2 * INTER, generator=gen, dtype=torch.float64) * 3 * LIMIT
    d = torch.randn(20, INTER, generator=gen, dtype=torch.float64)
    dh = torch.randn(20, 2 * INTER, generator=gen, dtype=torch.float64)
    cu = torch.tensor([0, 9, 9, 16], dtype=torch.int32)
    got = dh.clone()
    TorchOps().glu_bwd_add_(got, h, d, cu, activation)
    dgate, dup = glu_backward(h[:16, ::2], h[:16, 1::2], d[:16], activation)
    torch.testing.assert_close(got[:16, ::2], dh[:16, ::2] + dgate)
    torch.testing.assert_close(got[:16, 1::2], dh[:16, 1::2] + dup)
    assert torch.equal(got[16:], dh[16:])


@pytest.mark.parametrize(("model_type", "cls_name"), MODELS)
@pytest.mark.parametrize("mode", ["full", "materialized", "split"])
def test_quack_path_matches_eager_experts(model_type, cls_name, mode):
    gen = torch.Generator().manual_seed(2)
    experts = _experts(model_type, cls_name, gen)
    x, idx, weights = _routing(gen)
    base1, base2 = experts.gate_up_proj, experts.down_proj
    full = mode == "full"
    base1.requires_grad_(full)
    base2.requires_grad_(full)
    lora1 = lora2 = None
    if not full:
        lora1, lora2 = _lora_pair(base1, gen), _lora_pair(base2, gen)
    leaves = [base1, base2] if full else [*lora1, *lora2]
    grad_out = torch.randn(T, H, generator=gen, dtype=torch.float64)

    h = torch.einsum("th,eoh->teo", x, base1)
    assert (h[..., :INTER] > LIMIT).double().mean() > 0.1
    assert (h[..., INTER:].abs() > LIMIT).double().mean() > 0.2

    def effective():
        if full:
            return base1, base2
        return (
            MoELoRAMaterialize.apply(base1, *lora1, S1),
            MoELoRAMaterialize.apply(base2, *lora2, S2),
        )

    def run(fn):
        xx = x.clone().requires_grad_(True)
        ww = weights.clone().requires_grad_(True)
        out = fn(xx, ww)
        grads = torch.autograd.grad((out * grad_out).sum(), [xx, ww, *leaves])
        return out.detach(), grads

    def quack(xx, ww):
        split = mode == "split"
        w1, w2 = (base1, base2) if split else effective()
        facade = sonic._LoRAExpertsFacade(experts, w1, None, w2, None)
        assert facade.limit == LIMIT
        pairs = sonic._grid_pairs(xx, idx, ww)
        loras = ((*lora1, S1), (*lora2, S2)) if split else (None, None)
        return sonicmoe_lora_split_forward(facade, xx, *pairs, *loras, ops=TorchOps())

    want_out, want = run(lambda xx, ww: _eager(experts, xx, idx, ww, *effective()))
    got_out, got = run(quack)

    torch.testing.assert_close(got_out, want_out, rtol=1e-10, atol=1e-10)
    for g, w in zip(got, want, strict=True):
        assert torch.isfinite(g).all()
        torch.testing.assert_close(g, w, rtol=1e-9, atol=1e-10)
    assert not got_out[NO_LOCAL_TOKEN].any() and not got[0][NO_LOCAL_TOKEN].any()
    assert not got[1][idx == E].any()
    if full:
        assert not got[2][IDLE_EXPERT].any() and not got[3][IDLE_EXPERT].any()


@pytest.mark.parametrize(("model_type", "cls_name"), MODELS)
def test_clamped_forward_runs_bf16_experts(model_type, cls_name):
    """The routing glue around the autograd, in bf16 against the module's own fp32 forward."""
    gen = torch.Generator().manual_seed(6)
    experts = _experts(model_type, cls_name, gen)
    x, idx, weights = _routing(gen)
    want = experts.float()(x.float(), idx, weights.float())
    experts = experts.to(torch.bfloat16)
    facade = sonic._LoRAExpertsFacade(
        experts, experts.gate_up_proj, None, experts.down_proj, None
    )
    got = sonic._sonicmoe_clamped_forward(
        experts,
        facade,
        x.bfloat16(),
        idx,
        weights.bfloat16(),
        ops=TorchOps(),
    )
    assert got.dtype == torch.bfloat16
    rel = (got.float() - want).norm() / want.norm()
    assert rel < 2e-2


def test_full_finetune_matches_the_module_forward():
    """The eager reference above is the module's own loop: its forward agrees with it."""
    gen = torch.Generator().manual_seed(3)
    experts = _experts("deepseek_v4", "DeepseekV4Experts", gen)
    x, idx, weights = _routing(gen)
    torch.testing.assert_close(
        experts(x, idx, weights),
        _eager(experts, x, idx, weights, experts.gate_up_proj, experts.down_proj),
    )


@pytest.mark.parametrize(
    ("bias", "dtype", "reason"),
    [
        (True, torch.bfloat16, "expert bias"),
        (False, torch.float16, "non-bf16 experts"),
    ],
)
def test_unsupported_clamped_calls_raise(bias, dtype, reason):
    """No fallback runs a clamped epilogue on SonicMoE, so an unsupported call must raise."""
    gen = torch.Generator().manual_seed(4)
    experts = _experts("deepseek_v4", "DeepseekV4Experts", gen).to(dtype)
    x, idx, weights = _routing(gen)
    b1 = torch.zeros(E, 2 * INTER, dtype=dtype) if bias else None
    facade = sonic._LoRAExpertsFacade(
        experts, experts.gate_up_proj, b1, experts.down_proj, None
    )
    with pytest.raises(NotImplementedError, match=reason):
        sonic._sonicmoe_clamped_forward(
            experts, facade, x.to(dtype), idx, weights.to(dtype), ops=TorchOps()
        )


def test_model_epilogues_registered_before_the_first_forward():
    from axolotl.integrations.kernels.libs.sonicmoe.lora_split import (
        prepare_model_epilogues,
    )

    gen = torch.Generator().manual_seed(7)
    model = torch.nn.ModuleList(
        [
            _experts("deepseek_v4", "DeepseekV4Experts", gen),
            _experts("glm5_next", "Glm5NextTextExperts", gen, limit=7.5),
        ]
    )
    prepared = []
    ops = type("Ops", (), {"prepare_activation": lambda self, a: prepared.append(a)})()
    assert prepare_model_epilogues(model, ops=ops) == [
        "swiglu_limit1",
        "swiglu_limit7.5",
    ]
    assert prepared == ["swiglu_limit1", "swiglu_limit7.5"]
    assert prepare_model_epilogues(torch.nn.Linear(2, 2)) == []


def test_declared_limit_reads_both_spellings_and_skips_unclamped():
    gen = torch.Generator().manual_seed(5)
    assert declared_limit(_experts("deepseek_v4", "DeepseekV4Experts", gen)) == LIMIT
    assert declared_limit(_experts("glm5_next", "Glm5NextTextExperts", gen)) == LIMIT

    class Unclamped(torch.nn.Module):
        limit = math.inf

    assert declared_limit(Unclamped()) is None
    assert (
        dense_limit(_experts("deepseek_v4", "DeepseekV4Experts", gen), "gelu") is None
    )
