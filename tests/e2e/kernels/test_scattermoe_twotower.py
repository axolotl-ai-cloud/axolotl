# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""ScatterMoE on the Nemotron-Labs-TwoTower expert layout (GPU).

TwoTower's MoE layers are Nemotron-3-Nano's: 128 non-gated relu² experts, top-6,
sigmoid router with correction bias and ``routed_scaling_factor`` 2.5, plus one
shared expert. These tests shrink the widths but keep that structure and check,
forward and backward:

* the MoE layer through scattermoe vs eager vs a per-token naive oracle;
* the whole two-tower denoiser (both towers share one config, so one
  ``set_experts_implementation`` call must reach every MoE layer);
* PEFT ``target_parameters`` LoRA on the denoiser experts (scoped as the
  model-support hook scopes it) through the fused scattermoe LoRA path.
"""

import contextlib

import pytest
import torch

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required"),
]


def _register():
    from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
        register_scattermoe_experts,
    )

    register_scattermoe_experts()


def _rel(a, b):
    return (a.float() - b.float()).abs().max().item() / max(
        b.float().abs().max().item(), 1e-8
    )


@contextlib.contextmanager
def _same_routing(model, record):
    """Record every MoE router's top-k choices into an empty ``record``, else replay them.

    In bf16, last-bit differences between expert implementations flip a few
    near-tied top-6 choices downstream, which moves those tokens' logits by O(1)
    and says nothing about the kernel. Replaying the reference pass's choice
    compares the implementations on identical routing; the weights are recomputed
    from the live router logits (as ``NemotronHTopkRouter`` does) so router
    gradients still flow.
    """

    replay = bool(record)
    calls: dict[str, int] = {}

    def hook(name, module, _inputs, output):
        # routers run once per call (the context tower once per document)
        router_logits, _, indices = output
        if not replay:
            record.setdefault(name, []).append(indices.detach())
            return None
        call = calls.get(name, 0)
        calls[name] = call + 1
        indices = record[name][call]
        weights = router_logits.sigmoid().gather(1, indices)
        if module.norm_topk_prob:
            weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)
        return router_logits, weights * module.routed_scaling_factor, indices

    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHTopkRouter

    handles = [
        module.register_forward_hook(lambda *a, name=name: hook(name, *a))
        for name, module in model.named_modules()
        if isinstance(module, NemotronHTopkRouter)
    ]
    assert handles
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
    if replay:
        assert calls == {name: len(seen) for name, seen in record.items()}


def _model(dtype=torch.bfloat16, seed=0):
    from tests.model_support.test_nemotron_twotower import tiny_model

    model = tiny_model(
        seed,
        hidden_size=256,
        n_routed_experts=16,
        num_experts_per_tok=6,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        mamba_num_heads=8,
        mamba_head_dim=32,
        n_groups=2,
        ssm_state_size=32,
        chunk_size=64,
    )
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if ".experts." in name or name.endswith((".up_proj", ".down_proj")):
                parameter.normal_(0, 0.05)
    return model.to("cuda", dtype)


def _batch():
    torch.manual_seed(11)
    lengths = ((70, 32), (45, 32), (90, 32))
    ids, positions, canvas = [], [], []
    for prompt, width in lengths:
        tokens = torch.randint(10, 257, (prompt + width,))
        tokens[prompt + 2 : prompt + 12] = 3
        ids.append(tokens)
        positions.append(torch.arange(prompt + width))
        canvas.append(torch.cat([torch.zeros(prompt), torch.ones(width)]).bool())
    return (
        torch.cat(ids)[None].cuda(),
        torch.cat(positions)[None].cuda(),
        torch.cat(canvas)[None].cuda(),
    )


def test_moe_layer_matches_eager_and_oracle():
    from tests.model_support.test_nemotron_twotower import _oracle_moe

    _register()
    model = _model(torch.float32)
    moe = model.denoiser_tower.layers[1].mixer
    x = torch.randn(3, 40, model.config.hidden_size, device="cuda") * 0.5

    def run(impl):
        model.config._experts_implementation = impl
        moe.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        out = moe(xi)
        out.square().mean().backward()
        return (
            out.detach(),
            xi.grad,
            moe.experts.up_proj.grad,
            moe.experts.down_proj.grad,
        )

    eager = run("eager")
    scatter = run("scattermoe")
    for name, ref, got in zip(
        ("out", "dx", "d_up", "d_down"), eager, scatter, strict=True
    ):
        # tf32 tensor-core accumulation in the triton kernels
        assert _rel(got, ref) < 1e-2, f"{name}: rel={_rel(got, ref):.3e}"
    with torch.no_grad():
        oracle = _oracle_moe(moe, x, model.config)
    assert _rel(scatter[0], oracle) < 1e-2


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_two_tower_forward_backward_matches_eager(dtype):
    _register()
    model = _model(dtype).train()
    ids, positions, canvas = _batch()
    routing = {}

    def run(impl):
        model.set_experts_implementation(impl)
        assert all(
            layer.mixer.experts.config._experts_implementation == impl
            for tower in (model.context_tower, model.denoiser_tower)
            for layer in tower.layers
            if hasattr(layer.mixer, "experts")
        )
        model.zero_grad(set_to_none=True)
        with _same_routing(model, routing):
            out = model(ids, position_ids=positions, axolotl_canvas_mask=canvas)
        loss = out.logits[canvas].float().logsumexp(-1).mean()
        loss.backward()
        grads = {
            n: p.grad.float().clone()
            for n, p in model.named_parameters()
            if p.grad is not None
        }
        return out.logits[canvas].detach().float(), grads

    logits_ref, grads_ref = run("eager")
    logits, grads = run("scattermoe")
    tolerance = 5e-2 if dtype is torch.bfloat16 else 1e-2
    assert _rel(logits, logits_ref) < tolerance
    assert grads.keys() == grads_ref.keys()
    assert not any(name.startswith("context") for name in grads)
    for name, ref in grads_ref.items():
        diff = (grads[name] - ref).abs().max().item()
        rel = diff / (ref.abs().max().item() + 1e-8)
        assert not (rel > 0.1 and diff > 1e-2), f"{name}: rel={rel:.3f}"


def test_expert_lora_through_scattermoe_matches_eager():
    from peft import LoraConfig, get_peft_model

    from axolotl.model_support.nemotron_twotower import (
        scope_target_modules,
        scope_target_parameters,
    )

    _register()
    base = _model(torch.bfloat16)
    peft_model = get_peft_model(
        base,
        LoraConfig(
            r=8,
            lora_alpha=8,
            use_rslora=True,
            lora_dropout=0.0,
            target_modules=scope_target_modules(["q_proj", "in_proj", "out_proj"]),
            target_parameters=scope_target_parameters(
                ["mixer.experts.up_proj", "mixer.experts.down_proj"], base.config
            ),
        ),
    ).cuda()
    with torch.no_grad():
        # non-zero B so the expert LoRA delta is active in the forward
        for name, parameter in peft_model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.02)
    expert_lora = [
        n
        for n, p in peft_model.named_parameters()
        if p.requires_grad and ".experts." in n
    ]
    assert expert_lora and all("denoiser_tower" in n for n in expert_lora)
    model = peft_model.base_model.model
    ids, positions, canvas = _batch()
    routing = {}

    def run(impl):
        model.set_experts_implementation(impl)
        peft_model.zero_grad(set_to_none=True)
        with _same_routing(model, routing):
            out = peft_model(ids, position_ids=positions, axolotl_canvas_mask=canvas)
        out.logits[canvas].float().logsumexp(-1).mean().backward()
        grads = {
            n: p.grad.float().clone()
            for n, p in peft_model.named_parameters()
            if p.grad is not None
        }
        return out.logits[canvas].detach().float(), grads

    logits_ref, grads_ref = run("eager")
    logits, grads = run("scattermoe")
    assert _rel(logits, logits_ref) < 5e-2
    assert grads.keys() == grads_ref.keys()
    assert any(".experts." in name for name in grads)
    for name, ref in grads_ref.items():
        diff = (grads[name] - ref).abs().max().item()
        rel = diff / (ref.abs().max().item() + 1e-8)
        assert not (rel > 0.1 and diff > 1e-2), f"{name}: rel={rel:.3f}"


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs two GPUs")
def test_towers_on_two_gpus_match_one_gpu():
    """The 30B-A3B release needs ~63 GB per bf16 tower, so real runs split the
    frozen context tower and the denoiser across two GPUs."""
    _register()
    model = _model(torch.float32).eval()
    model.set_experts_implementation("scattermoe")
    ids, positions, canvas = _batch()
    with torch.no_grad():
        single = model(ids, position_ids=positions, axolotl_canvas_mask=canvas).logits
        model.place_towers_on_devices("cuda:0", "cuda:1")
        split = model(ids, position_ids=positions, axolotl_canvas_mask=canvas).logits
    # logits come back on the batch's device for the trainer's loss tensors
    assert split.device == ids.device
    assert _rel(split[canvas], single[canvas]) < 1e-5
