# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Nemotron-3 latentmoe (non-gated relu² experts) through the MoE kernels.

Parity vs the eager NemotronH experts at three depths:
  - experts module (fwd + full-param grads), latent and non-latent widths
  - single-MoE-layer causal LM ("*E" hybrid pattern), full-param and PEFT
    ``target_parameters`` LoRA
  - NVFP4 (modelopt-style block-16) base + LoRA via the grouped fp4 (scattermoe)
    and grouped dequant (sonicmoe) paths

The latent geometry is what makes this arch distinct: tokens reach the experts
already projected to ``moe_latent_size`` (¼ of hidden), so the expert weights
are [E, I, L] / [E, L, I] with L != hidden_size and the router still scores in
full hidden width.
"""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _register_all():
    from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
        register_scattermoe_experts,
    )
    from axolotl.integrations.kernels.libs.sonicmoe.experts import (
        register_sonicmoe_experts,
    )

    register_scattermoe_experts()
    register_sonicmoe_experts()


def _nemotron_cfg(latent, **overrides):
    from transformers.models.nemotron_h.configuration_nemotron_h import NemotronHConfig

    kw = dict(
        hidden_size=256,
        moe_latent_size=latent,
        intermediate_size=512,
        moe_intermediate_size=128,
        moe_shared_expert_intermediate_size=192,
        n_routed_experts=16,
        num_experts_per_tok=4,
        n_shared_experts=1,
        norm_topk_prob=True,
        routed_scaling_factor=2.5,
        num_hidden_layers=2,
        hybrid_override_pattern="*E",
        num_attention_heads=8,
        num_key_value_heads=2,
        attention_head_dim=32,
        ssm_state_size=16,
        vocab_size=512,
        max_position_embeddings=256,
        use_cache=False,
    )
    kw.update(overrides)
    cfg = NemotronHConfig(**kw)
    cfg._experts_implementation = "eager"
    return cfg


def _rel(a, b):
    return (a - b).float().abs().max().item() / max(b.float().abs().max().item(), 1e-8)


def _experts_run(experts, cfg, impl, x, idx, wts):
    cfg._experts_implementation = impl
    experts.zero_grad(set_to_none=True)
    xi = x.clone().requires_grad_(True)
    out = experts(xi, idx, wts)
    out.float().square().mean().backward()
    return (
        out.detach(),
        xi.grad,
        experts.up_proj.grad.clone(),
        experts.down_proj.grad.clone(),
    )


@pytest.mark.parametrize("latent", [64, None], ids=["latent", "no-latent"])
@pytest.mark.parametrize("impl", ["scattermoe", "sonicmoe"])
def test_experts_module_parity(impl, latent):
    """Non-gated experts fwd + full-param grads match eager."""
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

    _register_all()
    torch.manual_seed(0)
    cfg = _nemotron_cfg(latent)
    experts = NemotronHExperts(cfg).to("cuda", torch.bfloat16)
    with torch.no_grad():
        experts.up_proj.normal_(0, 0.05)
        experts.down_proj.normal_(0, 0.05)

    width = latent or cfg.hidden_size
    T, K = 128, cfg.num_experts_per_tok
    x = torch.randn(T, width, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, cfg.n_routed_experts, (T, K), device="cuda")
    wts = torch.softmax(torch.randn(T, K, device="cuda"), -1).to(torch.bfloat16)

    ref = _experts_run(experts, cfg, "eager", x, idx, wts)
    got = _experts_run(experts, cfg, impl, x, idx, wts)
    for name, r, g in zip(("out", "dx", "d_up", "d_down"), ref, got, strict=True):
        assert _rel(g, r) < 5e-2, f"{impl}/{name}: rel={_rel(g, r):.4e}"


def _model_compare(model, cfg, impls, grad_filter=None):
    input_ids = torch.randint(0, 512, (2, 32), device="cuda")

    def run(impl):
        cfg._experts_implementation = impl
        model.zero_grad(set_to_none=True)
        out = model(input_ids, labels=input_ids)
        out.loss.backward()
        grads = {
            n: p.grad.float().clone()
            for n, p in model.named_parameters()
            if p.grad is not None and (grad_filter is None or grad_filter(n))
        }
        return out.logits.detach().float(), out.loss.item(), grads

    logits_ref, loss_ref, grads_ref = run("eager")
    for impl in impls:
        logits, loss, grads = run(impl)
        assert _rel(logits, logits_ref) < 5e-2, f"{impl}: logits diverged"
        assert abs(loss - loss_ref) < 0.05, f"{impl}: loss {loss} vs {loss_ref}"
        assert set(grads) == set(grads_ref), f"{impl}: grad params differ"
        for n, gr in grads_ref.items():
            g = grads[n]
            md = (gr - g).abs().max().item()
            rel = md / (gr.abs().max().item() + 1e-8)
            assert not (rel > 0.1 and md > 1e-2), f"{impl}/{n}: rel={rel:.3f}"


def test_single_layer_model_full_param():
    """1 attention + 1 latent-MoE layer: logits/loss/all-grads parity."""
    from transformers import AutoModelForCausalLM

    _register_all()
    torch.manual_seed(0)
    cfg = _nemotron_cfg(64)
    model = AutoModelForCausalLM.from_config(cfg).cuda().bfloat16()
    _model_compare(model, cfg, ("scattermoe", "sonicmoe"))


def test_single_layer_model_peft_lora():
    """PEFT target_parameters LoRA on experts.up/down + LoRA on the latent projs."""
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    _register_all()
    torch.manual_seed(0)
    cfg = _nemotron_cfg(64)
    model = AutoModelForCausalLM.from_config(cfg).cuda().bfloat16()
    lcfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        bias="none",
        target_modules=["fc1_latent_proj", "fc2_latent_proj"],
        target_parameters=["experts.up_proj", "experts.down_proj"],
    )
    model = get_peft_model(model, lcfg).cuda()
    _model_compare(
        model.base_model.model,
        cfg,
        ("scattermoe", "sonicmoe"),
        grad_filter=lambda n: "lora" in n,
    )


try:
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
except ImportError:
    NVFP4Tensor = None  # type: ignore[assignment,misc]


@pytest.mark.skipif(NVFP4Tensor is None, reason="torchao required")
@pytest.mark.parametrize("impl", ["scattermoe", "sonicmoe"])
def test_nvfp4_lora_parity(impl, monkeypatch):
    """NVFP4 base + LoRA matches the same forward on the dequantized base."""
    from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHExperts

    import axolotl.integrations.kernels.libs.scattermoe_lora.experts as sm_ex
    from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
        _convert_smoe_lora,
    )

    _register_all()
    torch.manual_seed(0)
    E, L, IM, T, K, R, SC = 16, 64, 128, 128, 4, 8, 2.0
    cfg = _nemotron_cfg(L, moe_intermediate_size=IM)
    dt = torch.bfloat16

    g = torch.Generator(device="cuda").manual_seed(0)
    up_bf = torch.randn(E, IM, L, device="cuda", dtype=dt, generator=g) * 0.1
    dn_bf = torch.randn(E, L, IM, device="cuda", dtype=dt, generator=g) * 0.1
    up_nv = NVFP4Tensor.to_nvfp4(up_bf.clone(), block_size=16)
    dn_nv = NVFP4Tensor.to_nvfp4(dn_bf.clone(), block_size=16)
    up_dq, dn_dq = up_nv.dequantize(dt).contiguous(), dn_nv.dequantize(dt).contiguous()

    x = torch.randn(T, L, device="cuda", dtype=dt, generator=g)
    idx = torch.randint(0, E, (T, K), device="cuda", generator=g)
    wts = torch.softmax(torch.randn(T, K, device="cuda", generator=g), -1).to(dt)
    gout = torch.randn(T, L, device="cuda", dtype=dt, generator=g)

    def mk(*s):
        return (
            torch.randn(*s, device="cuda", dtype=dt, generator=g) * 0.02
        ).requires_grad_()

    A1, B1, A2, B2 = mk(R * E, L), mk(IM, R * E), mk(R * E, IM), mk(L, R * E)
    loras = [A1, B1, A2, B2]

    def run(up, dn, fp4_mode):
        experts = NemotronHExperts(cfg).to("cuda", dt)
        experts.up_proj = torch.nn.Parameter(up, requires_grad=False)
        experts.down_proj = torch.nn.Parameter(dn, requires_grad=False)
        if impl == "scattermoe":
            experts._scattermoe_lora = {
                "up_proj": _convert_smoe_lora(A1, B1, E, R, SC),
                "down_proj": _convert_smoe_lora(A2, B2, E, R, SC),
            }
            monkeypatch.setattr(sm_ex.RUNTIME, "fp4_grouped_mode", fp4_mode)
        else:
            experts._sonicmoe_lora = {
                "up_proj": (A1, B1, SC),
                "down_proj": (A2, B2, SC),
            }
        for t in loras:
            t.grad = None
        cfg._experts_implementation = impl
        xi = x.clone().requires_grad_(True)
        out = experts(xi, idx, wts)
        out.backward(gout)
        return out.detach(), xi.grad, [t.grad.clone() for t in loras]

    out_nv, dx_nv, gl_nv = run(up_nv, dn_nv, "nvfp4")
    out_bf, dx_bf, gl_bf = run(up_dq, dn_dq, None)

    assert _rel(out_nv, out_bf) < 6e-2
    assert _rel(dx_nv, dx_bf) < 6e-2
    for i, (a, b) in enumerate(zip(gl_nv, gl_bf, strict=True)):
        assert _rel(a, b) < 6e-2, f"lora grad {i}"
        assert torch.isfinite(a).all()
