"""GLM Liger plugin flags, loss preservation, and kernel parity."""

import copy

import pytest
import torch

from axolotl.utils.dict import DictDefault


@pytest.fixture
def glm(monkeypatch):
    from transformers.models.glm4_moe_lite import modeling_glm4_moe_lite as module

    for name in (
        "Glm4MoeLiteRMSNorm",
        "apply_rotary_pos_emb",
        "apply_rotary_pos_emb_interleave",
    ):
        monkeypatch.setattr(module, name, getattr(module, name))
    monkeypatch.setattr(module.Glm4MoeLiteMLP, "forward", module.Glm4MoeLiteMLP.forward)
    return module


def _patch(**flags):
    from axolotl.integrations.liger import LigerPlugin

    config = DictDefault(model_config_type="glm4_moe_lite", **flags)
    LigerPlugin().pre_model_load(config)


def test_nonloss_flags_preserve_loss_and_shared_expert_width(glm):
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.swiglu import LigerSwiGLUMLP
    from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import (
        Glm4MoeLiteConfig,
    )

    loss_forward = glm.Glm4MoeLiteForCausalLM.forward
    experts_forward = glm.Glm4MoeLiteExperts.forward
    _patch(
        liger_rope=True,
        liger_rms_norm=True,
        liger_glu_activation=True,
        liger_cross_entropy=False,
        liger_fused_linear_cross_entropy=False,
    )
    config = Glm4MoeLiteConfig(hidden_size=32, intermediate_size=64)
    shared = glm.Glm4MoeLiteMLP(config, intermediate_size=48)
    assert shared.gate_proj.out_features == 48
    assert shared.forward.__func__ is LigerSwiGLUMLP.forward
    assert glm.Glm4MoeLiteRMSNorm is LigerRMSNorm
    assert glm.Glm4MoeLiteForCausalLM.forward is loss_forward
    assert glm.Glm4MoeLiteExperts.forward is experts_forward


def test_disabled_flags_are_noop(glm):
    before = (
        glm.Glm4MoeLiteRMSNorm,
        glm.Glm4MoeLiteMLP.forward,
        glm.apply_rotary_pos_emb_interleave,
    )
    _patch(
        liger_rope=False,
        liger_rms_norm=False,
        liger_glu_activation=False,
        liger_cross_entropy=False,
        liger_fused_linear_cross_entropy=False,
    )
    assert before == (
        glm.Glm4MoeLiteRMSNorm,
        glm.Glm4MoeLiteMLP.forward,
        glm.apply_rotary_pos_emb_interleave,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_liger_forward_and_gradients(glm):
    from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import (
        Glm4MoeLiteConfig,
    )

    torch.manual_seed(42)
    config = Glm4MoeLiteConfig(hidden_size=128, intermediate_size=256)
    original_rope = glm.apply_rotary_pos_emb_interleave
    norm = glm.Glm4MoeLiteRMSNorm(128).cuda().bfloat16()
    mlp = glm.Glm4MoeLiteMLP(config, intermediate_size=192).cuda().bfloat16()
    original_mlp_forward = glm.Glm4MoeLiteMLP.forward
    expected_mlp = copy.deepcopy(mlp)
    expected_mlp.forward = original_mlp_forward.__get__(expected_mlp)
    _patch(
        liger_rope=True,
        liger_rms_norm=True,
        liger_glu_activation=True,
        liger_cross_entropy=False,
        liger_fused_linear_cross_entropy=False,
    )
    fused_norm = glm.Glm4MoeLiteRMSNorm(128).cuda().bfloat16()
    fused_norm.load_state_dict(norm.state_dict())
    x = torch.randn(2, 17, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    q = x.view(2, 17, 2, 64).transpose(1, 2)
    k = q[:, :1]
    cos = torch.randn(2, 17, 64, device="cuda", dtype=torch.bfloat16)
    sin = torch.randn_like(cos)
    reference_rope = original_rope(q, k, cos, sin)
    actual_rope = glm.apply_rotary_pos_emb_interleave(
        y.view(2, 17, 2, 64).transpose(1, 2), y[..., :64].unsqueeze(1), cos, sin
    )
    expected = expected_mlp(norm(x))
    actual = mlp(fused_norm(y))
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.01)
    for a, b in zip(actual_rope, reference_rope, strict=True):
        torch.testing.assert_close(a, b, rtol=0.03, atol=0.04)
    gradients = [torch.randn_like(t) for t in (expected, *reference_rope)]
    torch.autograd.backward((expected, *reference_rope), gradients)
    torch.autograd.backward((actual, *actual_rope), gradients)
    torch.testing.assert_close(y.grad, x.grad, rtol=0.04, atol=0.08)
    for a, b in zip(mlp.parameters(), expected_mlp.parameters(), strict=True):
        torch.testing.assert_close(a.grad, b.grad, rtol=0.04, atol=0.04)
    torch.testing.assert_close(
        fused_norm.weight.grad, norm.weight.grad, rtol=0.04, atol=0.04
    )
