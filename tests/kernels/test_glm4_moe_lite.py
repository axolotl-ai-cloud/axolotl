"""Output and gradient parity for fused GLM MLA projection assembly."""

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference(q, kv, k_rot, cos, sin, nope, interleave):
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        apply_rotary_pos_emb,
        apply_rotary_pos_emb_interleave,
    )

    rotary = apply_rotary_pos_emb_interleave if interleave else apply_rotary_pos_emb
    q_rot, k_rot = rotary(q[..., nope:].transpose(1, 2), k_rot.unsqueeze(1), cos, sin)
    q_out = torch.cat((q[..., :nope], q_rot.transpose(1, 2)), dim=-1)
    k_out = torch.cat(
        (kv[..., :nope], k_rot.transpose(1, 2).expand_as(q[..., nope:])), dim=-1
    )
    return q_out, k_out, kv[..., nope:]


def _inputs(shape, dtype, broadcast=False):
    b, s, h, p, r, v = shape
    q = torch.randn(b, s, h, p + r, device="cuda", dtype=dtype, requires_grad=True)
    kv = torch.randn(b, s, h, p + v, device="cuda", dtype=dtype, requires_grad=True)
    compressed = torch.randn(
        b, s, 512 + r, device="cuda", dtype=dtype, requires_grad=True
    )
    k_rot = compressed[..., 512:]
    k_rot.retain_grad()
    cos = torch.randn(1 if broadcast else b, s, r, device="cuda", dtype=dtype)
    sin = torch.randn_like(cos)
    return q, kv, k_rot, cos, sin


def _assert_close(actual, expected, dtype):
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.02 if dtype != torch.float32 else 1e-5,
        atol=0.04 if dtype != torch.float32 else 2e-6,
    )
    difference = (actual.float() - expected.float()).norm()
    assert difference / expected.float().norm().clamp_min(1e-8) < (
        0.01 if dtype != torch.float32 else 1e-5
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("shape", [(2, 17, 5, 24, 8, 16), (1, 127, 20, 192, 64, 256)])
def test_forward_backward(dtype, interleave, shape):
    from axolotl.kernels.glm4_moe_lite import fused_mla_prepare

    torch.manual_seed(7)
    inputs = _inputs(shape, dtype, broadcast=True)
    reference_inputs = tuple(
        x.detach().clone().requires_grad_(x.requires_grad) for x in inputs
    )
    actual = fused_mla_prepare(*inputs, shape[3], interleave)
    expected = _reference(*reference_inputs, shape[3], interleave)
    gradients = [torch.randn_like(x) for x in expected]
    for got, want in zip(actual, expected, strict=True):
        assert got.is_contiguous()
        _assert_close(got, want, dtype)
    torch.autograd.backward(actual, gradients)
    torch.autograd.backward(expected, gradients)
    for got, want in zip(inputs[:3], reference_inputs[:3], strict=True):
        _assert_close(got.grad, want.grad, dtype)


def test_unused_outputs_and_strided_rotary_key():
    from axolotl.kernels.glm4_moe_lite import fused_mla_prepare

    q, kv, k_rot, cos, sin = _inputs((2, 7, 3, 8, 16, 12), torch.float32)
    k_rot = k_rot[..., ::2]
    cos, sin = cos[..., ::2], sin[..., ::2]
    q = q[..., :16].detach().requires_grad_()
    _, _, value = fused_mla_prepare(q, kv, k_rot, cos, sin, 8, True)
    value.sum().backward()
    torch.testing.assert_close(q.grad, torch.zeros_like(q))
    torch.testing.assert_close(kv.grad[..., 8:], torch.ones_like(kv.grad[..., 8:]))
    torch.testing.assert_close(kv.grad[..., :8], torch.zeros_like(kv.grad[..., :8]))


def test_fullgraph_compile_backward():
    from axolotl.kernels.glm4_moe_lite import fused_mla_prepare

    inputs = _inputs((1, 17, 5, 24, 8, 16), torch.bfloat16)
    compiled = torch.compile(fused_mla_prepare, fullgraph=True)
    actual = compiled(*inputs, 24, True)
    expected = _reference(*inputs, 24, True)
    for got, want in zip(actual, expected, strict=True):
        _assert_close(got, want, torch.bfloat16)
    sum(x.float().square().mean() for x in actual).backward()
    assert all(torch.isfinite(x.grad).all() for x in inputs[:3])


@pytest.mark.parametrize("invalid", ["negative_nope", "extra_kv_dimension"])
def test_invalid_projection_shapes(invalid):
    from axolotl.kernels.glm4_moe_lite import fused_mla_prepare

    q, kv, k_rot, cos, sin = _inputs((1, 3, 2, 4, 8, 4), torch.float32)
    nope = 4
    if invalid == "negative_nope":
        q, nope = q[..., :4], -4
    else:
        kv = kv.unsqueeze(-1)
    with pytest.raises(ValueError):
        fused_mla_prepare(q, kv, k_rot, cos, sin, nope, True)
