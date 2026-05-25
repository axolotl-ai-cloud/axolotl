"""Correctness tests for the ``unit_offset=True`` (Gemma-style) path in the fused RMSNorm+RoPE kernel."""

import pytest
import torch

torch.manual_seed(7)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def _reference_unit_offset(x, weight, cos, sin, eps):
    x_fp32 = x.float()
    rstd = torch.rsqrt(x_fp32.pow(2).mean(-1, keepdim=True) + eps)
    normed = (x_fp32 * rstd * (1.0 + weight.float())).to(x.dtype)
    cos_b = cos.unsqueeze(2)
    sin_b = sin.unsqueeze(2)
    return normed * cos_b + _rotate_half(normed) * sin_b


class TestForward:
    def test_matches_reference(self):
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        B, S, H, D = 2, 32, 4, 64
        eps = 1e-6
        x = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.1
        cos = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)

        y_ref = _reference_unit_offset(x.clone(), weight, cos, sin, eps)
        y_fused = fused_rms_norm_rope(
            x.clone(), weight, cos, sin, eps=eps, unit_offset=True
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            y_ref.flatten().float(), y_fused.flatten().float(), dim=0
        )
        assert cos_sim > 0.999, f"forward cosine_sim={cos_sim:.6f}"
        torch.testing.assert_close(y_fused, y_ref, rtol=5e-2, atol=5e-2)

    def test_differs_from_no_offset(self):
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        B, S, H, D = 1, 8, 2, 32
        x = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        cos = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)

        y_off = fused_rms_norm_rope(x, weight, cos, sin, unit_offset=False)
        y_on = fused_rms_norm_rope(x, weight, cos, sin, unit_offset=True)
        diff = (y_off.float() - y_on.float()).abs().max().item()
        assert diff > 1e-3, f"unit_offset toggle had no effect: max_abs_diff={diff}"


class TestBackward:
    def test_x_and_w_grad_match_eager(self):
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        B, S, H, D = 2, 16, 4, 64
        eps = 1e-6
        cos = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        weight_init = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.1

        x_ref = torch.randn(
            B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w_ref = weight_init.clone().requires_grad_(True)
        y_ref = _reference_unit_offset(x_ref, w_ref, cos, sin, eps)
        y_ref.sum().backward()

        x_fused = x_ref.data.clone().requires_grad_(True)
        w_fused = weight_init.clone().requires_grad_(True)
        y_fused = fused_rms_norm_rope(
            x_fused, w_fused, cos, sin, eps=eps, unit_offset=True
        )
        y_fused.sum().backward()

        cos_sim_x = torch.nn.functional.cosine_similarity(
            x_fused.grad.flatten().float(), x_ref.grad.flatten().float(), dim=0
        )
        cos_sim_w = torch.nn.functional.cosine_similarity(
            w_fused.grad.flatten().float(), w_ref.grad.flatten().float(), dim=0
        )
        assert cos_sim_x > 0.999, f"x grad cosine_sim={cos_sim_x:.6f}"
        assert cos_sim_w > 0.995, f"w grad cosine_sim={cos_sim_w:.6f}"


class TestCompile:
    def test_compile_fullgraph(self):
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        B, S, H, D = 1, 8, 2, 32
        eps = 1e-6
        x = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(D, device="cuda", dtype=torch.bfloat16) * 0.1
        cos = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)

        eager = fused_rms_norm_rope(x, weight, cos, sin, eps=eps, unit_offset=True)
        compiled_fn = torch.compile(fused_rms_norm_rope, fullgraph=True)
        compiled = compiled_fn(x, weight, cos, sin, eps=eps, unit_offset=True)

        torch.testing.assert_close(compiled, eager, rtol=0, atol=0)
