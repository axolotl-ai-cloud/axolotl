"""
Correctness tests for the fused RMSNorm+RoPE Triton kernel.

Tests forward and backward against the reference Gemma4 implementation
(Gemma4RMSNorm + apply_rotary_pos_emb) across both sliding window
(head_dim=256) and global attention (head_dim=512) layer configurations.
"""

import pytest
import torch

torch.manual_seed(42)

# Skip entire module if no CUDA
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _reference_norm_rope(x, weight, cos, sin, eps):
    """Reference: separate Gemma4RMSNorm + apply_rotary_pos_emb."""
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4RMSNorm,
        apply_rotary_pos_emb,
    )

    D = x.shape[-1]
    norm = Gemma4RMSNorm(D, eps=eps).to(x.device, x.dtype)
    norm.weight.data.copy_(weight)
    normed = norm(x)
    return apply_rotary_pos_emb(normed, cos, sin, unsqueeze_dim=2)


def _reference_norm_noscale(x, eps):
    """Reference: Gemma4RMSNorm with_scale=False."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

    D = x.shape[-1]
    norm = Gemma4RMSNorm(D, eps=eps, with_scale=False).to(x.device, x.dtype)
    return norm(x)


@pytest.fixture(
    params=[
        (2, 64, 32, 256),  # sliding window layer shape
        (2, 64, 4, 512),  # global attention layer shape
        (1, 128, 16, 256),  # different batch/seq
        (1, 1, 1, 8),  # minimal size
    ],
    ids=["sliding_256", "global_512", "varied", "minimal"],
)
def shapes(request):
    return request.param


@pytest.fixture(params=[torch.bfloat16, torch.float16], ids=["bf16", "fp16"])
def dtype(request):
    return request.param


class TestFusedRMSNormRoPEForward:
    """Forward pass correctness."""

    def test_matches_reference(self, shapes, dtype):
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        B, S, H, D = shapes
        eps = 1e-6
        x = torch.randn(B, S, H, D, device="cuda", dtype=dtype)
        weight = torch.randn(D, device="cuda", dtype=dtype)
        cos = torch.randn(B, S, D, device="cuda", dtype=dtype)
        sin = torch.randn(B, S, D, device="cuda", dtype=dtype)

        y_ref = _reference_norm_rope(x.clone(), weight, cos, sin, eps)
        y_fused = fused_rms_norm_rope(x.clone(), weight, cos, sin, eps=eps)

        cos_sim = torch.nn.functional.cosine_similarity(
            y_ref.flatten().float(), y_fused.flatten().float(), dim=0
        )
        assert cos_sim > 0.999, f"Forward cosine_sim={cos_sim:.6f}, expected > 0.999"

    def test_output_shape(self, shapes):
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        B, S, H, D = shapes
        x = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        cos = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)

        y = fused_rms_norm_rope(x, weight, cos, sin, eps=1e-6)
        assert y.shape == x.shape
        assert y.dtype == x.dtype


class TestFusedRMSNormRoPEBackward:
    """Backward pass correctness via gradient comparison."""

    @pytest.mark.parametrize(
        "B,S,H,D",
        [(2, 64, 32, 256), (2, 64, 4, 512)],
        ids=["sliding_256", "global_512"],
    )
    def test_x_grad_matches_reference(self, B, S, H, D):
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4RMSNorm,
            apply_rotary_pos_emb,
        )

        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        eps = 1e-6
        cos = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        weight_init = torch.randn(D, device="cuda", dtype=torch.bfloat16)

        # Reference backward
        x_ref = torch.randn(
            B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        norm_ref = Gemma4RMSNorm(D, eps=eps).cuda().to(torch.bfloat16)
        norm_ref.weight.data.copy_(weight_init)
        y_ref = apply_rotary_pos_emb(norm_ref(x_ref), cos, sin, unsqueeze_dim=2)
        y_ref.sum().backward()

        # Fused backward
        x_fused = x_ref.data.clone().requires_grad_(True)
        w_fused = weight_init.clone().requires_grad_(True)
        y_fused = fused_rms_norm_rope(x_fused, w_fused, cos, sin, eps=eps)
        y_fused.sum().backward()

        cos_sim_x = torch.nn.functional.cosine_similarity(
            x_fused.grad.flatten().float(), x_ref.grad.flatten().float(), dim=0
        )
        assert cos_sim_x > 0.999, f"x grad cosine_sim={cos_sim_x:.6f}, expected > 0.999"

    @pytest.mark.parametrize(
        "B,S,H,D",
        [(2, 64, 32, 256), (2, 64, 4, 512)],
        ids=["sliding_256", "global_512"],
    )
    def test_weight_grad_matches_reference(self, B, S, H, D):
        from transformers.models.gemma4.modeling_gemma4 import (
            Gemma4RMSNorm,
            apply_rotary_pos_emb,
        )

        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        eps = 1e-6
        cos = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        weight_init = torch.randn(D, device="cuda", dtype=torch.bfloat16)

        # Reference
        x_ref = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        norm_ref = Gemma4RMSNorm(D, eps=eps).cuda().to(torch.bfloat16)
        norm_ref.weight = torch.nn.Parameter(weight_init.clone())
        apply_rotary_pos_emb(
            norm_ref(x_ref), cos, sin, unsqueeze_dim=2
        ).sum().backward()

        # Fused
        w_fused = weight_init.clone().requires_grad_(True)
        fused_rms_norm_rope(x_ref.clone(), w_fused, cos, sin, eps=eps).sum().backward()

        cos_sim_w = torch.nn.functional.cosine_similarity(
            w_fused.grad.flatten().float(),
            norm_ref.weight.grad.flatten().float(),
            dim=0,
        )
        assert cos_sim_w > 0.995, (
            f"weight grad cosine_sim={cos_sim_w:.6f}, expected > 0.995"
        )

    def test_grad_flows(self):
        """Verify gradients are non-zero and finite."""
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        B, S, H, D = 1, 16, 4, 64
        x = torch.randn(
            B, S, H, D, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        w = torch.randn(D, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        cos = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)

        y = fused_rms_norm_rope(x, w, cos, sin, eps=1e-6)
        y.sum().backward()

        assert x.grad is not None, "x.grad is None"
        assert w.grad is not None, "w.grad is None"
        assert x.grad.isfinite().all(), "x.grad has non-finite values"
        assert w.grad.isfinite().all(), "w.grad has non-finite values"
        assert x.grad.abs().sum() > 0, "x.grad is all zeros"
        assert w.grad.abs().sum() > 0, "w.grad is all zeros"


class TestFusedRMSNormNoScale:
    """Tests for v_norm (RMSNorm without learnable scale)."""

    def test_forward_matches_reference(self, shapes, dtype):
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_noscale

        B, S, H, D = shapes
        eps = 1e-6
        x = torch.randn(B, S, H, D, device="cuda", dtype=dtype)

        y_ref = _reference_norm_noscale(x.clone(), eps)
        y_fused = fused_rms_norm_noscale(x.clone(), eps=eps)

        cos_sim = torch.nn.functional.cosine_similarity(
            y_ref.flatten().float(), y_fused.flatten().float(), dim=0
        )
        assert cos_sim > 0.999, f"v_norm cosine_sim={cos_sim:.6f}, expected > 0.999"

    def test_backward_flows(self):
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_noscale

        x = torch.randn(
            1, 16, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        y = fused_rms_norm_noscale(x, eps=1e-6)
        y.sum().backward()

        assert x.grad is not None
        assert x.grad.isfinite().all()
        assert x.grad.abs().sum() > 0
