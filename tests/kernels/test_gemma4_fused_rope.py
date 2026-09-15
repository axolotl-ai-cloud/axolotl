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


def _reference_partial_norm_rope(x, weight, cos, sin, eps):
    """Reference: Gemma4RMSNorm over the full head_dim, then stock
    ``apply_rotary_pos_emb`` over the first ``cos.shape[-1]`` columns, with
    the trailing columns passed through unchanged. Mirrors how Llama-style
    partial rotary is layered on top of the stock RMSNorm + RoPE primitives.
    """
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4RMSNorm,
        apply_rotary_pos_emb,
    )

    D = x.shape[-1]
    n_rot = cos.shape[-1]
    norm = Gemma4RMSNorm(D, eps=eps).to(x.device, x.dtype)
    norm.weight.data.copy_(weight)
    normed = norm(x)
    if n_rot == D:
        return apply_rotary_pos_emb(normed, cos, sin, unsqueeze_dim=2)
    x_rot = normed[..., :n_rot]
    x_pass = normed[..., n_rot:]
    rotated = apply_rotary_pos_emb(x_rot, cos, sin, unsqueeze_dim=2)
    return torch.cat([rotated, x_pass], dim=-1)


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


class TestFusedRMSNormRoPEPartialRotary:
    """Partial-rotary: cos/sin last dim is smaller than head_dim.

    Compares against the original primitives (`Gemma4RMSNorm` +
    `apply_rotary_pos_emb`) applied to the rotated slice with the trailing
    columns passed through. Without the kernel fix this used to crash with
    `RuntimeError: shape '[..., D]' is invalid for input of size B*S*n_rot`.
    """

    @pytest.mark.parametrize(
        "B,S,H,D,n_rot",
        [
            (2, 16, 4, 64, 32),  # half rotary (Llama-style 0.5)
            (2, 16, 4, 64, 16),  # quarter rotary
            (2, 32, 8, 128, 64),  # half rotary, larger heads
            (1, 8, 2, 256, 64),  # 26B sliding-shape, 0.25 partial
            (1, 8, 2, 64, 64),  # n_rot == D: must still match full-rotary path
        ],
        ids=["half_64", "quarter_64", "half_128", "quarter_256", "full_64"],
    )
    def test_forward_matches_reference(self, B, S, H, D, n_rot):
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        eps = 1e-6
        x = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        cos = torch.randn(B, S, n_rot, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, n_rot, device="cuda", dtype=torch.bfloat16)

        y_ref = _reference_partial_norm_rope(x.clone(), weight, cos, sin, eps)
        y_fused = fused_rms_norm_rope(x.clone(), weight, cos, sin, eps=eps)

        assert y_fused.shape == y_ref.shape == (B, S, H, D)
        cos_sim = torch.nn.functional.cosine_similarity(
            y_ref.flatten().float(), y_fused.flatten().float(), dim=0
        )
        assert cos_sim > 0.999, (
            f"partial rotary forward cosine_sim={cos_sim:.6f} "
            f"(B={B},S={S},H={H},D={D},n_rot={n_rot})"
        )

        # The pass-through tail must equal the reference RMSNorm output bit-
        # for-bit (any deviation would mean the kernel is touching it with a
        # spurious rotation, which is the original bug class).
        torch.testing.assert_close(
            y_fused[..., n_rot:], y_ref[..., n_rot:], rtol=1e-2, atol=1e-2
        )

    @pytest.mark.parametrize(
        "B,S,H,D,n_rot",
        [(2, 16, 4, 64, 32), (1, 8, 2, 256, 64)],
        ids=["half_64", "quarter_256"],
    )
    def test_x_grad_matches_reference(self, B, S, H, D, n_rot):
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        eps = 1e-6
        cos = torch.randn(B, S, n_rot, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, n_rot, device="cuda", dtype=torch.bfloat16)
        weight_init = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        x_data = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

        # Reference backward via the original primitives
        x_ref = x_data.clone().requires_grad_(True)
        w_ref = weight_init.clone()
        y_ref = _reference_partial_norm_rope(x_ref, w_ref, cos, sin, eps)
        y_ref.sum().backward()

        # Fused backward
        x_fused = x_data.clone().requires_grad_(True)
        w_fused = weight_init.clone().requires_grad_(True)
        y_fused = fused_rms_norm_rope(x_fused, w_fused, cos, sin, eps=eps)
        y_fused.sum().backward()

        cos_sim_x = torch.nn.functional.cosine_similarity(
            x_fused.grad.flatten().float(), x_ref.grad.flatten().float(), dim=0
        )
        assert cos_sim_x > 0.999, f"partial rotary x grad cosine_sim={cos_sim_x:.6f}"

    @pytest.mark.parametrize(
        "B,S,H,D,n_rot",
        [(2, 16, 4, 64, 32), (1, 8, 2, 256, 64)],
        ids=["half_64", "quarter_256"],
    )
    def test_weight_grad_matches_reference(self, B, S, H, D, n_rot):
        from transformers.models.gemma4.modeling_gemma4 import Gemma4RMSNorm

        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        eps = 1e-6
        cos = torch.randn(B, S, n_rot, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, n_rot, device="cuda", dtype=torch.bfloat16)
        weight_init = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        x_data = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)

        # Reference: Gemma4RMSNorm whose .weight collects grads, then partial
        # rotary applied to the rotated slice.
        norm_ref = Gemma4RMSNorm(D, eps=eps).cuda().to(torch.bfloat16)
        norm_ref.weight = torch.nn.Parameter(weight_init.clone())
        normed = norm_ref(x_data)
        from transformers.models.gemma4.modeling_gemma4 import apply_rotary_pos_emb

        rotated = apply_rotary_pos_emb(normed[..., :n_rot], cos, sin, unsqueeze_dim=2)
        y_ref = torch.cat([rotated, normed[..., n_rot:]], dim=-1)
        y_ref.sum().backward()

        w_fused = weight_init.clone().requires_grad_(True)
        fused_rms_norm_rope(x_data.clone(), w_fused, cos, sin, eps=eps).sum().backward()

        cos_sim_w = torch.nn.functional.cosine_similarity(
            w_fused.grad.flatten().float(),
            norm_ref.weight.grad.flatten().float(),
            dim=0,
        )
        assert cos_sim_w > 0.995, (
            f"partial rotary weight grad cosine_sim={cos_sim_w:.6f}"
        )

    def test_full_rotary_unchanged_when_n_rot_equals_d(self):
        """Regression: passing cos/sin with shape == head_dim must still
        match the full-rotary reference (the partial-rotary code path must
        not perturb the existing full-rotary output)."""
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        B, S, H, D = 2, 16, 4, 64
        eps = 1e-6
        x = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(D, device="cuda", dtype=torch.bfloat16)
        cos = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, D, device="cuda", dtype=torch.bfloat16)

        y_ref = _reference_norm_rope(x.clone(), weight, cos, sin, eps)
        y_fused = fused_rms_norm_rope(x.clone(), weight, cos, sin, eps=eps)
        cos_sim = torch.nn.functional.cosine_similarity(
            y_ref.flatten().float(), y_fused.flatten().float(), dim=0
        )
        assert cos_sim > 0.999, f"full-rotary regression cos_sim={cos_sim:.6f}"

    def test_validation_errors(self):
        """Wrapper rejects misshaped inputs cleanly (instead of a cryptic
        Triton crash deeper in the kernel)."""
        from axolotl.kernels.gemma4_fused_rope import fused_rms_norm_rope

        B, S, H, D = 1, 4, 2, 64
        x = torch.randn(B, S, H, D, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(D, device="cuda", dtype=torch.bfloat16)

        # n_rot > head_dim
        cos_big = torch.randn(B, S, D + 16, device="cuda", dtype=torch.bfloat16)
        sin_big = torch.randn(B, S, D + 16, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="cannot exceed head_dim"):
            fused_rms_norm_rope(x, w, cos_big, sin_big)

        # cos/sin last-dim mismatch
        cos = torch.randn(B, S, 32, device="cuda", dtype=torch.bfloat16)
        sin = torch.randn(B, S, 16, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="same last dim"):
            fused_rms_norm_rope(x, w, cos, sin)

        # odd rotary dim
        cos_odd = torch.randn(B, S, 31, device="cuda", dtype=torch.bfloat16)
        sin_odd = torch.randn(B, S, 31, device="cuda", dtype=torch.bfloat16)
        with pytest.raises(ValueError, match="must be even"):
            fused_rms_norm_rope(x, w, cos_odd, sin_odd)


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
