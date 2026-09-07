"""
Correctness tests for fused RMSNorm + SiLU Gate kernel.

Tests against the eager Qwen3_5RMSNormGated implementation.
"""

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("triton", reason="triton required for fused kernels")

if not torch.cuda.is_available():
    pytest.skip("CUDA required for fused kernel tests", allow_module_level=True)

from axolotl.kernels.rms_norm_gated import FusedRMSNormGated


class EagerRMSNormGated(torch.nn.Module):
    """Reference implementation matching Qwen3_5RMSNormGated exactly."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states, gate=None):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        hidden_states = self.weight * hidden_states.to(input_dtype)
        hidden_states = hidden_states * F.silu(gate.to(torch.float32))
        return hidden_states.to(input_dtype)


def _sync_weights(eager_mod, fused_mod):
    """Copy weights from eager to fused module."""
    fused_mod.weight.data.copy_(eager_mod.weight.data)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 128, 256),
        (4, 64, 512),
        (1, 32, 1024),
        (2, 16, 2560),  # Qwen3.5-4B hidden_size
        (2, 16, 4096),  # Qwen3.5-9B hidden_size
        (1, 8, 5120),  # Qwen3.5-27B hidden_size
        (4, 16, 2048),  # Qwen3.5-35B-A3B (MoE) hidden_size
        (4, 16, 3072),  # Qwen3.5-122B-A10B (MoE) hidden_size
    ],
)
class TestRMSNormGatedForward:
    def test_output_matches_eager(self, dtype, shape):
        torch.manual_seed(42)
        B, T, H = shape
        X = torch.randn(B, T, H, dtype=dtype, device="cuda")
        G = torch.randn(B, T, H, dtype=dtype, device="cuda")

        eager = EagerRMSNormGated(H).to(dtype=dtype, device="cuda")
        fused = FusedRMSNormGated(H).to(dtype=dtype, device="cuda")
        _sync_weights(eager, fused)

        y_eager = eager(X, gate=G)
        y_fused = fused(X, gate=G)

        if dtype == torch.float32:
            torch.testing.assert_close(y_fused, y_eager, atol=1e-5, rtol=1e-5)
        else:
            torch.testing.assert_close(y_fused, y_eager, atol=1e-2, rtol=1e-2)

    def test_output_shape(self, dtype, shape):
        B, T, H = shape
        X = torch.randn(B, T, H, dtype=dtype, device="cuda")
        G = torch.randn(B, T, H, dtype=dtype, device="cuda")

        fused = FusedRMSNormGated(H).to(dtype=dtype, device="cuda")
        y = fused(X, gate=G)
        assert y.shape == (B, T, H)
        assert y.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "shape",
    [
        (2, 32, 256),
        (2, 16, 512),
        (2, 16, 2560),  # Qwen3.5-4B
        (1, 8, 4096),  # Qwen3.5-9B
        (1, 8, 5120),  # Qwen3.5-27B
        (2, 16, 2048),  # Qwen3.5-35B-A3B (MoE)
        (2, 16, 3072),  # Qwen3.5-122B-A10B (MoE)
    ],
)
class TestRMSNormGatedBackward:
    def test_grad_x(self, dtype, shape):
        torch.manual_seed(42)
        B, T, H = shape
        X = torch.randn(B, T, H, dtype=dtype, device="cuda", requires_grad=True)
        G = torch.randn(B, T, H, dtype=dtype, device="cuda", requires_grad=True)
        X_ref = X.detach().clone().requires_grad_(True)
        G_ref = G.detach().clone().requires_grad_(True)

        eager = EagerRMSNormGated(H).to(dtype=dtype, device="cuda")
        fused = FusedRMSNormGated(H).to(dtype=dtype, device="cuda")
        _sync_weights(eager, fused)

        y_eager = eager(X_ref, gate=G_ref)
        y_fused = fused(X, gate=G)

        grad_out = torch.randn_like(y_eager)
        y_eager.backward(grad_out)
        y_fused.backward(grad_out)

        if dtype == torch.float32:
            atol, rtol = 1e-4, 1e-4
        else:
            atol, rtol = 5e-2, 5e-2

        torch.testing.assert_close(X.grad, X_ref.grad, atol=atol, rtol=rtol)

    def test_grad_gate(self, dtype, shape):
        torch.manual_seed(42)
        B, T, H = shape
        X = torch.randn(B, T, H, dtype=dtype, device="cuda", requires_grad=True)
        G = torch.randn(B, T, H, dtype=dtype, device="cuda", requires_grad=True)
        X_ref = X.detach().clone().requires_grad_(True)
        G_ref = G.detach().clone().requires_grad_(True)

        eager = EagerRMSNormGated(H).to(dtype=dtype, device="cuda")
        fused = FusedRMSNormGated(H).to(dtype=dtype, device="cuda")
        _sync_weights(eager, fused)

        y_eager = eager(X_ref, gate=G_ref)
        y_fused = fused(X, gate=G)

        grad_out = torch.randn_like(y_eager)
        y_eager.backward(grad_out)
        y_fused.backward(grad_out)

        if dtype == torch.float32:
            atol, rtol = 1e-4, 1e-4
        else:
            atol, rtol = 5e-2, 5e-2

        torch.testing.assert_close(G.grad, G_ref.grad, atol=atol, rtol=rtol)

    def test_grad_weight(self, dtype, shape):
        torch.manual_seed(42)
        B, T, H = shape
        X = torch.randn(B, T, H, dtype=dtype, device="cuda", requires_grad=True)
        G = torch.randn(B, T, H, dtype=dtype, device="cuda", requires_grad=True)
        X_ref = X.detach().clone().requires_grad_(True)
        G_ref = G.detach().clone().requires_grad_(True)

        eager = EagerRMSNormGated(H).to(dtype=dtype, device="cuda")
        fused = FusedRMSNormGated(H).to(dtype=dtype, device="cuda")
        _sync_weights(eager, fused)

        y_eager = eager(X_ref, gate=G_ref)
        y_fused = fused(X, gate=G)

        grad_out = torch.randn_like(y_eager)
        y_eager.backward(grad_out)
        y_fused.backward(grad_out)

        if dtype == torch.float32:
            atol, rtol = 1e-4, 1e-4
        else:
            atol, rtol = 5e-2, 5e-2

        torch.testing.assert_close(
            fused.weight.grad, eager.weight.grad, atol=atol, rtol=rtol
        )


class TestRMSNormGatedEdgeCases:
    def test_gate_none_raises(self):
        fused = FusedRMSNormGated(256).cuda()
        X = torch.randn(2, 4, 256, device="cuda")
        with pytest.raises(ValueError, match="requires a gate tensor"):
            fused(X, gate=None)

    def test_2d_input(self):
        """Test with (BxT, H) shaped input instead of (B, T, H)."""
        torch.manual_seed(42)
        H = 512
        X = torch.randn(64, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        G = torch.randn(64, H, dtype=torch.bfloat16, device="cuda", requires_grad=True)
        X_ref = X.detach().clone().requires_grad_(True)
        G_ref = G.detach().clone().requires_grad_(True)

        eager = EagerRMSNormGated(H).to(dtype=torch.bfloat16, device="cuda")
        fused = FusedRMSNormGated(H).to(dtype=torch.bfloat16, device="cuda")
        _sync_weights(eager, fused)

        y_eager = eager(X_ref, gate=G_ref)
        y_fused = fused(X, gate=G)

        torch.testing.assert_close(y_fused, y_eager, atol=1e-2, rtol=1e-2)

        grad_out = torch.randn_like(y_eager)
        y_eager.backward(grad_out)
        y_fused.backward(grad_out)

        torch.testing.assert_close(X.grad, X_ref.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(G.grad, G_ref.grad, atol=5e-2, rtol=5e-2)

    def test_random_weight_init(self):
        """Test with non-default weight values."""
        torch.manual_seed(123)
        H = 256
        X = torch.randn(2, 16, H, dtype=torch.bfloat16, device="cuda")
        G = torch.randn(2, 16, H, dtype=torch.bfloat16, device="cuda")

        eager = EagerRMSNormGated(H).to(dtype=torch.bfloat16, device="cuda")
        # Randomize weights
        eager.weight.data = torch.randn_like(eager.weight.data)

        fused = FusedRMSNormGated(H).to(dtype=torch.bfloat16, device="cuda")
        _sync_weights(eager, fused)

        y_eager = eager(X, gate=G)
        y_fused = fused(X, gate=G)
        torch.testing.assert_close(y_fused, y_eager, atol=1e-2, rtol=1e-2)
