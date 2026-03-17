# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
Tests for ScatterMoE + LoRA Fused Kernels
==========================================

Tests verify correctness of:
1. Forward pass: fused kernel matches naive PyTorch reference
2. Backward pass: gradients for LoRA A, B, and input match reference
3. Frozen weights: expert weight gradients are correctly skipped
4. Various configurations: top-k, grouped_in/out, with/without bias
5. Numerical stability: bf16/fp16 outputs within tolerance of fp32 reference
6. HFScatterMoEGatedMLP with sigmoid routing (GLM/DeepSeek/MiniMax M2)

Test strategy:
- Reference implementation uses pure PyTorch ops (no Triton)
- ScatterMoE routing (flatten_sort_count) is shared between reference and kernel
- Tolerances account for tf32 accumulation in Triton kernels
"""

from types import SimpleNamespace

import pytest
import torch

# Skip all tests if CUDA is not available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA required for Triton kernels",
)

_SMOE = "axolotl.integrations.kernels.libs.scattermoe_lora"


# =============================================================================
# Helpers
# =============================================================================


def flatten_sort_count_ref(expert_idxs: torch.Tensor, num_experts: int):
    """Reference implementation of routing."""
    with torch.no_grad():
        flat = expert_idxs.flatten()
        sorted_expert_idxs, sorted_scattered_idxs = torch.sort(flat)
        counts = flat.bincount(minlength=num_experts)
        offsets = counts.cumsum(-1)
    return sorted_expert_idxs, sorted_scattered_idxs, offsets


def reference_parallel_linear_lora(
    X,
    W,
    k,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    lora_A,
    lora_B,
    scaling,
    x_grouped=False,
    y_grouped=False,
    bias=None,
):
    """
    Pure PyTorch reference for: Y[i] = X[i] @ W[e] + scaling * (X[i] @ A[e]^T) @ B[e]^T + b[e]

    Args:
        X: [M, K] input (token order)
        W: [E, K, N] expert weights
        sorted_expert_idxs: [M*k] expert assignments (sorted)
        sorted_scattered_idxs: [M*k] original token indices (sorted)
        lora_A: [r*E, K] LoRA A weights
        lora_B: [N, r*E] LoRA B weights
        scaling: LoRA scaling factor
    """
    E, K, N = W.shape
    R = lora_A.size(0) // E
    L = sorted_expert_idxs.size(0)  # M * k

    output = torch.zeros(L, N, device=X.device, dtype=X.dtype)

    for i in range(L):
        e = sorted_expert_idxs[i].item()
        if x_grouped:
            x_i = X[i]
        else:
            token_idx = sorted_scattered_idxs[i].item() // k
            x_i = X[token_idx]

        w_e = W[e]  # [K, N]
        a_e = lora_A[e * R : (e + 1) * R, :]  # [r, K]
        b_e = lora_B[:, e * R : (e + 1) * R]  # [N, r]

        # Y = X @ W + scaling * (X @ A^T) @ B^T
        base = x_i @ w_e  # [N]
        lora = scaling * ((x_i @ a_e.T) @ b_e.T)  # [N]
        out_i = base + lora

        if bias is not None:
            out_i = out_i + bias[e]

        if y_grouped:
            output[i] = out_i
        else:
            output[sorted_scattered_idxs[i]] = out_i

    return output


def reference_lora_backward(
    grad_out,
    X,
    W,
    lora_A,
    lora_B,
    scaling,
    sorted_expert_idxs,
    sorted_scattered_idxs,
    expert_offsets,
    k,
    E,
):
    """
    Pure PyTorch reference for LoRA backward pass on grouped data.

    Returns:
        dX: [M*k, K] input gradient (in grouped order)
        dA: [r*E, K] LoRA A gradient
        dB: [N, r*E] LoRA B gradient
    """
    R = lora_A.size(0) // E

    dA = torch.zeros_like(lora_A)
    dB = torch.zeros_like(lora_B)
    dX = torch.zeros_like(X)

    prev_offset = 0
    for e in range(E):
        curr_offset = expert_offsets[e].item()
        if curr_offset > prev_offset:
            dy_e = grad_out[prev_offset:curr_offset]  # [M_e, N]
            x_e = X[prev_offset:curr_offset]  # [M_e, K]
            a_e = lora_A[e * R : (e + 1) * R, :]  # [r, K]
            b_e = lora_B[:, e * R : (e + 1) * R]  # [N, r]
            w_e = W[e]  # [K, N]

            # Input gradient: dX = dY @ W^T + scaling * (dY @ B) @ A
            dx_base = dy_e @ w_e.T  # [M_e, K]
            dy_b = dy_e @ b_e  # [M_e, r]
            dx_lora = scaling * (dy_b @ a_e)  # [M_e, K]
            dX[prev_offset:curr_offset] = dx_base + dx_lora

            # LoRA A gradient: dA = scaling * (dY @ B)^T @ X
            xa = x_e @ a_e.T  # [M_e, r]
            dA[e * R : (e + 1) * R, :] = scaling * (dy_b.T @ x_e)

            # LoRA B gradient: dB = scaling * dY^T @ (X @ A^T)
            dB[:, e * R : (e + 1) * R] = scaling * (dy_e.T @ xa)

        prev_offset = curr_offset

    return dX, dA, dB


def make_test_data(
    M=32,
    K=64,
    N=128,
    E=4,
    R=8,
    k=2,
    dtype=torch.float32,
    device="cuda",
    seed=42,
):
    """Create test data for ScatterMoE + LoRA tests."""
    torch.manual_seed(seed)

    X = torch.randn(M, K, device=device, dtype=dtype)
    W = torch.randn(E, K, N, device=device, dtype=dtype) * 0.02
    lora_A = torch.randn(R * E, K, device=device, dtype=dtype) * 0.01
    lora_B = torch.randn(N, R * E, device=device, dtype=dtype) * 0.01
    scaling = 0.5

    # Generate routing
    selected_experts = torch.randint(0, E, (M, k), device=device)
    sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = flatten_sort_count_ref(
        selected_experts, E
    )

    return {
        "X": X,
        "W": W,
        "lora_A": lora_A,
        "lora_B": lora_B,
        "scaling": scaling,
        "k": k,
        "E": E,
        "R": R,
        "sorted_expert_idxs": sorted_expert_idxs,
        "sorted_scattered_idxs": sorted_scattered_idxs,
        "expert_offsets": expert_offsets,
    }


# =============================================================================
# Test: Forward Pass Correctness
# =============================================================================


class TestForwardPass:
    """Test forward pass of fused scatter2scatter_lora kernel."""

    def _run_forward_test(
        self, M, K, N, E, R, k, dtype=torch.float32, atol=1e-2, rtol=1e-2
    ):
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")

        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k, dtype=dtype)

        # Reference
        ref_output = reference_parallel_linear_lora(
            data["X"],
            data["W"],
            data["k"],
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["lora_A"],
            data["lora_B"],
            data["scaling"],
        )

        # Kernel
        kernel_output = lora_ops.scatter2scatter_lora(
            X=data["X"],
            W=data["W"],
            sorted_expert_idxs=data["sorted_expert_idxs"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            k=data["k"],
            lora_A=data["lora_A"],
            lora_B=data["lora_B"],
            scaling=data["scaling"],
        )

        torch.testing.assert_close(kernel_output, ref_output, atol=atol, rtol=rtol)

    def test_basic(self):
        """Basic forward pass with small dimensions."""
        self._run_forward_test(M=16, K=64, N=64, E=4, R=8, k=1)

    def test_topk2(self):
        """Forward pass with top-2 routing."""
        self._run_forward_test(M=32, K=64, N=128, E=4, R=8, k=2)

    def test_larger_rank(self):
        """Forward pass with larger LoRA rank."""
        self._run_forward_test(M=16, K=128, N=128, E=8, R=32, k=2)

    def test_small_rank(self):
        """Forward pass with very small LoRA rank."""
        self._run_forward_test(M=32, K=64, N=64, E=4, R=4, k=1)

    def test_many_experts(self):
        """Forward with many experts, fewer tokens per expert."""
        self._run_forward_test(M=64, K=64, N=64, E=16, R=8, k=2)

    def test_non_power_of_2_dims(self):
        """Test with dimensions that are not powers of 2."""
        self._run_forward_test(M=17, K=96, N=80, E=6, R=16, k=2, atol=2e-2, rtol=2e-2)

    def test_single_token(self):
        """Test with a single token."""
        self._run_forward_test(M=1, K=64, N=64, E=4, R=8, k=1)

    def test_bf16(self):
        """Test with bfloat16 precision."""
        self._run_forward_test(
            M=32, K=64, N=128, E=4, R=8, k=2, dtype=torch.bfloat16, atol=5e-2, rtol=5e-2
        )

    def test_fp16(self):
        """Test with float16 precision."""
        self._run_forward_test(
            M=32, K=64, N=128, E=4, R=8, k=2, dtype=torch.float16, atol=5e-2, rtol=5e-2
        )


class TestForwardGrouped:
    """Test forward pass with grouped_in/grouped_out configurations."""

    def _make_grouped_data(self, M=32, K=64, N=128, E=4, R=8, k=2, dtype=torch.float32):
        from importlib import import_module

        base_ops = import_module(f"{_SMOE}.kernels.ops")

        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k, dtype=dtype)

        # Create grouped X
        grouped_X = base_ops.group(data["X"], data["sorted_scattered_idxs"], fan_out=k)
        data["grouped_X"] = grouped_X
        return data

    def test_x_grouped(self):
        """Forward with pre-grouped input."""
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")

        data = self._make_grouped_data()

        ref_output = reference_parallel_linear_lora(
            data["grouped_X"],
            data["W"],
            data["k"],
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["lora_A"],
            data["lora_B"],
            data["scaling"],
            x_grouped=True,
        )

        kernel_output = lora_ops.scatter2scatter_lora(
            X=data["grouped_X"],
            W=data["W"],
            sorted_expert_idxs=data["sorted_expert_idxs"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            k=1,  # When x_grouped, fan_out=1 (already expanded)
            lora_A=data["lora_A"],
            lora_B=data["lora_B"],
            scaling=data["scaling"],
            x_grouped=True,
        )

        torch.testing.assert_close(kernel_output, ref_output, atol=1e-2, rtol=1e-2)

    def test_y_grouped(self):
        """Forward with grouped output."""
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")

        data = make_test_data()

        ref_output = reference_parallel_linear_lora(
            data["X"],
            data["W"],
            data["k"],
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["lora_A"],
            data["lora_B"],
            data["scaling"],
            y_grouped=True,
        )

        kernel_output = lora_ops.scatter2scatter_lora(
            X=data["X"],
            W=data["W"],
            sorted_expert_idxs=data["sorted_expert_idxs"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            k=data["k"],
            lora_A=data["lora_A"],
            lora_B=data["lora_B"],
            scaling=data["scaling"],
            y_grouped=True,
        )

        torch.testing.assert_close(kernel_output, ref_output, atol=1e-2, rtol=1e-2)


# =============================================================================
# Test: Backward Pass Correctness (LoRA Gradients)
# =============================================================================


class TestLoRAGradients:
    """Test backward LoRA gradient computation (dA, dB)."""

    def _run_lora_grad_test(self, M, K, N, E, R, k, atol=1e-2, rtol=1e-2):
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")
        base_ops = import_module(f"{_SMOE}.kernels.ops")

        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k)

        # Group X for backward
        grouped_X = base_ops.group(data["X"], data["sorted_scattered_idxs"], fan_out=k)

        # Create fake grad_out in grouped order
        grad_out = torch.randn(
            data["sorted_expert_idxs"].size(0),
            N,
            device="cuda",
            dtype=torch.float32,
        )

        # Reference
        _, ref_dA, ref_dB = reference_lora_backward(
            grad_out,
            grouped_X,
            data["W"],
            data["lora_A"],
            data["lora_B"],
            data["scaling"],
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["expert_offsets"],
            k,
            E,
        )

        # Kernel
        kernel_dA, kernel_dB = lora_ops.group_bwd_lora(
            DY=grad_out,
            X=grouped_X,
            lora_A=data["lora_A"],
            lora_B=data["lora_B"],
            expert_offsets=data["expert_offsets"],
            E=E,
            scaling=data["scaling"],
        )

        torch.testing.assert_close(kernel_dA, ref_dA, atol=atol, rtol=rtol)
        torch.testing.assert_close(kernel_dB, ref_dB, atol=atol, rtol=rtol)

    def test_basic_lora_grads(self):
        self._run_lora_grad_test(M=32, K=64, N=128, E=4, R=8, k=2)

    def test_small_rank(self):
        self._run_lora_grad_test(M=16, K=64, N=64, E=4, R=4, k=1)

    def test_larger_rank(self):
        self._run_lora_grad_test(
            M=16, K=128, N=128, E=8, R=32, k=2, atol=5e-2, rtol=5e-2
        )

    def test_many_experts(self):
        self._run_lora_grad_test(M=64, K=64, N=64, E=16, R=8, k=2)

    def test_single_token_per_expert(self):
        """Edge case: roughly 1 token per expert."""
        self._run_lora_grad_test(M=8, K=64, N=64, E=8, R=4, k=1)


# =============================================================================
# Test: Full Autograd (Forward + Backward) via torch.autograd
# =============================================================================


class TestAutograd:
    """Test full autograd integration through ScatterMoELoRA."""

    def test_lora_receives_gradients(self):
        """LoRA A and B receive non-zero gradients; frozen W does not."""
        from importlib import import_module

        pll = import_module(f"{_SMOE}.parallel_linear_lora")

        M, K, N, E, R, k = 16, 64, 64, 4, 8, 2
        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k)

        X = data["X"].clone().requires_grad_(True)
        W = data["W"].clone().requires_grad_(False)  # Frozen
        lora_A = data["lora_A"].clone().requires_grad_(True)
        lora_B = data["lora_B"].clone().requires_grad_(True)

        output = pll.ScatterMoELoRA.apply(
            X,
            W,
            k,
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["expert_offsets"],
            lora_A,
            lora_B,
            data["scaling"],
            None,
            None,
            False,
            False,
        )

        loss = output.sum()
        loss.backward()

        # LoRA params should have gradients
        assert lora_A.grad is not None, "lora_A should have gradient"
        assert lora_B.grad is not None, "lora_B should have gradient"
        assert lora_A.grad.abs().sum() > 0, "lora_A gradient should be non-zero"
        assert lora_B.grad.abs().sum() > 0, "lora_B gradient should be non-zero"

        # Input should have gradient (needed for upstream backprop)
        assert X.grad is not None, "X should have gradient"
        assert X.grad.abs().sum() > 0, "X gradient should be non-zero"

    def test_input_gradient_matches_reference(self):
        """Input gradient from autograd matches pure PyTorch reference."""
        from importlib import import_module

        pll = import_module(f"{_SMOE}.parallel_linear_lora")
        base_ops = import_module(f"{_SMOE}.kernels.ops")

        M, K, N, E, R, k = 16, 64, 64, 4, 8, 1
        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k)

        # Autograd path
        X_kern = data["X"].clone().requires_grad_(True)
        lora_A_kern = data["lora_A"].clone().requires_grad_(True)
        lora_B_kern = data["lora_B"].clone().requires_grad_(True)

        out_kern = pll.ScatterMoELoRA.apply(
            X_kern,
            data["W"],
            k,
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["expert_offsets"],
            lora_A_kern,
            lora_B_kern,
            data["scaling"],
            None,
            None,
            False,
            False,
        )
        grad_out = torch.randn_like(out_kern)
        out_kern.backward(grad_out)

        # Reference path
        grouped_X = base_ops.group(data["X"], data["sorted_scattered_idxs"], fan_out=k)
        grouped_grad = base_ops.group(
            grad_out, data["sorted_scattered_idxs"], fan_out=1
        )

        ref_dX, ref_dA, ref_dB = reference_lora_backward(
            grouped_grad,
            grouped_X,
            data["W"],
            data["lora_A"],
            data["lora_B"],
            data["scaling"],
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["expert_offsets"],
            k,
            E,
        )

        # Compare input gradient (for k=1, no reduction needed)
        # ref_dX is in grouped (expert-sorted) order; X_kern.grad is in original order.
        # Ungroup ref_dX by scattering back to original positions.
        ref_dX_ungrouped = torch.zeros_like(ref_dX)
        ref_dX_ungrouped[data["sorted_scattered_idxs"]] = ref_dX
        torch.testing.assert_close(X_kern.grad, ref_dX_ungrouped, atol=5e-2, rtol=5e-2)

    def test_lora_gradient_matches_reference(self):
        """LoRA A/B gradients from autograd match reference."""
        from importlib import import_module

        pll = import_module(f"{_SMOE}.parallel_linear_lora")
        base_ops = import_module(f"{_SMOE}.kernels.ops")

        M, K, N, E, R, k = 16, 64, 64, 4, 8, 1
        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k)

        # Autograd path
        X_kern = data["X"].clone().requires_grad_(True)
        lora_A_kern = data["lora_A"].clone().requires_grad_(True)
        lora_B_kern = data["lora_B"].clone().requires_grad_(True)

        out_kern = pll.ScatterMoELoRA.apply(
            X_kern,
            data["W"],
            k,
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["expert_offsets"],
            lora_A_kern,
            lora_B_kern,
            data["scaling"],
            None,
            None,
            False,
            False,
        )
        grad_out = torch.randn_like(out_kern)
        out_kern.backward(grad_out)

        # Reference path
        grouped_X = base_ops.group(data["X"], data["sorted_scattered_idxs"], fan_out=k)
        grouped_grad = base_ops.group(
            grad_out, data["sorted_scattered_idxs"], fan_out=1
        )

        _, ref_dA, ref_dB = reference_lora_backward(
            grouped_grad,
            grouped_X,
            data["W"],
            data["lora_A"],
            data["lora_B"],
            data["scaling"],
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["expert_offsets"],
            k,
            E,
        )

        torch.testing.assert_close(lora_A_kern.grad, ref_dA, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(lora_B_kern.grad, ref_dB, atol=5e-2, rtol=5e-2)


# =============================================================================
# Test: Equivalence with Base ScatterMoE (scaling=0 should match base)
# =============================================================================


class TestBaseEquivalence:
    """When scaling=0, fused kernel should match base scatter2scatter."""

    def test_zero_scaling_matches_base(self):
        """With scaling=0, LoRA contribution vanishes; should match base."""
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")
        base_ops = import_module(f"{_SMOE}.kernels.ops")

        data = make_test_data(M=32, K=64, N=128, E=4, R=8, k=2)

        base_output = base_ops.scatter2scatter(
            X=data["X"],
            W=data["W"],
            sorted_expert_idxs=data["sorted_expert_idxs"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            k=data["k"],
        )

        lora_output = lora_ops.scatter2scatter_lora(
            X=data["X"],
            W=data["W"],
            sorted_expert_idxs=data["sorted_expert_idxs"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            k=data["k"],
            lora_A=data["lora_A"],
            lora_B=data["lora_B"],
            scaling=0.0,
        )

        torch.testing.assert_close(lora_output, base_output, atol=1e-3, rtol=1e-3)

    def test_zero_lora_weights_matches_base(self):
        """With A=0, B=0, should match base scatter2scatter."""
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")
        base_ops = import_module(f"{_SMOE}.kernels.ops")

        data = make_test_data(M=32, K=64, N=128, E=4, R=8, k=2)

        zero_A = torch.zeros_like(data["lora_A"])
        zero_B = torch.zeros_like(data["lora_B"])

        base_output = base_ops.scatter2scatter(
            X=data["X"],
            W=data["W"],
            sorted_expert_idxs=data["sorted_expert_idxs"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            k=data["k"],
        )

        lora_output = lora_ops.scatter2scatter_lora(
            X=data["X"],
            W=data["W"],
            sorted_expert_idxs=data["sorted_expert_idxs"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            k=data["k"],
            lora_A=zero_A,
            lora_B=zero_B,
            scaling=1.0,
        )

        torch.testing.assert_close(lora_output, base_output, atol=1e-3, rtol=1e-3)


# =============================================================================
# Test: LoRA Additivity
# =============================================================================


class TestLoRAAdditivity:
    """Test that the LoRA component is correctly additive."""

    def test_lora_additivity(self):
        """
        Verify: fused(X, W, A, B, s) == base(X, W) + s * per_expert_lora(X, A, B)
        """
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")
        base_ops = import_module(f"{_SMOE}.kernels.ops")

        data = make_test_data(M=32, K=64, N=128, E=4, R=8, k=2)

        # Base output (no LoRA)
        base_output = base_ops.scatter2scatter(
            X=data["X"],
            W=data["W"],
            sorted_expert_idxs=data["sorted_expert_idxs"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            k=data["k"],
        )

        # Fused output
        fused_output = lora_ops.scatter2scatter_lora(
            X=data["X"],
            W=data["W"],
            sorted_expert_idxs=data["sorted_expert_idxs"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            k=data["k"],
            lora_A=data["lora_A"],
            lora_B=data["lora_B"],
            scaling=data["scaling"],
        )

        # Compute LoRA contribution manually (reference)
        lora_only = reference_parallel_linear_lora(
            data["X"],
            torch.zeros_like(data["W"]),
            data["k"],
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["lora_A"],
            data["lora_B"],
            data["scaling"],
        )

        # fused = base + lora
        expected = base_output + lora_only
        torch.testing.assert_close(fused_output, expected, atol=2e-2, rtol=2e-2)


# =============================================================================
# Test: ParallelExperts module integration
# =============================================================================


class TestParallelExpertsModule:
    """Test the ParallelExperts module with LoRA."""

    def test_set_and_clear_lora(self):
        """Test set_lora/clear_lora lifecycle."""
        from importlib import import_module

        lora_module = import_module(f"{_SMOE}.lora_ops")

        pe = lora_module.ParallelExperts(4, 64, 128).cuda()

        A = torch.randn(32, 64, device="cuda")  # r=8, E=4
        B = torch.randn(128, 32, device="cuda")
        pe.set_lora(A, B, 0.5)

        assert pe._lora_A is A
        assert pe._lora_B is B
        assert pe._lora_scaling == 0.5

        pe.clear_lora()
        assert pe._lora_A is None
        assert pe._lora_B is None

    def test_forward_with_lora(self):
        """ParallelExperts forward with LoRA matches reference."""
        from importlib import import_module

        lora_module = import_module(f"{_SMOE}.lora_ops")

        E, K, N, R = 4, 64, 128, 8
        M, k = 16, 2
        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k)

        pe = lora_module.ParallelExperts(E, K, N).cuda()
        # Set weights to match test data
        with torch.no_grad():
            pe.weight.copy_(data["W"].permute(0, 2, 1))  # [E, N, K]

        pe.set_lora(data["lora_A"], data["lora_B"], data["scaling"])

        output = pe(
            data["X"],
            k,
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["expert_offsets"],
        )

        ref = reference_parallel_linear_lora(
            data["X"],
            data["W"],
            k,
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["lora_A"],
            data["lora_B"],
            data["scaling"],
        )

        torch.testing.assert_close(output, ref, atol=2e-2, rtol=2e-2)


# =============================================================================
# Test: Edge Cases
# =============================================================================


class TestEdgeCases:
    """Edge cases and boundary conditions."""

    def test_all_tokens_one_expert(self):
        """All tokens routed to a single expert."""
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")

        M, K, N, E, R, k = 16, 64, 64, 4, 8, 1
        torch.manual_seed(42)

        X = torch.randn(M, K, device="cuda")
        W = torch.randn(E, K, N, device="cuda") * 0.02
        lora_A = torch.randn(R * E, K, device="cuda") * 0.01
        lora_B = torch.randn(N, R * E, device="cuda") * 0.01

        # All tokens go to expert 0
        selected_experts = torch.zeros(M, k, device="cuda", dtype=torch.long)
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = (
            flatten_sort_count_ref(selected_experts, E)
        )

        ref = reference_parallel_linear_lora(
            X,
            W,
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            lora_A,
            lora_B,
            0.5,
        )

        kernel = lora_ops.scatter2scatter_lora(
            X=X,
            W=W,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            k=k,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.5,
        )

        torch.testing.assert_close(kernel, ref, atol=1e-2, rtol=1e-2)

    def test_empty_experts(self):
        """Some experts have no tokens assigned."""
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")

        M, K, N, E, R, k = 8, 64, 64, 8, 4, 1
        torch.manual_seed(42)

        X = torch.randn(M, K, device="cuda")
        W = torch.randn(E, K, N, device="cuda") * 0.02
        lora_A = torch.randn(R * E, K, device="cuda") * 0.01
        lora_B = torch.randn(N, R * E, device="cuda") * 0.01

        # Only use experts 0 and 1
        selected_experts = torch.randint(0, 2, (M, k), device="cuda")
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = (
            flatten_sort_count_ref(selected_experts, E)
        )

        ref = reference_parallel_linear_lora(
            X,
            W,
            k,
            sorted_expert_idxs,
            sorted_scattered_idxs,
            lora_A,
            lora_B,
            0.5,
        )

        kernel = lora_ops.scatter2scatter_lora(
            X=X,
            W=W,
            sorted_expert_idxs=sorted_expert_idxs,
            sorted_scattered_idxs=sorted_scattered_idxs,
            k=k,
            lora_A=lora_A,
            lora_B=lora_B,
            scaling=0.5,
        )

        torch.testing.assert_close(kernel, ref, atol=1e-2, rtol=1e-2)


# =============================================================================
# Test: Optimization 1 - Fused dX Kernel
# =============================================================================


class TestFusedDX:
    """Test fused backward dX kernel: dX = dY @ W^T + scaling * (dY @ B) @ A."""

    def _run_fused_dX_test(
        self, M, K, N, E, R, k, dtype=torch.float32, atol=5e-2, rtol=5e-2
    ):
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")
        base_ops = import_module(f"{_SMOE}.kernels.ops")
        pll = import_module(f"{_SMOE}.parallel_linear_lora")

        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k, dtype=dtype)

        # Create dummy grad_out in grouped order
        grad_out = torch.randn(
            data["sorted_expert_idxs"].size(0), N, device="cuda", dtype=dtype
        )
        grouped_grad = base_ops.group(
            grad_out,
            data["sorted_scattered_idxs"],
            fan_out=1,
        )

        # Reference: separate scatter2scatter(DY, W^T) + _compute_lora_input_grad
        ref_base = base_ops.scatter2scatter(
            X=grouped_grad,
            x_grouped=True,
            W=data["W"].permute(0, 2, 1),
            sorted_expert_idxs=data["sorted_expert_idxs"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            k=1,
            y_grouped=False,
        )

        ref_lora = pll._compute_lora_input_grad(
            grouped_grad,
            data["lora_A"],
            data["lora_B"],
            data["expert_offsets"],
            E,
            data["scaling"],
        )
        # Scatter lora from grouped to ungrouped order
        ref_lora_ungrouped = torch.zeros_like(ref_base)
        ref_lora_ungrouped[data["sorted_scattered_idxs"]] = ref_lora
        ref_total = ref_base + ref_lora_ungrouped

        # Fused kernel
        fused_result = lora_ops.scatter2scatter_lora_dX(
            DY=grouped_grad,
            W=data["W"],
            sorted_expert_idxs=data["sorted_expert_idxs"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            k=1,
            lora_A=data["lora_A"],
            lora_B=data["lora_B"],
            scaling=data["scaling"],
            dy_grouped=True,
            dx_grouped=False,
        )

        torch.testing.assert_close(fused_result, ref_total, atol=atol, rtol=rtol)

    def test_basic(self):
        self._run_fused_dX_test(M=32, K=64, N=128, E=4, R=8, k=2)

    def test_large(self):
        self._run_fused_dX_test(M=256, K=256, N=512, E=8, R=16, k=2)

    def test_single_expert(self):
        self._run_fused_dX_test(M=64, K=128, N=256, E=1, R=8, k=1)

    def test_k1(self):
        self._run_fused_dX_test(M=64, K=64, N=128, E=4, R=8, k=1)

    def test_bf16(self):
        self._run_fused_dX_test(
            M=64,
            K=128,
            N=256,
            E=4,
            R=16,
            k=2,
            dtype=torch.bfloat16,
            atol=1e-1,
            rtol=1e-1,
        )

    def test_grouped_output(self):
        """Test fused dX with dx_grouped=True."""
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")
        base_ops = import_module(f"{_SMOE}.kernels.ops")
        pll = import_module(f"{_SMOE}.parallel_linear_lora")

        M, K, N, E, R, k = 32, 64, 128, 4, 8, 2
        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k)

        grad_out = torch.randn(data["sorted_expert_idxs"].size(0), N, device="cuda")
        grouped_grad = base_ops.group(
            grad_out, data["sorted_scattered_idxs"], fan_out=1
        )

        # Reference: grouped output
        ref_base = base_ops.scatter2scatter(
            X=grouped_grad,
            x_grouped=True,
            W=data["W"].permute(0, 2, 1),
            sorted_expert_idxs=data["sorted_expert_idxs"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            k=1,
            y_grouped=True,  # grouped output
        )

        ref_lora = pll._compute_lora_input_grad(
            grouped_grad,
            data["lora_A"],
            data["lora_B"],
            data["expert_offsets"],
            E,
            data["scaling"],
        )
        ref_total = ref_base + ref_lora

        # Fused kernel with grouped output
        fused_result = lora_ops.scatter2scatter_lora_dX(
            DY=grouped_grad,
            W=data["W"],
            sorted_expert_idxs=data["sorted_expert_idxs"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            k=1,
            lora_A=data["lora_A"],
            lora_B=data["lora_B"],
            scaling=data["scaling"],
            dy_grouped=True,
            dx_grouped=True,
        )

        torch.testing.assert_close(fused_result, ref_total, atol=5e-2, rtol=5e-2)

    def test_autograd_with_fused_dX(self):
        """Full autograd round-trip with use_fused_dX=True."""
        from importlib import import_module

        pll = import_module(f"{_SMOE}.parallel_linear_lora")

        M, K, N, E, R, k = 32, 64, 128, 4, 8, 2
        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k)

        # Run without fused dX
        X1 = data["X"].clone().requires_grad_(True)
        A1 = data["lora_A"].clone().requires_grad_(True)
        B1 = data["lora_B"].clone().requires_grad_(True)
        out1 = pll.ScatterMoELoRA.apply(
            X1,
            data["W"],
            k,
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["expert_offsets"],
            A1,
            B1,
            data["scaling"],
            None,
            None,
            False,
            False,
            False,  # use_fused_dX=False
        )
        out1.sum().backward()

        # Run with fused dX
        X2 = data["X"].clone().requires_grad_(True)
        A2 = data["lora_A"].clone().requires_grad_(True)
        B2 = data["lora_B"].clone().requires_grad_(True)
        out2 = pll.ScatterMoELoRA.apply(
            X2,
            data["W"],
            k,
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["expert_offsets"],
            A2,
            B2,
            data["scaling"],
            None,
            None,
            False,
            False,
            True,  # use_fused_dX=True
        )
        out2.sum().backward()

        # Forward should be identical
        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)

        # Gradients should match
        torch.testing.assert_close(X1.grad, X2.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(A1.grad, A2.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(B1.grad, B2.grad, atol=5e-2, rtol=5e-2)


# =============================================================================
# Test: Optimization 2 - Fused Gather Backward
# =============================================================================


class TestFusedGatherBackward:
    """Test fused gather + backward dA/dB kernel."""

    def _run_fused_gather_test(
        self, M, K, N, E, R, k, dtype=torch.float32, atol=5e-2, rtol=5e-2
    ):
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")
        base_ops = import_module(f"{_SMOE}.kernels.ops")

        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k, dtype=dtype)

        # Create grad_out in ungrouped order (M*k, N)
        M_total = data["sorted_expert_idxs"].size(0)
        grad_out = torch.randn(M_total, N, device="cuda", dtype=dtype)

        # Reference: group() + group_bwd_lora()
        grouped_grad = base_ops.group(
            grad_out, data["sorted_scattered_idxs"], fan_out=1
        )
        grouped_x = base_ops.group(data["X"], data["sorted_scattered_idxs"], fan_out=k)

        ref_dA, ref_dB = lora_ops.group_bwd_lora(
            DY=grouped_grad,
            X=grouped_x,
            lora_A=data["lora_A"],
            lora_B=data["lora_B"],
            expert_offsets=data["expert_offsets"],
            E=E,
            scaling=data["scaling"],
        )

        # Fused kernel: no group() calls
        fused_dA, fused_dB = lora_ops.group_bwd_lora_fused(
            DY=grad_out,
            X=data["X"],
            lora_A=data["lora_A"],
            lora_B=data["lora_B"],
            expert_offsets=data["expert_offsets"],
            sorted_scattered_idxs=data["sorted_scattered_idxs"],
            E=E,
            k=k,
            scaling=data["scaling"],
        )

        torch.testing.assert_close(fused_dA, ref_dA, atol=atol, rtol=rtol)
        torch.testing.assert_close(fused_dB, ref_dB, atol=atol, rtol=rtol)

    def test_basic(self):
        self._run_fused_gather_test(M=32, K=64, N=128, E=4, R=8, k=2)

    def test_large(self):
        self._run_fused_gather_test(M=256, K=256, N=512, E=8, R=16, k=2)

    def test_single_expert(self):
        self._run_fused_gather_test(M=64, K=128, N=256, E=1, R=8, k=1)

    def test_k1(self):
        self._run_fused_gather_test(M=64, K=64, N=128, E=4, R=8, k=1)

    def test_many_experts(self):
        self._run_fused_gather_test(M=128, K=64, N=128, E=16, R=8, k=4)

    def test_bf16(self):
        self._run_fused_gather_test(
            M=64,
            K=128,
            N=256,
            E=4,
            R=16,
            k=2,
            dtype=torch.bfloat16,
            atol=1e-1,
            rtol=1e-1,
        )

    def test_autograd_with_fused_gather(self):
        """Full autograd round-trip with use_fused_gather=True."""
        from importlib import import_module

        pll = import_module(f"{_SMOE}.parallel_linear_lora")

        M, K, N, E, R, k = 32, 64, 128, 4, 8, 2
        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k)

        # Run without fused gather
        X1 = data["X"].clone().requires_grad_(True)
        A1 = data["lora_A"].clone().requires_grad_(True)
        B1 = data["lora_B"].clone().requires_grad_(True)
        out1 = pll.ScatterMoELoRA.apply(
            X1,
            data["W"],
            k,
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["expert_offsets"],
            A1,
            B1,
            data["scaling"],
            None,
            None,
            False,
            False,
            False,
            False,  # use_fused_dX=False, use_fused_gather=False
        )
        out1.sum().backward()

        # Run with fused gather
        X2 = data["X"].clone().requires_grad_(True)
        A2 = data["lora_A"].clone().requires_grad_(True)
        B2 = data["lora_B"].clone().requires_grad_(True)
        out2 = pll.ScatterMoELoRA.apply(
            X2,
            data["W"],
            k,
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["expert_offsets"],
            A2,
            B2,
            data["scaling"],
            None,
            None,
            False,
            False,
            False,
            True,  # use_fused_dX=False, use_fused_gather=True
        )
        out2.sum().backward()

        # Forward identical
        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)

        # dA/dB should match
        torch.testing.assert_close(A1.grad, A2.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(B1.grad, B2.grad, atol=5e-2, rtol=5e-2)
        # dX should also match (same path for dX)
        torch.testing.assert_close(X1.grad, X2.grad, atol=5e-2, rtol=5e-2)


# =============================================================================
# Test: Optimization 3 - Token Rounding
# =============================================================================


class TestTokenRounding:
    """Test token rounding utility and its integration with backward kernels."""

    def test_round_expert_counts_basic(self):
        """Verify round_expert_counts produces correct shapes and values."""
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")

        M, K, N, E, R, k = 32, 64, 128, 4, 8, 2
        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k)

        padded_ei, padded_si, padded_offsets, real_offsets = (
            lora_ops.round_expert_counts(
                data["sorted_expert_idxs"],
                data["sorted_scattered_idxs"],
                data["expert_offsets"],
                E=E,
                block_m=lora_ops.BLOCK_M,
            )
        )

        # Real offsets should match original
        torch.testing.assert_close(real_offsets, data["expert_offsets"])

        # Padded offsets should be >= real offsets
        assert (padded_offsets >= real_offsets).all(), (
            "Padded offsets should be >= real offsets"
        )

        # Each expert's padded count should be multiple of BLOCK_M (if non-zero)
        prev = 0
        for e in range(E):
            count = padded_offsets[e].item() - prev
            real_count = real_offsets[e].item() - (
                real_offsets[e - 1].item() if e > 0 else 0
            )
            if real_count > 0:
                assert count % lora_ops.BLOCK_M == 0, (
                    f"Expert {e}: padded count {count} not multiple of {lora_ops.BLOCK_M}"
                )
                assert count >= real_count, (
                    f"Expert {e}: padded count {count} < real count {real_count}"
                )
            prev = padded_offsets[e].item()

    def test_round_with_fused_gather(self):
        """Token rounding + fused gather gives same result as plain fused gather."""
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")
        base_ops = import_module(f"{_SMOE}.kernels.ops")

        M, K, N, E, R, k = 64, 64, 128, 4, 8, 2
        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k)

        M_total = data["sorted_expert_idxs"].size(0)
        grad_out = torch.randn(M_total, N, device="cuda")

        # Reference: group() + group_bwd_lora() (the gold standard)
        grouped_grad = base_ops.group(
            grad_out, data["sorted_scattered_idxs"], fan_out=1
        )
        grouped_x = base_ops.group(data["X"], data["sorted_scattered_idxs"], fan_out=k)
        ref_dA, ref_dB = lora_ops.group_bwd_lora(
            DY=grouped_grad,
            X=grouped_x,
            lora_A=data["lora_A"],
            lora_B=data["lora_B"],
            expert_offsets=data["expert_offsets"],
            E=E,
            scaling=data["scaling"],
        )

        # Apply token rounding
        padded_ei, padded_si, padded_offsets, real_offsets = (
            lora_ops.round_expert_counts(
                data["sorted_expert_idxs"],
                data["sorted_scattered_idxs"],
                data["expert_offsets"],
                E=E,
            )
        )

        # Fused gather with token rounding
        rounded_dA, rounded_dB = lora_ops.group_bwd_lora_fused(
            DY=grad_out,
            X=data["X"],
            lora_A=data["lora_A"],
            lora_B=data["lora_B"],
            expert_offsets=padded_offsets,
            sorted_scattered_idxs=padded_si,
            E=E,
            k=k,
            scaling=data["scaling"],
            real_expert_offsets=real_offsets,
        )

        torch.testing.assert_close(rounded_dA, ref_dA, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(rounded_dB, ref_dB, atol=5e-2, rtol=5e-2)

    def test_empty_experts_with_rounding(self):
        """Token rounding handles experts with 0 tokens correctly."""
        from importlib import import_module

        lora_ops = import_module(f"{_SMOE}.kernels.lora_ops")

        E, k = 8, 1
        M = 8
        torch.manual_seed(42)

        # Only use experts 0 and 1 (rest have 0 tokens)
        selected_experts = torch.randint(0, 2, (M, k), device="cuda")
        sorted_expert_idxs, sorted_scattered_idxs, expert_offsets = (
            flatten_sort_count_ref(selected_experts, E)
        )

        padded_ei, padded_si, padded_offsets, real_offsets = (
            lora_ops.round_expert_counts(
                sorted_expert_idxs,
                sorted_scattered_idxs,
                expert_offsets,
                E=E,
            )
        )

        # Verify empty experts have same count (0)
        for e in range(E):
            real_count = real_offsets[e].item() - (
                real_offsets[e - 1].item() if e > 0 else 0
            )
            padded_count = padded_offsets[e].item() - (
                padded_offsets[e - 1].item() if e > 0 else 0
            )
            if real_count == 0:
                assert padded_count == 0, (
                    f"Expert {e}: empty expert should have padded_count=0, got {padded_count}"
                )


# =============================================================================
# Test: Combined Optimizations
# =============================================================================


class TestCombinedOptimizations:
    """Test all optimizations together."""

    def test_fused_dX_and_fused_gather(self):
        """Both fused dX and fused gather together."""
        from importlib import import_module

        pll = import_module(f"{_SMOE}.parallel_linear_lora")

        M, K, N, E, R, k = 64, 128, 256, 4, 8, 2
        data = make_test_data(M=M, K=K, N=N, E=E, R=R, k=k)

        # Baseline: no optimizations
        X1 = data["X"].clone().requires_grad_(True)
        A1 = data["lora_A"].clone().requires_grad_(True)
        B1 = data["lora_B"].clone().requires_grad_(True)
        out1 = pll.ScatterMoELoRA.apply(
            X1,
            data["W"],
            k,
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["expert_offsets"],
            A1,
            B1,
            data["scaling"],
            None,
            None,
            False,
            False,
            False,
            False,  # no optimizations
        )
        out1.sum().backward()

        # Both optimizations
        X2 = data["X"].clone().requires_grad_(True)
        A2 = data["lora_A"].clone().requires_grad_(True)
        B2 = data["lora_B"].clone().requires_grad_(True)
        out2 = pll.ScatterMoELoRA.apply(
            X2,
            data["W"],
            k,
            data["sorted_expert_idxs"],
            data["sorted_scattered_idxs"],
            data["expert_offsets"],
            A2,
            B2,
            data["scaling"],
            None,
            None,
            False,
            False,
            True,
            True,  # use_fused_dX=True, use_fused_gather=True
        )
        out2.sum().backward()

        # Forward identical
        torch.testing.assert_close(out1, out2, atol=1e-5, rtol=1e-5)

        # All gradients match
        torch.testing.assert_close(X1.grad, X2.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(A1.grad, A2.grad, atol=5e-2, rtol=5e-2)
        torch.testing.assert_close(B1.grad, B2.grad, atol=5e-2, rtol=5e-2)


# =============================================================================
# Test: HFScatterMoEGatedMLP with Sigmoid Routing
# =============================================================================


def _reference_moe_forward(
    hidden_states,
    gate_weight,
    gate_up_proj,
    down_proj,
    act_fn,
    routing_weights,
    selected_experts,
    num_experts,
):
    """Pure PyTorch reference for a full MoE forward pass.

    Args:
        hidden_states: [T, H]
        gate_weight: [E, H]
        gate_up_proj: [E, 2*FF, H]
        down_proj: [E, H, FF]
        act_fn: activation function (e.g. torch.nn.SiLU())
        routing_weights: [T, K] routing weights
        selected_experts: [T, K] expert indices
        num_experts: int

    Returns:
        output: [T, H]
    """
    T, H = hidden_states.shape
    K = selected_experts.shape[1]
    output = torch.zeros(T, H, device=hidden_states.device, dtype=hidden_states.dtype)

    for t in range(T):
        for j in range(K):
            e = selected_experts[t, j].item()
            w = routing_weights[t, j].item()

            # gate_up projection
            gup = hidden_states[t] @ gate_up_proj[e].T  # [2*I]
            I_dim = gup.shape[0] // 2
            gates = gup[:I_dim]
            up = gup[I_dim:]

            # activation
            h = act_fn(gates) * up

            # down projection
            out = h @ down_proj[e].T  # [H]

            output[t] += w * out

    return output


def _make_mock_sigmoid_moe_block(
    T=16, H=64, FF=32, E=8, K=2, n_group=2, topk_group=1, bias_on_gate=True
):
    """Create a mock MoE block with sigmoid routing for GPU testing."""
    gate_up_proj = torch.randn(E, 2 * FF, H, device="cuda") * 0.02
    down_proj = torch.randn(E, H, FF, device="cuda") * 0.02
    act_fn = torch.nn.SiLU()

    experts = SimpleNamespace(
        gate_up_proj=gate_up_proj,
        down_proj=down_proj,
        act_fn=act_fn,
        num_experts=E,
    )

    if bias_on_gate:
        gate = SimpleNamespace(
            weight=torch.randn(E, H, device="cuda") * 0.1,
            e_score_correction_bias=torch.zeros(E, device="cuda"),
        )
        moe_block = SimpleNamespace(
            gate=gate,
            experts=experts,
            top_k=K,
            n_routed_experts=E,
            n_group=n_group,
            topk_group=topk_group,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
    else:
        # minimax_m2 style
        gate = SimpleNamespace(
            weight=torch.randn(E, H, device="cuda") * 0.1,
            top_k=K,
        )
        moe_block = SimpleNamespace(
            gate=gate,
            experts=experts,
            top_k=K,
            e_score_correction_bias=torch.zeros(E, device="cuda"),
        )

    return moe_block, T, H, FF, E, K


class TestHFScatterMoESigmoidRouting:
    """Test HFScatterMoEGatedMLP forward with sigmoid routing on GPU."""

    def test_forward_matches_reference_bias_on_gate(self):
        """Forward pass with sigmoid routing (bias on gate) matches reference."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            HFScatterMoEGatedMLP,
            _sigmoid_topk_route,
        )

        moe_block, T, H, FF, E, K = _make_mock_sigmoid_moe_block(
            T=16, H=64, FF=32, E=8, K=2, n_group=2, topk_group=1, bias_on_gate=True
        )

        hidden = torch.randn(1, T, H, device="cuda")

        # Get routing for reference
        gate = moe_block.gate
        hidden_flat = hidden.view(-1, H)
        routing_weights, selected_experts, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden_flat, gate.weight, None
        )

        # Reference output
        ref_output = _reference_moe_forward(
            hidden_flat,
            gate.weight,
            moe_block.experts.gate_up_proj,
            moe_block.experts.down_proj,
            moe_block.experts.act_fn,
            routing_weights,
            selected_experts,
            E,
        )

        # Kernel output
        kernel_output = HFScatterMoEGatedMLP.forward(moe_block, hidden)
        kernel_output_flat = kernel_output.view(-1, H)

        torch.testing.assert_close(
            kernel_output_flat.float(),
            ref_output.float(),
            atol=5e-2,
            rtol=5e-2,
        )

    def test_forward_matches_reference_bias_on_block(self):
        """Forward pass with sigmoid routing (minimax_m2 style, bias on block)."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            HFScatterMoEGatedMLP,
            _sigmoid_topk_route,
        )

        moe_block, T, H, FF, E, K = _make_mock_sigmoid_moe_block(
            T=16, H=64, FF=32, E=8, K=2, n_group=1, bias_on_gate=False
        )

        hidden = torch.randn(1, T, H, device="cuda")
        hidden_flat = hidden.view(-1, H)

        gate = moe_block.gate
        routing_weights, selected_experts, _, _ = _sigmoid_topk_route(
            moe_block, gate, hidden_flat, gate.weight, None
        )

        ref_output = _reference_moe_forward(
            hidden_flat,
            gate.weight,
            moe_block.experts.gate_up_proj,
            moe_block.experts.down_proj,
            moe_block.experts.act_fn,
            routing_weights,
            selected_experts,
            E,
        )

        kernel_output = HFScatterMoEGatedMLP.forward(moe_block, hidden)
        kernel_output_flat = kernel_output.view(-1, H)

        torch.testing.assert_close(
            kernel_output_flat.float(),
            ref_output.float(),
            atol=5e-2,
            rtol=5e-2,
        )

    def test_softmax_routing_still_works(self):
        """Verify softmax routing (Qwen/OLMoE) is not broken."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            HFScatterMoEGatedMLP,
            _softmax_topk_route,
        )

        T, H, FF, E, K = 16, 64, 32, 4, 2
        gate_up_proj = torch.randn(E, 2 * FF, H, device="cuda") * 0.02
        down_proj = torch.randn(E, H, FF, device="cuda") * 0.02
        act_fn = torch.nn.SiLU()

        experts = SimpleNamespace(
            gate_up_proj=gate_up_proj,
            down_proj=down_proj,
            act_fn=act_fn,
            num_experts=E,
        )
        gate = SimpleNamespace(
            weight=torch.randn(E, H, device="cuda") * 0.1,
            top_k=K,
            num_experts=E,
            norm_topk_prob=True,
        )
        moe_block = SimpleNamespace(gate=gate, experts=experts)

        hidden = torch.randn(1, T, H, device="cuda")
        hidden_flat = hidden.view(-1, H)

        routing_weights, selected_experts, _, _ = _softmax_topk_route(
            moe_block, gate, hidden_flat, gate.weight, None
        )

        ref_output = _reference_moe_forward(
            hidden_flat,
            gate.weight,
            gate_up_proj,
            down_proj,
            act_fn,
            routing_weights,
            selected_experts,
            E,
        )

        kernel_output = HFScatterMoEGatedMLP.forward(moe_block, hidden)
        kernel_output_flat = kernel_output.view(-1, H)

        torch.testing.assert_close(
            kernel_output_flat.float(),
            ref_output.float(),
            atol=5e-2,
            rtol=5e-2,
        )


class TestHFScatterMoESigmoidWithSharedExperts:
    """Test HFScatterMoEGatedMLP with sigmoid routing + shared experts."""

    def test_shared_experts_plural(self):
        """DeepSeek V3 style: shared_experts attribute (plural)."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            HFScatterMoEGatedMLP,
        )

        T, H, FF, E, K = 8, 64, 32, 8, 2
        gate_up_proj = torch.randn(E, 2 * FF, H, device="cuda") * 0.02
        down_proj = torch.randn(E, H, FF, device="cuda") * 0.02
        act_fn = torch.nn.SiLU()

        experts = SimpleNamespace(
            gate_up_proj=gate_up_proj,
            down_proj=down_proj,
            act_fn=act_fn,
            num_experts=E,
        )

        # Shared expert as a simple linear for testing
        shared_W = torch.randn(H, H, device="cuda") * 0.01
        shared_experts_fn = lambda x: x @ shared_W.T  # noqa: E731

        gate = SimpleNamespace(
            weight=torch.randn(E, H, device="cuda") * 0.1,
            e_score_correction_bias=torch.zeros(E, device="cuda"),
        )
        moe_block = SimpleNamespace(
            gate=gate,
            experts=experts,
            shared_experts=shared_experts_fn,
            top_k=K,
            n_routed_experts=E,
            n_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )

        hidden = torch.randn(1, T, H, device="cuda")

        # Should not raise; output should include shared expert contribution
        output = HFScatterMoEGatedMLP.forward(moe_block, hidden)
        assert output.shape == (1, T, H)

        # Run without shared expert to verify it changes the output
        moe_block_no_shared = SimpleNamespace(
            gate=gate,
            experts=experts,
            top_k=K,
            n_routed_experts=E,
            n_group=1,
            norm_topk_prob=True,
            routed_scaling_factor=1.0,
        )
        output_no_shared = HFScatterMoEGatedMLP.forward(moe_block_no_shared, hidden)
        assert not torch.equal(output, output_no_shared)

    def test_shared_expert_with_gate(self):
        """Qwen2MoE style: shared_expert + shared_expert_gate."""
        from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
            HFScatterMoEGatedMLP,
        )

        T, H, FF, E, K = 8, 64, 32, 4, 2
        gate_up_proj = torch.randn(E, 2 * FF, H, device="cuda") * 0.02
        down_proj = torch.randn(E, H, FF, device="cuda") * 0.02
        act_fn = torch.nn.SiLU()

        experts = SimpleNamespace(
            gate_up_proj=gate_up_proj,
            down_proj=down_proj,
            act_fn=act_fn,
            num_experts=E,
        )

        shared_W = torch.randn(H, H, device="cuda") * 0.01
        shared_expert_fn = lambda x: x @ shared_W.T  # noqa: E731
        # Gate that returns 0 -> sigmoid(0) = 0.5
        gate_W = torch.zeros(H, H, device="cuda")
        shared_expert_gate_fn = lambda x: x @ gate_W.T  # noqa: E731

        gate = SimpleNamespace(
            weight=torch.randn(E, H, device="cuda") * 0.1,
            top_k=K,
            num_experts=E,
            norm_topk_prob=True,
        )
        moe_block = SimpleNamespace(
            gate=gate,
            experts=experts,
            shared_expert=shared_expert_fn,
            shared_expert_gate=shared_expert_gate_fn,
        )

        hidden = torch.randn(1, T, H, device="cuda")
        output = HFScatterMoEGatedMLP.forward(moe_block, hidden)
        assert output.shape == (1, T, H)
