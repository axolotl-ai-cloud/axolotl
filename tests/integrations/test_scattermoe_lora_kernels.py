# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
Unit tests for ScatterMoE LoRA Triton kernels.

Tests correctness of:
  - scatter2scatter_lora (forward)
  - scatter2scatter_lora_dX (backward input gradient)
  - group_bwd_lora (backward LoRA weight gradients via split dA/dB)
  - ScatterMoELoRA autograd function (full forward + backward)

Each kernel is tested against a pure PyTorch per-expert-loop reference
implementation at multiple model shapes and LoRA ranks.
"""

import pytest
import torch

from axolotl.integrations.kernels.libs.scattermoe_lora.kernels import (
    lora_ops,
    ops as base_ops,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_experts import (
    flatten_sort_count,
)
from axolotl.integrations.kernels.libs.scattermoe_lora.parallel_linear_lora import (
    ScatterMoELoRA,
)

DEVICE = "cuda"
DTYPE = torch.bfloat16


def _requires_cuda():
    return pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )


pytestmark = _requires_cuda()


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _setup(E, K, N, T, top_k, R, seed=42):
    """Create synthetic expert weights, LoRA, routing, and grouped inputs."""
    torch.manual_seed(seed)
    x = torch.randn(T, K, device=DEVICE, dtype=DTYPE)
    W = torch.randn(E, K, N, device=DEVICE, dtype=DTYPE) * 0.02
    lora_A = torch.randn(R * E, K, device=DEVICE, dtype=DTYPE) * 0.01
    lora_B = torch.randn(N, R * E, device=DEVICE, dtype=DTYPE) * 0.01
    logits = torch.randn(T, E, device=DEVICE)
    _, top_idx = torch.topk(torch.softmax(logits, dim=-1), top_k, dim=-1)
    sei, ssi, eo = flatten_sort_count(top_idx, E)
    return x, W, lora_A, lora_B, sei, ssi, eo


def _reference_fwd(x, W, sei, ssi, eo, k, lora_A, lora_B, scaling, E):
    """Per-expert loop reference: Y = X@W + scaling*(X@A^T)@B^T."""
    grouped_x = base_ops.group(x, ssi, fan_out=k)
    M, N = grouped_x.size(0), W.size(2)
    R = lora_A.size(0) // E
    out = torch.zeros(M, N, device=DEVICE, dtype=DTYPE)
    for e in range(E):
        s = eo[e - 1].item() if e > 0 else 0
        end = eo[e].item()
        if s == end:
            continue
        xe = grouped_x[s:end].float()
        we = W[e].float()
        ae = lora_A[e * R : (e + 1) * R].float()
        be = lora_B[:, e * R : (e + 1) * R].float()
        out[s:end] = (xe @ we + scaling * (xe @ ae.T) @ be.T).to(DTYPE)
    result = torch.zeros(M, N, device=DEVICE, dtype=DTYPE)
    result[ssi] = out
    return result


def _reference_dX(dy_grouped, W, sei, ssi, eo, lora_A, lora_B, scaling, E):
    """Per-expert loop reference: dX = dY@W^T + scaling*(dY@B)@A."""
    M, K = dy_grouped.size(0), W.size(1)
    R = lora_A.size(0) // E
    out = torch.zeros(M, K, device=DEVICE, dtype=DTYPE)
    for e in range(E):
        s = eo[e - 1].item() if e > 0 else 0
        end = eo[e].item()
        if s == end:
            continue
        dye = dy_grouped[s:end].float()
        we = W[e].float()
        ae = lora_A[e * R : (e + 1) * R].float()
        be = lora_B[:, e * R : (e + 1) * R].float()
        out[s:end] = (dye @ we.T + scaling * (dye @ be) @ ae).to(DTYPE)
    result = torch.zeros(M, K, device=DEVICE, dtype=DTYPE)
    result[ssi] = out
    return result


def _reference_bwd_lora(dy, grouped_x, lora_A, lora_B, eo, E, scaling):
    """Per-expert loop reference: dA, dB for LoRA weight gradients."""
    R = lora_A.size(0) // E
    dA = torch.zeros_like(lora_A)
    dB = torch.zeros_like(lora_B)
    for e in range(E):
        s = eo[e - 1].item() if e > 0 else 0
        end = eo[e].item()
        if s == end:
            continue
        xe = grouped_x[s:end].float()
        dye = dy[s:end].float()
        ae = lora_A[e * R : (e + 1) * R].float()
        be = lora_B[:, e * R : (e + 1) * R].float()
        dA[e * R : (e + 1) * R] = (scaling * (dye @ be).T @ xe).to(DTYPE)
        dB[:, e * R : (e + 1) * R] = (scaling * dye.T @ (xe @ ae.T)).to(DTYPE)
    return dA, dB


# ─── Model shape configs ────────────────────────────────────────────────────

# (E, K, N, T, top_k, R, description)
CONFIGS_SMALL = [
    (32, 128, 64, 64, 2, 4, "tiny"),
    (64, 256, 128, 128, 4, 8, "small"),
]

CONFIGS_REAL = [
    (256, 2048, 1024, 2048, 8, 16, "qwen35_gate_up"),
    (256, 512, 2048, 2048, 8, 16, "qwen35_down"),
    (64, 2048, 2048, 2048, 8, 16, "olmoe_gate_up"),
    (128, 2048, 1536, 2048, 8, 16, "qwen3_gate_up"),
]

SCALING = 2.0


# ─── Forward tests ──────────────────────────────────────────────────────────


class TestScatter2ScatterLoRAForward:
    """Test scatter2scatter_lora forward kernel vs reference."""

    @pytest.fixture(params=CONFIGS_SMALL + CONFIGS_REAL)
    def config(self, request):
        return request.param

    def test_matches_reference(self, config):
        E, K, N, T, k, R, desc = config
        x, W, lA, lB, sei, ssi, eo = _setup(E, K, N, T, k, R)

        kernel_out = lora_ops.scatter2scatter_lora(
            X=x,
            W=W,
            sorted_expert_idxs=sei,
            sorted_scattered_idxs=ssi,
            k=k,
            lora_A=lA,
            lora_B=lB,
            scaling=SCALING,
        )
        ref_out = _reference_fwd(x, W, sei, ssi, eo, k, lA, lB, SCALING, E)

        err = (kernel_out.float() - ref_out.float()).abs().max().item()
        assert err < 1.0, f"[{desc}] fwd max_err={err}"

    def test_output_shape(self, config):
        E, K, N, T, k, R, desc = config
        x, W, lA, lB, sei, ssi, eo = _setup(E, K, N, T, k, R)

        out = lora_ops.scatter2scatter_lora(
            X=x,
            W=W,
            sorted_expert_idxs=sei,
            sorted_scattered_idxs=ssi,
            k=k,
            lora_A=lA,
            lora_B=lB,
            scaling=SCALING,
        )
        assert out.shape == (T * k, N)
        assert out.dtype == DTYPE


# ─── Backward dX tests ──────────────────────────────────────────────────────


class TestScatter2ScatterLoRADX:
    """Test scatter2scatter_lora_dX backward kernel vs reference."""

    @pytest.fixture(params=CONFIGS_SMALL + CONFIGS_REAL)
    def config(self, request):
        return request.param

    def test_matches_reference(self, config):
        E, K, N, T, k, R, desc = config
        x, W, lA, lB, sei, ssi, eo = _setup(E, K, N, T, k, R)
        gx = base_ops.group(x, ssi, fan_out=k)
        dy = torch.randn(gx.size(0), N, device=DEVICE, dtype=DTYPE)

        kernel_dx = lora_ops.scatter2scatter_lora_dX(
            DY=dy,
            W=W,
            sorted_expert_idxs=sei,
            sorted_scattered_idxs=ssi,
            k=1,
            lora_A=lA,
            lora_B=lB,
            scaling=SCALING,
            dy_grouped=True,
            dx_grouped=False,
        )
        ref_dx = _reference_dX(dy, W, sei, ssi, eo, lA, lB, SCALING, E)

        err = (kernel_dx.float() - ref_dx.float()).abs().max().item()
        assert err < 1.0, f"[{desc}] dX max_err={err}"


# ─── Backward LoRA gradient tests ───────────────────────────────────────────


class TestGroupBwdLoRA:
    """Test group_bwd_lora (split dA/dB kernel) vs reference."""

    @pytest.fixture(params=CONFIGS_SMALL + CONFIGS_REAL)
    def config(self, request):
        return request.param

    def test_matches_reference(self, config):
        E, K, N, T, k, R, desc = config
        x, W, lA, lB, sei, ssi, eo = _setup(E, K, N, T, k, R)
        gx = base_ops.group(x, ssi, fan_out=k)
        dy = torch.randn(gx.size(0), N, device=DEVICE, dtype=DTYPE)

        kern_dA, kern_dB = lora_ops.group_bwd_lora(
            DY=dy,
            X=gx,
            lora_A=lA,
            lora_B=lB,
            expert_offsets=eo,
            E=E,
            scaling=SCALING,
        )
        ref_dA, ref_dB = _reference_bwd_lora(dy, gx, lA, lB, eo, E, SCALING)

        # Use norm-relative error: bf16 accumulation order differs between
        # kernel (tiled + different reduction order) and reference (per-expert
        # fp32 loop), so max absolute error can be large on individual elements
        # while the overall tensor is correct.
        dA_norm_err = (
            (kern_dA.float() - ref_dA.float()).norm() / (ref_dA.float().norm() + 1e-6)
        ).item()
        dB_norm_err = (
            (kern_dB.float() - ref_dB.float()).norm() / (ref_dB.float().norm() + 1e-6)
        ).item()
        assert dA_norm_err < 0.01, f"[{desc}] dA norm_rel_err={dA_norm_err}"
        assert dB_norm_err < 0.01, f"[{desc}] dB norm_rel_err={dB_norm_err}"

    def test_zero_expert_tokens(self):
        """Experts with zero routed tokens produce zero gradients."""
        E, K, N, R = 8, 64, 32, 4
        torch.manual_seed(42)
        # Route all tokens to expert 0 only
        T, k = 16, 1
        top_idx = torch.zeros(T, k, dtype=torch.long, device=DEVICE)
        sei, ssi, eo = flatten_sort_count(top_idx, E)
        gx = torch.randn(T, K, device=DEVICE, dtype=DTYPE)
        dy = torch.randn(T, N, device=DEVICE, dtype=DTYPE)
        lA = torch.randn(R * E, K, device=DEVICE, dtype=DTYPE)
        lB = torch.randn(N, R * E, device=DEVICE, dtype=DTYPE)

        dA, dB = lora_ops.group_bwd_lora(
            DY=dy,
            X=gx,
            lora_A=lA,
            lora_B=lB,
            expert_offsets=eo,
            E=E,
            scaling=2.0,
        )

        # Experts 1..7 should have zero gradients
        for e in range(1, E):
            assert dA[e * R : (e + 1) * R].abs().max() == 0, f"Expert {e} dA not zero"
            assert dB[:, e * R : (e + 1) * R].abs().max() == 0, (
                f"Expert {e} dB not zero"
            )


# ─── Full autograd tests ────────────────────────────────────────────────────


class TestScatterMoELoRAAutograd:
    """Test full forward + backward through ScatterMoELoRA autograd function."""

    @pytest.fixture(params=CONFIGS_SMALL + CONFIGS_REAL[:2])
    def config(self, request):
        return request.param

    def test_gradients_exist_and_finite(self, config):
        E, K, N, T, k, R, desc = config
        x, W, lA, lB, sei, ssi, eo = _setup(E, K, N, T, k, R)

        x = x.requires_grad_(True)
        lA = lA.requires_grad_(True)
        lB = lB.requires_grad_(True)

        out = ScatterMoELoRA.apply(
            x,
            W,
            k,
            sei,
            ssi,
            eo,
            lA,
            lB,
            SCALING,
            None,
            None,
            False,
            False,
            True,
            False,
        )
        out.sum().backward()

        assert x.grad is not None, f"[{desc}] x.grad is None"
        assert lA.grad is not None, f"[{desc}] lA.grad is None"
        assert lB.grad is not None, f"[{desc}] lB.grad is None"
        assert torch.isfinite(x.grad).all(), f"[{desc}] x.grad has non-finite"
        assert torch.isfinite(lA.grad).all(), f"[{desc}] lA.grad has non-finite"
        assert torch.isfinite(lB.grad).all(), f"[{desc}] lB.grad has non-finite"
        assert x.grad.abs().sum() > 0, f"[{desc}] x.grad all zero"
        assert lA.grad.abs().sum() > 0, f"[{desc}] lA.grad all zero"

    def test_split_matches_fused(self):
        """Split dispatch (for few large experts) matches fused kernel."""
        # Use a shape where split would be dispatched (large K*N, few E)
        E, K, N, T, k, R = 8, 512, 1024, 128, 2, 16
        x, W, lA, lB, sei, ssi, eo = _setup(E, K, N, T, k, R)

        # Force fused path
        orig = lora_ops._SPLIT_LORA_FWD_THRESHOLD
        lora_ops._SPLIT_LORA_FWD_THRESHOLD = 10**18
        out_fused = lora_ops.scatter2scatter_lora(
            X=x,
            W=W,
            sorted_expert_idxs=sei,
            sorted_scattered_idxs=ssi,
            k=k,
            lora_A=lA,
            lora_B=lB,
            scaling=SCALING,
        )

        # Force split path
        lora_ops._SPLIT_LORA_FWD_THRESHOLD = 0
        out_split = lora_ops.scatter2scatter_lora(
            X=x,
            W=W,
            sorted_expert_idxs=sei,
            sorted_scattered_idxs=ssi,
            k=k,
            lora_A=lA,
            lora_B=lB,
            scaling=SCALING,
        )
        lora_ops._SPLIT_LORA_FWD_THRESHOLD = orig

        norm_err = (
            (out_fused.float() - out_split.float()).norm()
            / (out_fused.float().norm() + 1e-6)
        ).item()
        assert norm_err < 0.01, f"split vs fused norm_err={norm_err}"

    def test_scaling_zero_gives_base_only(self):
        """With scaling=0.0, LoRA contribution vanishes. Output = X@W."""
        E, K, N, T, k, R = 16, 64, 32, 32, 2, 4
        x, W, lA, lB, sei, ssi, eo = _setup(E, K, N, T, k, R)

        out_lora = ScatterMoELoRA.apply(
            x,
            W,
            k,
            sei,
            ssi,
            eo,
            lA,
            lB,
            0.0,
            None,
            None,
            False,
            False,
            True,
            False,
        )
        out_base = base_ops.scatter2scatter(
            X=x,
            W=W,
            sorted_expert_idxs=sei,
            sorted_scattered_idxs=ssi,
            k=k,
        )
        err = (out_lora.float() - out_base.float()).abs().max().item()
        assert err < 0.01, f"scaling=0 should match base: err={err}"
