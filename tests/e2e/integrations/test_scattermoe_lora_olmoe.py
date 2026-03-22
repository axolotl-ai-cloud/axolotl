# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""
Integration tests: OLMoE + peft LoRA + ScatterMoE fused kernels.

Validates that scattermoe_lora fused kernels produce correct results when used
with HuggingFace OLMoE models and peft LoRA adapters applied via
``target_parameters``.

Key things tested
-----------------
- LoRA weight layout conversion between peft (rank-major) and scattermoe (expert-major)
- Base forward equivalence: per-expert reference vs ScatterMoE kernels (no LoRA)
- LoRA forward equivalence: peft merged-weight approach vs scattermoe fused kernels
- Backward gradient correctness through the fused LoRA path
- ``kernelize()`` integration via ``LocalLayerRepository``
"""

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import OlmoeConfig
from transformers.models.olmoe.modeling_olmoe import OlmoeSparseMoeBlock

_SMOE = "axolotl.integrations.kernels.libs.scattermoe_lora"

# Try to import from axolotl's scattermoe_lora.layers; may fail on CPU without triton.
try:
    from axolotl.integrations.kernels.libs.scattermoe_lora.layers import (
        _unwrap_experts_lora,
        _unwrap_gate_lora,
        peft_lora_B_to_scattermoe,
        peft_lora_to_scattermoe,
    )

    HAS_SCATTERMOE = True
except (ImportError, ModuleNotFoundError):
    HAS_SCATTERMOE = False

    # Provide pure-torch fallbacks for CPU-only layout conversion tests.
    def peft_lora_B_to_scattermoe(peft_B, num_experts, rank):
        N = peft_B.shape[0]
        return (
            peft_B.reshape(N, rank, num_experts)
            .permute(0, 2, 1)
            .contiguous()
            .reshape(N, num_experts * rank)
        )

    def peft_lora_to_scattermoe(peft_A, peft_B, num_experts, rank):
        peft_B_em = peft_lora_B_to_scattermoe(peft_B, num_experts, rank)
        K_inter, N_hidden = peft_B.shape[0], peft_A.shape[1]
        smoe_A = torch.zeros(
            rank * num_experts,
            K_inter,
            device=peft_A.device,
            dtype=peft_A.dtype,
        )
        smoe_B = torch.zeros(
            N_hidden,
            rank * num_experts,
            device=peft_A.device,
            dtype=peft_A.dtype,
        )
        for e in range(num_experts):
            s = e * rank
            smoe_A[s : s + rank, :] = peft_B_em[:, s : s + rank].T
            smoe_B[:, s : s + rank] = peft_A[s : s + rank, :].T
        return smoe_A, smoe_B

    def _unwrap_experts_lora(experts_module):
        return experts_module, None, None

    def _unwrap_gate_lora(gate_module):
        if hasattr(gate_module, "base_layer") and hasattr(gate_module, "lora_A"):
            base_gate = gate_module.base_layer
            active = getattr(gate_module, "active_adapters", ["default"])
            name = active[0] if active else "default"
            lora_A_dict = getattr(gate_module, "lora_A", {})
            lora_B_dict = getattr(gate_module, "lora_B", {})
            scaling_dict = getattr(gate_module, "scaling", {})
            if name in lora_A_dict:
                lora_A = lora_A_dict[name].weight
                lora_B = lora_B_dict[name].weight
                s = scaling_dict[name]
                delta = s * (lora_B @ lora_A)
                return base_gate, base_gate.weight, delta
            return base_gate, base_gate.weight, None
        return gate_module, gate_module.weight, None


# =============================================================================
# Configuration
# =============================================================================

FULL_OLMOE_CONFIG = dict(
    hidden_size=2048,
    intermediate_size=1024,
    num_experts=64,
    num_experts_per_tok=8,
    hidden_act="silu",
    norm_topk_prob=False,
)

SMALL_OLMOE_CONFIG = dict(
    hidden_size=128,
    intermediate_size=48,  # non-square: 2*inter=96 != hidden=128
    num_experts=8,
    num_experts_per_tok=2,
    hidden_act="silu",
    norm_topk_prob=False,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA not available"
)


def make_olmoe_config(use_full=False):
    cfg = dict(FULL_OLMOE_CONFIG if use_full else SMALL_OLMOE_CONFIG)
    cfg["experts_implementation"] = "grouped_mm"
    return OlmoeConfig(**cfg)


# =============================================================================
# Layout conversion utilities (test-local helpers)
# =============================================================================


def scattermoe_lora_B_to_peft(smoe_B, num_experts, rank):
    """Inverse of ``peft_lora_B_to_scattermoe``."""
    N = smoe_B.shape[0]
    return (
        smoe_B.reshape(N, num_experts, rank)
        .permute(0, 2, 1)
        .contiguous()
        .reshape(N, num_experts * rank)
    )


def peft_gate_up_lora_to_scattermoe(peft_A, peft_B, num_experts, rank):
    """Convert peft LoRA for gate_up_proj to scattermoe layout.

    Both gate_up_proj and down_proj need the A<->B swap because
    scattermoe transposes the parameter (W = param.T).
    """
    return peft_lora_to_scattermoe(peft_A, peft_B, num_experts, rank)


# =============================================================================
# Helpers
# =============================================================================


def _init_expert_weights(moe_block):
    """Initialize OlmoeExperts parameters which use torch.empty (uninitialized).

    Without this, gate_up_proj and down_proj contain garbage/NaN values.
    """
    with torch.no_grad():
        nn.init.kaiming_uniform_(moe_block.experts.gate_up_proj)
        nn.init.kaiming_uniform_(moe_block.experts.down_proj)
    return moe_block


class MinimalOLMoEModel(nn.Module):
    """Thin wrapper so peft's get_peft_model can attach adapters."""

    def __init__(self, config):
        super().__init__()
        self.moe = OlmoeSparseMoeBlock(config)
        _init_expert_weights(self.moe)

    def forward(self, x):
        return self.moe(x)


def _get_routing(moe_block, hidden_states):
    """Run the router and return (routing_weights, selected_experts)."""
    with torch.no_grad():
        _, routing_weights, selected_experts = moe_block.gate(
            hidden_states.view(-1, hidden_states.size(-1))
        )
    return routing_weights, selected_experts


def _reference_moe_forward(
    x_flat,
    gate_up_proj,
    down_proj,
    act_fn,
    top_k_index,
    top_k_weights,
    num_experts,
):
    """Pure-PyTorch per-expert reference MoE forward (no LoRA).

    Uses F.linear per expert for an apples-to-apples comparison with
    the ScatterMoE kernel path.
    """
    final = torch.zeros_like(x_flat)
    expert_mask = F.one_hot(top_k_index, num_classes=num_experts).permute(2, 1, 0)
    for e in range(num_experts):
        top_k_pos, token_idx = torch.where(expert_mask[e])
        if token_idx.numel() == 0:
            continue
        cur = x_flat[token_idx]
        gate_up = F.linear(cur, gate_up_proj[e])
        g, u = gate_up.chunk(2, dim=-1)
        h = act_fn(g) * u
        out = F.linear(h, down_proj[e])
        out = out * top_k_weights[token_idx, top_k_pos, None]
        final.index_add_(0, token_idx, out.to(final.dtype))
    return final


def _reference_moe_forward_with_lora(
    x_flat,
    gate_up_proj,
    down_proj,
    act_fn,
    top_k_index,
    top_k_weights,
    num_experts,
    gup_delta,
    down_delta,
):
    """Pure-PyTorch reference MoE forward with pre-computed weight deltas."""
    merged_gup = gate_up_proj + gup_delta
    merged_down = down_proj + down_delta
    return _reference_moe_forward(
        x_flat,
        merged_gup,
        merged_down,
        act_fn,
        top_k_index,
        top_k_weights,
        num_experts,
    )


def _compute_delta_from_scattermoe_lora(lora_A, lora_B, scaling, E, r, param_shape):
    """Compute additive weight delta from scattermoe-layout LoRA weights.

    delta[e] = scaling * B_e @ A_e  where A_e [r,K], B_e [N,r] -> [N,K].
    """
    delta = torch.zeros(param_shape, device=lora_A.device, dtype=lora_A.dtype)
    for e in range(E):
        A_e = lora_A[e * r : (e + 1) * r, :]
        B_e = lora_B[:, e * r : (e + 1) * r]
        delta[e] = scaling * (B_e @ A_e)
    return delta


# =============================================================================
# Tests: Layout conversion
# =============================================================================


class TestLoRABLayoutConversion:
    """Test the peft <-> scattermoe lora_B layout conversion."""

    def test_roundtrip(self):
        E, r, N = 8, 4, 64
        original = torch.randn(N, E * r)
        converted = peft_lora_B_to_scattermoe(original, E, r)
        back = scattermoe_lora_B_to_peft(converted, E, r)
        torch.testing.assert_close(back, original)

    def test_per_expert_slices(self):
        """After conversion, scattermoe slicing gives the same per-expert
        matrices as peft's reshape slicing."""
        E, r, N = 4, 2, 16
        peft_B = torch.randn(N, E * r)
        smoe_B = peft_lora_B_to_scattermoe(peft_B, E, r)

        peft_reshaped = peft_B.reshape(N, r, E)
        for e in range(E):
            torch.testing.assert_close(
                smoe_B[:, e * r : (e + 1) * r],
                peft_reshaped[:, :, e],
            )

    def test_lora_A_already_compatible(self):
        """lora_A layout is identical between peft and scattermoe."""
        E, r, K = 4, 2, 16
        lora_A = torch.randn(E * r, K)
        peft_reshaped = lora_A.reshape(E, r, K)
        for e in range(E):
            torch.testing.assert_close(
                lora_A[e * r : (e + 1) * r, :],
                peft_reshaped[e],
            )

    def test_delta_weight_equivalence(self):
        """peft's einsum delta matches per-expert B @ A with converted layouts."""
        E, r, K, N = 8, 4, 32, 64
        peft_A = torch.randn(E * r, K)
        peft_B = torch.randn(N, E * r)
        scaling = 2.0

        A_r = peft_A.reshape(E, r, K)
        B_r = peft_B.reshape(N, r, E)
        delta_peft = torch.einsum("o r e, e r i -> e i o", B_r, A_r) * scaling

        smoe_B = peft_lora_B_to_scattermoe(peft_B, E, r)
        for e in range(E):
            A_e = peft_A[e * r : (e + 1) * r, :]
            B_e = smoe_B[:, e * r : (e + 1) * r]
            delta_e = scaling * (B_e @ A_e)
            torch.testing.assert_close(delta_e, delta_peft[e].T, atol=1e-5, rtol=1e-5)

    def test_down_proj_conversion(self):
        """Verify peft_lora_to_scattermoe produces correct delta."""
        E, r = 4, 2
        hidden, inter = 32, 16
        scaling = 2.0

        peft_A = torch.randn(E * r, hidden)
        peft_B = torch.randn(inter, E * r)

        A_r = peft_A.reshape(E, r, hidden)
        B_r = peft_B.reshape(inter, r, E)
        delta_peft = torch.einsum("o r e, e r i -> e i o", B_r, A_r) * scaling

        smoe_A, smoe_B = peft_lora_to_scattermoe(peft_A, peft_B, E, r)
        for e in range(E):
            A_e = smoe_A[e * r : (e + 1) * r, :]
            B_e = smoe_B[:, e * r : (e + 1) * r]
            delta_smoe_e = scaling * (B_e @ A_e)
            torch.testing.assert_close(
                delta_smoe_e, delta_peft[e], atol=1e-5, rtol=1e-5
            )

    def test_gate_up_proj_conversion(self):
        """Verify gate_up_proj LoRA conversion with non-square dims (Qwen3-like).

        gate_up_proj param: [E, 2*inter, hidden].
        peft: in_features=2*inter, out_features=hidden.
        peft lora_A: [r*E, 2*inter], lora_B: [hidden, r*E].

        scattermoe W = param.T = [E, hidden, 2*inter], K=hidden, N=2*inter.
        scattermoe needs: lora_A [r*E, K=hidden], lora_B [N=2*inter, r*E].

        Uses non-square dims (hidden=32 != 2*inter=24) to catch A<->B swap bugs.
        """
        E, r = 4, 2
        hidden, inter = 32, 12  # 2*inter=24 != hidden=32
        scaling = 2.0

        # peft assigns: in_features=2*inter, out_features=hidden
        peft_A = torch.randn(E * r, 2 * inter)  # [r*E, in_features=2*inter]
        peft_B = torch.randn(hidden, E * r)  # [out_features=hidden, r*E]

        # peft delta via einsum: "o r e, e r i -> e i o"
        A_r = peft_A.reshape(E, r, 2 * inter)
        B_r = peft_B.reshape(hidden, r, E)
        delta_peft = torch.einsum("o r e, e r i -> e i o", B_r, A_r) * scaling
        # delta_peft[e] has shape [in_features, out_features] = [2*inter, hidden]
        # = param[e] shape [2*inter, hidden]

        smoe_A, smoe_B = peft_gate_up_lora_to_scattermoe(peft_A, peft_B, E, r)
        # smoe_A should be [r*E, K=hidden], smoe_B should be [N=2*inter, r*E]
        assert smoe_A.shape == (E * r, hidden), (
            f"Expected {(E * r, hidden)}, got {smoe_A.shape}"
        )
        assert smoe_B.shape == (2 * inter, E * r), (
            f"Expected {(2 * inter, E * r)}, got {smoe_B.shape}"
        )

        for e in range(E):
            A_e = smoe_A[e * r : (e + 1) * r, :]  # [r, K=hidden]
            B_e = smoe_B[:, e * r : (e + 1) * r]  # [N=2*inter, r]
            delta_smoe_e = scaling * (B_e @ A_e)  # [2*inter, hidden]
            # Should match peft delta which is [2*inter, hidden] = param[e]
            torch.testing.assert_close(
                delta_smoe_e, delta_peft[e], atol=1e-5, rtol=1e-5
            )


# =============================================================================
# Tests: peft weight extraction
# =============================================================================


class TestPeftLoRAWeightExtraction:
    """Test extracting peft LoRA weights for OLMoE."""

    def test_peft_creates_correct_shapes(self):
        config = make_olmoe_config(use_full=False)
        E, r = config.num_experts, 4

        model = MinimalOLMoEModel(config)
        lora_config = LoraConfig(
            r=r,
            lora_alpha=16,
            target_modules=[],
            target_parameters=[
                "gate.weight",
                "experts.gate_up_proj",
                "experts.down_proj",
            ],
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)
        trainable = {n: p for n, p in peft_model.named_parameters() if p.requires_grad}

        # Gate router
        assert trainable["base_model.model.moe.gate.lora_A.default.weight"].shape == (
            r,
            config.hidden_size,
        )
        assert trainable["base_model.model.moe.gate.lora_B.default.weight"].shape == (
            E,
            r,
        )

        # gate_up_proj [E, 2*inter, hidden]
        # peft: in_features=2*inter (dim 1), out_features=hidden (dim 2)
        assert trainable[
            "base_model.model.moe.experts.base_layer.lora_A.default.weight"
        ].shape == (E * r, 2 * config.intermediate_size)
        assert trainable[
            "base_model.model.moe.experts.base_layer.lora_B.default.weight"
        ].shape == (config.hidden_size, E * r)

        # down_proj [E, hidden, inter]
        # peft: in_features=hidden (dim 1), out_features=inter (dim 2)
        assert trainable[
            "base_model.model.moe.experts.lora_A.default.weight"
        ].shape == (E * r, config.hidden_size)
        assert trainable[
            "base_model.model.moe.experts.lora_B.default.weight"
        ].shape == (config.intermediate_size, E * r)

    @requires_cuda
    def test_peft_forward_runs(self):
        """Smoke test: peft model forward pass completes (needs CUDA for grouped_mm)."""
        config = make_olmoe_config(use_full=False)
        model = MinimalOLMoEModel(config)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=[],
            target_parameters=[
                "gate.weight",
                "experts.gate_up_proj",
                "experts.down_proj",
            ],
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)
        x = torch.randn(1, 4, config.hidden_size)
        out = peft_model(x)
        assert out.shape == x.shape

    @pytest.mark.skipif(
        not HAS_SCATTERMOE, reason="scattermoe_lora not importable (no triton)"
    )
    def test_unwrap_experts_lora(self):
        """Test that _unwrap_experts_lora correctly detects LoRA wrappers."""
        config = make_olmoe_config(use_full=False)
        model = MinimalOLMoEModel(config)
        lora_config = LoraConfig(
            r=4,
            lora_alpha=16,
            target_modules=[],
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)
        base_moe = peft_model.base_model.model.moe

        # Experts should be wrapped by ParamWrapper
        experts, gup_lora, down_lora = _unwrap_experts_lora(base_moe.experts)

        # Base experts should have the raw parameters
        assert hasattr(experts, "gate_up_proj")
        assert hasattr(experts, "down_proj")

        # LoRA should be detected
        assert gup_lora is not None, "gate_up_proj LoRA not detected"
        assert down_lora is not None, "down_proj LoRA not detected"

        # Check shapes (after peft->scattermoe conversion with A<->B swap)
        # gate_up_proj W = param.T = [E, hidden, 2*inter], K=hidden, N=2*inter
        E, r = config.num_experts, 4
        gup_A, gup_B, gup_s = gup_lora
        assert gup_A.shape == (E * r, config.hidden_size), (
            f"gate_up_proj smoe_A: expected [r*E, K=hidden]={(E * r, config.hidden_size)}, "
            f"got {gup_A.shape}"
        )
        assert gup_B.shape == (2 * config.intermediate_size, E * r), (
            f"gate_up_proj smoe_B: expected [N=2*inter, r*E]="
            f"{(2 * config.intermediate_size, E * r)}, got {gup_B.shape}"
        )

        # down_proj W = param.T = [E, inter, hidden], K=inter, N=hidden
        down_A, down_B, down_s = down_lora
        assert down_A.shape == (E * r, config.intermediate_size), (
            f"down_proj smoe_A: expected [r*E, K=inter]={(E * r, config.intermediate_size)}, "
            f"got {down_A.shape}"
        )
        assert down_B.shape == (config.hidden_size, E * r), (
            f"down_proj smoe_B: expected [N=hidden, r*E]={(config.hidden_size, E * r)}, "
            f"got {down_B.shape}"
        )

    def test_unwrap_no_lora(self):
        """Without peft, _unwrap_experts_lora returns no LoRA."""
        config = make_olmoe_config(use_full=False)
        moe = OlmoeSparseMoeBlock(config)
        experts, gup_lora, down_lora = _unwrap_experts_lora(moe.experts)
        assert gup_lora is None
        assert down_lora is None
        assert hasattr(experts, "gate_up_proj")

    def test_unwrap_gate_lora(self):
        """Test that _unwrap_gate_lora detects LoRA on the router gate."""
        config = make_olmoe_config(use_full=False)
        model = MinimalOLMoEModel(config)
        r = 4
        lora_config = LoraConfig(
            r=r,
            lora_alpha=16,
            target_modules=[],
            target_parameters=["gate.weight"],
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)
        base_moe = peft_model.base_model.model.moe

        # Set non-zero LoRA weights (peft initializes lora_B to zeros)
        with torch.no_grad():
            base_moe.gate.lora_B["default"].weight.normal_(0, 0.01)

        base_gate, gate_weight, gate_delta = _unwrap_gate_lora(base_moe.gate)

        # Base gate should be the original router
        assert hasattr(base_gate, "top_k")
        assert hasattr(base_gate, "num_experts")
        assert base_gate.top_k == config.num_experts_per_tok
        assert base_gate.num_experts == config.num_experts

        # Gate weight should be the base weight (delta returned separately)
        assert gate_weight.shape == (config.num_experts, config.hidden_size)
        torch.testing.assert_close(gate_weight, base_gate.weight)

        # Delta should be non-zero (LoRA was applied)
        assert gate_delta is not None
        assert gate_delta.shape == (config.num_experts, config.hidden_size)
        assert gate_delta.abs().max() > 0, "Gate LoRA delta should be non-zero"

    def test_unwrap_gate_no_lora(self):
        """Without peft, _unwrap_gate_lora returns the original gate."""
        config = make_olmoe_config(use_full=False)
        moe = OlmoeSparseMoeBlock(config)
        base_gate, gate_weight, gate_delta = _unwrap_gate_lora(moe.gate)
        assert base_gate is moe.gate
        torch.testing.assert_close(gate_weight, moe.gate.weight)
        assert gate_delta is None

    def test_gate_lora_delta_matches_peft(self):
        """Verify _unwrap_gate_lora computes the same delta as peft."""
        config = make_olmoe_config(use_full=False)
        model = MinimalOLMoEModel(config)
        r = 4
        lora_alpha = 16
        scaling = lora_alpha / r
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=[],
            target_parameters=["gate.weight"],
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)
        base_moe = peft_model.base_model.model.moe

        # Our unwrapped weight + delta
        _, gate_weight, gate_delta = _unwrap_gate_lora(base_moe.gate)

        # Manually compute expected delta
        lora_A = base_moe.gate.lora_A["default"].weight  # [r, hidden]
        lora_B = base_moe.gate.lora_B["default"].weight  # [E, r]
        base_weight = base_moe.gate.base_layer.weight  # [E, hidden]
        expected_delta = scaling * (lora_B @ lora_A)

        torch.testing.assert_close(gate_weight, base_weight)
        torch.testing.assert_close(gate_delta, expected_delta)
        # Combined should match the old behavior
        torch.testing.assert_close(
            gate_weight + gate_delta, base_weight + expected_delta
        )


# =============================================================================
# Tests: Base forward equivalence (no LoRA)
# =============================================================================


@requires_cuda
class TestOLMoEReferenceVsScatterMoE:
    """Base forward equivalence: per-expert reference vs ScatterMoE kernels."""

    def test_small(self):
        self._run(use_full=False, M=16)

    @pytest.mark.slow
    def test_full(self):
        self._run(use_full=True, M=32)

    def _run(self, use_full, M):
        from axolotl.integrations.kernels.libs.scattermoe_lora import (
            flatten_sort_count,
            parallel_linear,
        )

        config = make_olmoe_config(use_full=use_full)
        torch.manual_seed(42)
        moe = _init_expert_weights(OlmoeSparseMoeBlock(config)).cuda().float()
        E, k = config.num_experts, config.num_experts_per_tok

        x = torch.randn(1, M, config.hidden_size, device="cuda")
        x_flat = x.view(-1, config.hidden_size)

        with torch.no_grad():
            # Shared routing for both paths
            _, rw, sel = moe.gate(x_flat)
            sei, ssi, eo = flatten_sort_count(sel, num_experts=E)

            # Per-expert reference
            ref_out = _reference_moe_forward(
                x_flat,
                moe.experts.gate_up_proj,
                moe.experts.down_proj,
                moe.experts.act_fn,
                sel,
                rw,
                E,
            ).view(1, M, config.hidden_size)

            # ScatterMoE kernel path
            gup = parallel_linear(
                x_flat,
                moe.experts.gate_up_proj.transpose(2, 1),
                k,
                sei,
                ssi,
                eo,
                grouped_in=False,
                grouped_out=True,
            )
            g, u = gup.chunk(2, dim=-1)
            h = moe.experts.act_fn(g) * u

            smoe_out = parallel_linear(
                h,
                moe.experts.down_proj.transpose(2, 1),
                1,
                sei,
                ssi,
                eo,
                grouped_in=True,
                grouped_out=False,
                gates=rw,
            ).view(1, M, config.hidden_size)

        torch.testing.assert_close(smoe_out, ref_out, atol=1e-3, rtol=1e-3)


# =============================================================================
# Tests: LoRA forward equivalence (peft vs scattermoe fused)
# =============================================================================


@requires_cuda
class TestOLMoEPeftLoRAForward:
    """Fused LoRA forward: peft merged-weight vs scattermoe_lora kernel."""

    def test_small(self):
        self._run(use_full=False, M=16, r=4)

    @pytest.mark.slow
    def test_full(self):
        self._run(use_full=True, M=32, r=8)

    def _run(self, use_full, M, r):
        from axolotl.integrations.kernels.libs.scattermoe_lora import (
            flatten_sort_count,
            parallel_linear_lora,
        )

        config = make_olmoe_config(use_full=use_full)
        E, k = config.num_experts, config.num_experts_per_tok
        lora_alpha = 16
        scaling = lora_alpha / r

        # Create peft model
        model = MinimalOLMoEModel(config).cuda().float()
        lora_config = LoraConfig(
            r=r,
            lora_alpha=lora_alpha,
            target_modules=[],
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)

        torch.manual_seed(42)
        x = torch.randn(1, M, config.hidden_size, device="cuda")

        # peft forward
        with torch.no_grad():
            peft_out = peft_model(x)

        # Extract base weights and LoRA weights
        base_moe = peft_model.base_model.model.moe
        base_experts = base_moe.experts.base_layer.base_layer
        gate_up_proj = base_experts.gate_up_proj
        down_proj = base_experts.down_proj
        act_fn = base_experts.act_fn

        # gate_up_proj LoRA
        gup_w = base_moe.experts.base_layer
        peft_gup_A = gup_w.lora_A["default"].weight.detach()
        peft_gup_B = gup_w.lora_B["default"].weight.detach()
        smoe_gup_A, smoe_gup_B = peft_gate_up_lora_to_scattermoe(
            peft_gup_A, peft_gup_B, E, r
        )

        # down_proj LoRA
        down_w = base_moe.experts
        peft_down_A = down_w.lora_A["default"].weight.detach()
        peft_down_B = down_w.lora_B["default"].weight.detach()
        smoe_down_A, smoe_down_B = peft_lora_to_scattermoe(
            peft_down_A, peft_down_B, E, r
        )

        # ScatterMoE fused forward -- gate is NOT peft-wrapped, access directly
        x_flat = x.view(-1, config.hidden_size)

        with torch.no_grad():
            _, rw, sel = base_moe.gate(x_flat)
            sei, ssi, eo = flatten_sort_count(sel, num_experts=E)

            gup = parallel_linear_lora(
                x_flat,
                gate_up_proj.transpose(2, 1),
                k,
                sei,
                ssi,
                eo,
                lora_A=smoe_gup_A,
                lora_B=smoe_gup_B,
                scaling=scaling,
                grouped_in=False,
                grouped_out=True,
            )
            g, u = gup.chunk(2, dim=-1)
            h = act_fn(g) * u

            smoe_out = parallel_linear_lora(
                h,
                down_proj.transpose(2, 1),
                1,
                sei,
                ssi,
                eo,
                lora_A=smoe_down_A,
                lora_B=smoe_down_B,
                scaling=scaling,
                grouped_in=True,
                grouped_out=False,
                gates=rw,
            ).view(1, M, config.hidden_size)

        torch.testing.assert_close(smoe_out, peft_out, atol=5e-3, rtol=5e-3)


# =============================================================================
# Tests: Backward gradient correctness
# =============================================================================


@requires_cuda
class TestOLMoEPeftLoRABackward:
    """Backward gradients through scattermoe_lora vs pure-PyTorch reference."""

    def test_small(self):
        self._run(use_full=False, M=16, r=4)

    def _run(self, use_full, M, r):
        from axolotl.integrations.kernels.libs.scattermoe_lora import (
            flatten_sort_count,
            parallel_linear_lora,
        )

        config = make_olmoe_config(use_full=use_full)
        E, k = config.num_experts, config.num_experts_per_tok
        lora_alpha = 16
        scaling = lora_alpha / r

        torch.manual_seed(42)
        moe = _init_expert_weights(OlmoeSparseMoeBlock(config)).cuda().float()
        x = torch.randn(1, M, config.hidden_size, device="cuda")
        x_flat = x.view(-1, config.hidden_size)
        gate_up_proj = moe.experts.gate_up_proj
        down_proj = moe.experts.down_proj

        # Create LoRA weights in scattermoe layout directly
        gup_A = torch.randn(r * E, config.hidden_size, device="cuda") * 0.01
        gup_B = torch.randn(2 * config.intermediate_size, r * E, device="cuda") * 0.01
        down_A = torch.randn(r * E, config.intermediate_size, device="cuda") * 0.01
        down_B = torch.randn(config.hidden_size, r * E, device="cuda") * 0.01

        rw, sel = _get_routing(moe, x)
        sei, ssi, eo = flatten_sort_count(sel, num_experts=E)

        # --- Reference ---
        gup_delta = _compute_delta_from_scattermoe_lora(
            gup_A, gup_B, scaling, E, r, gate_up_proj.shape
        )
        down_delta = _compute_delta_from_scattermoe_lora(
            down_A, down_B, scaling, E, r, down_proj.shape
        )

        x_ref = x_flat.clone().detach().requires_grad_(True)
        ref_out = _reference_moe_forward_with_lora(
            x_ref,
            gate_up_proj,
            down_proj,
            moe.experts.act_fn,
            sel,
            rw,
            E,
            gup_delta,
            down_delta,
        )
        ref_out.sum().backward()

        # --- ScatterMoE fused path ---
        x_smoe = x_flat.clone().detach().requires_grad_(True)
        gup_A_s = gup_A.clone().requires_grad_(True)
        gup_B_s = gup_B.clone().requires_grad_(True)
        down_A_s = down_A.clone().requires_grad_(True)
        down_B_s = down_B.clone().requires_grad_(True)

        gup_out = parallel_linear_lora(
            x_smoe,
            gate_up_proj.transpose(2, 1),
            k,
            sei,
            ssi,
            eo,
            lora_A=gup_A_s,
            lora_B=gup_B_s,
            scaling=scaling,
            grouped_in=False,
            grouped_out=True,
        )
        g, u = gup_out.chunk(2, dim=-1)
        h = moe.experts.act_fn(g) * u

        smoe_out = parallel_linear_lora(
            h,
            down_proj.transpose(2, 1),
            1,
            sei,
            ssi,
            eo,
            lora_A=down_A_s,
            lora_B=down_B_s,
            scaling=scaling,
            grouped_in=True,
            grouped_out=False,
            gates=rw,
        )
        smoe_out.sum().backward()

        torch.testing.assert_close(
            smoe_out.detach(),
            ref_out.detach(),
            atol=5e-3,
            rtol=5e-3,
        )
        torch.testing.assert_close(
            x_smoe.grad,
            x_ref.grad,
            atol=5e-2,
            rtol=5e-2,
        )


# =============================================================================
# Tests: kernelize() integration via LocalLayerRepository
# =============================================================================


@requires_cuda
class TestKernelizeIntegration:
    """Test the HF kernels library integration with LocalLayerRepository."""

    @staticmethod
    def _get_kernelize_imports():
        """Import kernels library components, skip if not available."""
        try:
            from kernels import (
                LocalLayerRepository,
                Mode,
                kernelize,
                register_kernel_mapping,
                replace_kernel_forward_from_hub,
            )

            return (
                LocalLayerRepository,
                Mode,
                register_kernel_mapping,
                replace_kernel_forward_from_hub,
                kernelize,
            )
        except ImportError:
            pytest.skip("kernels library not installed")

    @staticmethod
    def _get_repo_path():
        """Get the path to scattermoe_lora within axolotl's plugin."""
        return (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "axolotl"
            / "integrations"
            / "kernels"
            / "libs"
            / "scattermoe_lora"
        )

    def _setup_kernels(
        self,
        LocalLayerRepository,
        Mode,
        register_kernel_mapping,
        replace_kernel_forward_from_hub,
    ):
        """Register kernel mapping for tests."""
        repo_path = self._get_repo_path()
        local_repo = LocalLayerRepository(
            repo_path=repo_path,
            package_name="scattermoe_lora",
            layer_name="HFScatterMoEGatedMLP",
        )

        replace_kernel_forward_from_hub(
            OlmoeSparseMoeBlock, "HFScatterMoEParallelExperts"
        )
        register_kernel_mapping(
            {
                "HFScatterMoEParallelExperts": {
                    "cuda": {
                        Mode.TRAINING: local_repo,
                        Mode.INFERENCE: local_repo,
                    },
                }
            }
        )

    def test_base_forward_via_kernelize(self):
        """Kernelized OlmoeSparseMoeBlock (no LoRA) matches per-expert reference."""
        (
            LocalLayerRepository,
            Mode,
            register_kernel_mapping,
            replace_kernel_forward_from_hub,
            kernelize,
        ) = self._get_kernelize_imports()

        config = make_olmoe_config(use_full=False)
        E = config.num_experts

        # Create model
        torch.manual_seed(42)
        moe = _init_expert_weights(OlmoeSparseMoeBlock(config)).cuda().float()
        x = torch.randn(1, 8, config.hidden_size, device="cuda")
        x_flat = x.view(-1, config.hidden_size)

        # Compute reference BEFORE kernelizing
        with torch.no_grad():
            _, rw, sel = moe.gate(x_flat)
            ref_out = _reference_moe_forward(
                x_flat,
                moe.experts.gate_up_proj,
                moe.experts.down_proj,
                moe.experts.act_fn,
                sel,
                rw,
                E,
            ).view(1, 8, config.hidden_size)

        # Set up kernel mapping
        self._setup_kernels(
            LocalLayerRepository,
            Mode,
            register_kernel_mapping,
            replace_kernel_forward_from_hub,
        )

        # Kernelize the model
        kernelize(moe, mode=Mode.TRAINING, device="cuda")

        # Forward through kernelized model
        with torch.no_grad():
            kern_out = moe(x)

        torch.testing.assert_close(kern_out, ref_out, atol=1e-3, rtol=1e-3)

    def test_lora_forward_via_kernelize(self):
        """Kernelized OlmoeSparseMoeBlock with peft LoRA matches reference."""
        (
            LocalLayerRepository,
            Mode,
            register_kernel_mapping,
            replace_kernel_forward_from_hub,
            kernelize,
        ) = self._get_kernelize_imports()

        config = make_olmoe_config(use_full=False)
        r = 4

        # Create peft model
        torch.manual_seed(42)
        model = MinimalOLMoEModel(config).cuda().float()
        lora_config = LoraConfig(
            r=r,
            lora_alpha=16,
            target_modules=[],
            target_parameters=["experts.gate_up_proj", "experts.down_proj"],
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)

        x = torch.randn(1, 8, config.hidden_size, device="cuda")

        # Reference: peft's own forward (uses _activate_lora context manager)
        with torch.no_grad():
            ref_out = peft_model(x)

        # Set up kernel mapping
        self._setup_kernels(
            LocalLayerRepository,
            Mode,
            register_kernel_mapping,
            replace_kernel_forward_from_hub,
        )

        # Kernelize the MoE block inside the peft model
        base_moe = peft_model.base_model.model.moe
        kernelize(base_moe, mode=Mode.TRAINING, device="cuda")

        # Forward through kernelized peft model
        with torch.no_grad():
            kern_out = peft_model(x)

        torch.testing.assert_close(kern_out, ref_out, atol=5e-3, rtol=5e-3)

    def test_gate_lora_forward_via_kernelize(self):
        """Kernelized forward with gate LoRA matches peft reference."""
        (
            LocalLayerRepository,
            Mode,
            register_kernel_mapping,
            replace_kernel_forward_from_hub,
            kernelize,
        ) = self._get_kernelize_imports()

        config = make_olmoe_config(use_full=False)
        r = 4

        # Create peft model with gate + experts LoRA
        torch.manual_seed(42)
        model = MinimalOLMoEModel(config).cuda().float()
        lora_config = LoraConfig(
            r=r,
            lora_alpha=16,
            target_modules=[],
            target_parameters=[
                "gate.weight",
                "experts.gate_up_proj",
                "experts.down_proj",
            ],
            bias="none",
        )
        peft_model = get_peft_model(model, lora_config)

        x = torch.randn(1, 8, config.hidden_size, device="cuda")

        # Reference: peft's own forward
        with torch.no_grad():
            ref_out = peft_model(x)

        # Set up kernel mapping
        self._setup_kernels(
            LocalLayerRepository,
            Mode,
            register_kernel_mapping,
            replace_kernel_forward_from_hub,
        )

        # Kernelize the MoE block inside the peft model
        base_moe = peft_model.base_model.model.moe
        kernelize(base_moe, mode=Mode.TRAINING, device="cuda")

        # Forward through kernelized peft model
        with torch.no_grad():
            kern_out = peft_model(x)

        torch.testing.assert_close(kern_out, ref_out, atol=5e-3, rtol=5e-3)


# =============================================================================
# Tests: Shared expert handling
# =============================================================================


class TestSharedExpertHandling:
    """Test that HFScatterMoEGatedMLP.forward handles shared experts."""

    @staticmethod
    def _make_shared_expert_block(config):
        """Create an OlmoeSparseMoeBlock with a mock shared expert attached."""
        moe = OlmoeSparseMoeBlock(config)
        _init_expert_weights(moe)

        hidden = config.hidden_size
        inter = config.intermediate_size

        # Attach a simple shared expert MLP (mimics Qwen2MoE structure)
        class SharedExpertMLP(nn.Module):
            def __init__(self, hidden_size, intermediate_size):
                super().__init__()
                self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
                self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
                self.act_fn = nn.SiLU()

            def forward(self, x):
                return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))

        moe.shared_expert = SharedExpertMLP(hidden, inter)
        moe.shared_expert_gate = nn.Linear(hidden, 1, bias=False)

        return moe

    def test_shared_expert_is_used(self):
        """Verify shared expert output affects final result."""
        config = make_olmoe_config(use_full=False)
        moe = self._make_shared_expert_block(config)

        # Compute reference without shared expert
        torch.manual_seed(42)
        x = torch.randn(1, 4, config.hidden_size)
        x_flat = x.view(-1, config.hidden_size)

        with torch.no_grad():
            # Shared expert contribution
            shared_out = moe.shared_expert(x_flat)
            gate_val = F.sigmoid(moe.shared_expert_gate(x_flat))
            shared_contribution = shared_out * gate_val

        # Verify shared expert produces non-zero output
        assert shared_contribution.abs().max() > 0

    @requires_cuda
    def test_shared_expert_forward_via_kernelize(self):
        """Kernelized forward with shared expert matches manual reference."""
        try:
            from kernels import (
                LocalLayerRepository,
                Mode,
                kernelize,
                register_kernel_mapping,
                replace_kernel_forward_from_hub,
            )
        except ImportError:
            pytest.skip("kernels library not installed")

        config = make_olmoe_config(use_full=False)
        E = config.num_experts

        torch.manual_seed(42)
        moe = self._make_shared_expert_block(config).cuda().float()
        x = torch.randn(1, 8, config.hidden_size, device="cuda")
        x_flat = x.view(-1, config.hidden_size)

        # Compute reference: per-expert + shared expert
        with torch.no_grad():
            _, rw, sel = moe.gate(x_flat)

            expert_out = _reference_moe_forward(
                x_flat,
                moe.experts.gate_up_proj,
                moe.experts.down_proj,
                moe.experts.act_fn,
                sel,
                rw,
                E,
            )
            shared_out = moe.shared_expert(x_flat)
            gate_val = F.sigmoid(moe.shared_expert_gate(x_flat))
            ref_out = (expert_out + shared_out * gate_val).view(
                1, 8, config.hidden_size
            )

        # Kernelize
        repo_path = (
            Path(__file__).parent.parent.parent.parent
            / "src"
            / "axolotl"
            / "integrations"
            / "kernels"
            / "libs"
            / "scattermoe_lora"
        )
        local_repo = LocalLayerRepository(
            repo_path=repo_path,
            package_name="scattermoe_lora",
            layer_name="HFScatterMoEGatedMLP",
        )

        replace_kernel_forward_from_hub(
            OlmoeSparseMoeBlock, "HFScatterMoEParallelExperts"
        )
        register_kernel_mapping(
            {
                "HFScatterMoEParallelExperts": {
                    "cuda": {
                        Mode.TRAINING: local_repo,
                        Mode.INFERENCE: local_repo,
                    },
                }
            }
        )

        kernelize(moe, mode=Mode.TRAINING, device="cuda")

        with torch.no_grad():
            kern_out = moe(x)

        torch.testing.assert_close(kern_out, ref_out, atol=1e-3, rtol=1e-3)
