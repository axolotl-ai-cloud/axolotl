# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Unit tests for SonicMoE LoRA support."""

from unittest.mock import MagicMock

import pytest
import torch

from axolotl.integrations.kernels.libs.sonicmoe.lora import (
    MoELoRAMaterialize,
    get_lora_params_from_wrapper,
    has_lora,
    materialize_expert_lora,
    unwrap_experts_lora,
    unwrap_gate_lora,
)

# =============================================================================
# Helpers: mock PEFT modules
# =============================================================================


def _make_mock_lora_module(weight_A, weight_B, scaling_val, param_name=None):
    """Create a mock PEFT-wrapped module with LoRA attributes."""
    mock = MagicMock()

    lora_A_linear = MagicMock()
    lora_A_linear.weight = weight_A

    lora_B_linear = MagicMock()
    lora_B_linear.weight = weight_B

    mock.lora_A = {"default": lora_A_linear}
    mock.lora_B = {"default": lora_B_linear}
    mock.scaling = {"default": scaling_val}
    mock.active_adapters = ["default"]

    if param_name is not None:
        mock.parameter_name = param_name

    return mock


def _make_peft_gate(hidden_size, num_experts, rank, scaling=0.5):
    """Create a mock PEFT-wrapped gate module."""
    base_gate = MagicMock()
    base_gate.weight = torch.randn(num_experts, hidden_size)
    base_gate.top_k = 2
    base_gate.norm_topk_prob = True

    lora_A = torch.randn(rank, hidden_size)
    lora_B = torch.randn(num_experts, rank)

    wrapper = _make_mock_lora_module(lora_A, lora_B, scaling)
    wrapper.base_layer = base_gate
    return wrapper, base_gate


def _make_peft_experts(
    num_experts, gate_up_dim, down_dim, hidden_size, rank, scaling=0.5
):
    """Create a mock PEFT-wrapped experts chain.

    Simulates: ParamWrapper(down_proj) -> ParamWrapper(gate_up_proj) -> Experts
    """
    base_experts = MagicMock()
    base_experts.gate_up_proj = torch.randn(num_experts, gate_up_dim, hidden_size)
    base_experts.down_proj = torch.randn(num_experts, hidden_size, down_dim)
    # Remove base_layer and lora_A from base_experts so the chain walk stops
    del base_experts.base_layer
    del base_experts.lora_A

    # gate_up_proj wrapper
    gup_A = torch.randn(rank * num_experts, hidden_size)
    gup_B = torch.randn(gate_up_dim, rank * num_experts)
    gup_wrapper = _make_mock_lora_module(gup_A, gup_B, scaling, "gate_up_proj")
    gup_wrapper.base_layer = base_experts

    # down_proj wrapper (outermost)
    down_A = torch.randn(rank * num_experts, down_dim)
    down_B = torch.randn(hidden_size, rank * num_experts)
    down_wrapper = _make_mock_lora_module(down_A, down_B, scaling, "down_proj")
    down_wrapper.base_layer = gup_wrapper

    return down_wrapper, base_experts, (gup_A, gup_B), (down_A, down_B)


# =============================================================================
# Tests: has_lora
# =============================================================================


class TestHasLora:
    def test_plain_module(self):
        m = MagicMock(spec=["weight"])
        del m.base_layer
        del m.lora_A
        assert not has_lora(m)

    def test_wrapped_module(self):
        m = MagicMock()
        m.base_layer = MagicMock()
        m.lora_A = {"default": MagicMock()}
        assert has_lora(m)


# =============================================================================
# Tests: get_lora_params_from_wrapper
# =============================================================================


class TestGetLoraParams:
    def test_no_lora_attrs(self):
        m = MagicMock(spec=["weight"])
        del m.lora_A
        del m.lora_B
        assert get_lora_params_from_wrapper(m) == (None, None, None)

    def test_extracts_params(self):
        A = torch.randn(4, 8)
        B = torch.randn(16, 4)
        wrapper = _make_mock_lora_module(A, B, 0.5)
        lora_A, lora_B, scaling = get_lora_params_from_wrapper(wrapper)
        assert torch.equal(lora_A, A)
        assert torch.equal(lora_B, B)
        assert scaling == 0.5

    def test_no_active_adapters(self):
        wrapper = _make_mock_lora_module(torch.randn(4, 8), torch.randn(16, 4), 0.5)
        wrapper.active_adapters = []
        assert get_lora_params_from_wrapper(wrapper) == (None, None, None)


# =============================================================================
# Tests: unwrap_gate_lora
# =============================================================================


class TestUnwrapGateLora:
    def test_plain_gate(self):
        gate = MagicMock(spec=["weight", "top_k"])
        del gate.base_layer
        del gate.lora_A
        gate.weight = torch.randn(8, 64)
        base, weight, delta = unwrap_gate_lora(gate)
        assert base is gate
        assert torch.equal(weight, gate.weight)
        assert delta is None

    def test_wrapped_gate(self):
        wrapper, base_gate = _make_peft_gate(
            hidden_size=64, num_experts=8, rank=4, scaling=0.5
        )
        base, weight, delta = unwrap_gate_lora(wrapper)
        assert base is base_gate
        assert torch.equal(weight, base_gate.weight)
        assert delta is not None
        assert delta.shape == base_gate.weight.shape

        # Verify delta = scaling * B @ A
        lora_A = wrapper.lora_A["default"].weight
        lora_B = wrapper.lora_B["default"].weight
        expected = 0.5 * (lora_B @ lora_A)
        assert torch.allclose(delta, expected)


# =============================================================================
# Tests: unwrap_experts_lora
# =============================================================================


class TestUnwrapExpertsLora:
    def test_plain_experts(self):
        experts = MagicMock(spec=["gate_up_proj", "down_proj"])
        del experts.base_layer
        del experts.lora_A
        base, lora_dict = unwrap_experts_lora(experts)
        assert base is experts
        assert lora_dict == {}

    def test_wrapped_experts(self):
        E, I2, I, H, r = 4, 256, 128, 64, 8  # noqa: E741
        wrapper, base_experts, (gup_A, gup_B), (down_A, down_B) = _make_peft_experts(
            E, I2, I, H, r, scaling=0.25
        )
        base, lora_dict = unwrap_experts_lora(wrapper)
        assert base is base_experts
        assert "gate_up_proj" in lora_dict
        assert "down_proj" in lora_dict

        gup_lA, gup_lB, gup_s = lora_dict["gate_up_proj"]
        assert torch.equal(gup_lA, gup_A)
        assert torch.equal(gup_lB, gup_B)
        assert gup_s == 0.25

        down_lA, down_lB, down_s = lora_dict["down_proj"]
        assert torch.equal(down_lA, down_A)
        assert torch.equal(down_lB, down_B)
        assert down_s == 0.25

    def test_partial_lora(self):
        """Only gate_up_proj has LoRA, down_proj does not."""
        base_experts = MagicMock(spec=["gate_up_proj", "down_proj"])
        del base_experts.base_layer
        del base_experts.lora_A

        gup_A = torch.randn(16, 64)
        gup_B = torch.randn(256, 16)
        gup_wrapper = _make_mock_lora_module(gup_A, gup_B, 0.5, "gate_up_proj")
        gup_wrapper.base_layer = base_experts

        base, lora_dict = unwrap_experts_lora(gup_wrapper)
        assert base is base_experts
        assert "gate_up_proj" in lora_dict
        assert "down_proj" not in lora_dict


# =============================================================================
# Tests: MoELoRAMaterialize
# =============================================================================


class TestMoELoRAMaterialize:
    @pytest.fixture()
    def setup(self):
        E, dim1, dim2, r = 4, 32, 16, 4
        scaling = 0.5
        W = torch.randn(E, dim1, dim2, dtype=torch.float64, requires_grad=False)
        A = torch.randn(r * E, dim2, dtype=torch.float64, requires_grad=True)
        B = torch.randn(dim1, r * E, dtype=torch.float64, requires_grad=True)
        return W, A, B, scaling, E, r

    def test_forward_shape(self, setup):
        W, A, B, scaling, E, r = setup
        W_eff = MoELoRAMaterialize.apply(W, A, B, scaling)
        assert W_eff.shape == W.shape

    def test_forward_correctness(self, setup):
        W, A, B, scaling, E, r = setup
        W_eff = MoELoRAMaterialize.apply(W, A, B, scaling)

        # Manual per-expert computation.
        # lora_A is expert-major: [r*E, dim2] -> rows [e*r:(e+1)*r] = expert e
        # lora_B is rank-major:   [dim1, r*E] -> reshape [dim1, r, E], slice [:, :, e]
        _, dim1, dim2 = W.shape
        expected = W.clone()
        B_3d = B.reshape(dim1, r, E)
        for e in range(E):
            A_e = A[e * r : (e + 1) * r, :]  # [r, dim2]
            B_e = B_3d[:, :, e]  # [dim1, r]
            expected[e] += scaling * (B_e @ A_e)

        assert torch.allclose(W_eff, expected, atol=1e-10)

    def test_backward_gradcheck(self, setup):
        W, A, B, scaling, E, r = setup
        # gradcheck requires float64
        assert torch.autograd.gradcheck(
            lambda a, b: MoELoRAMaterialize.apply(W, a, b, scaling),
            (A, B),
            eps=1e-6,
            atol=1e-4,
        )

    def test_no_grad_for_base_weight(self, setup):
        W, A, B, scaling, E, r = setup
        W.requires_grad_(True)
        W_eff = MoELoRAMaterialize.apply(W, A, B, scaling)
        loss = W_eff.sum()
        loss.backward()
        assert W.grad is None
        assert A.grad is not None
        assert B.grad is not None

    def test_scaling_zero(self, setup):
        W, A, B, _, E, r = setup
        W_eff = MoELoRAMaterialize.apply(W, A, B, 0.0)
        assert torch.allclose(W_eff, W)

    def test_gate_up_proj_shapes(self):
        """Test with realistic gate_up_proj shapes [E, 2*I, H]."""
        E, I2, H, r = 8, 512, 256, 16
        W = torch.randn(E, I2, H, dtype=torch.float64)
        A = torch.randn(r * E, H, dtype=torch.float64, requires_grad=True)
        B = torch.randn(I2, r * E, dtype=torch.float64, requires_grad=True)
        W_eff = MoELoRAMaterialize.apply(W, A, B, 1.0)
        assert W_eff.shape == (E, I2, H)
        loss = W_eff.sum()
        loss.backward()
        assert A.grad.shape == A.shape
        assert B.grad.shape == B.shape

    def test_down_proj_shapes(self):
        """Test with realistic down_proj shapes [E, H, I]."""
        E, H, I, r = 8, 256, 512, 16  # noqa: E741
        W = torch.randn(E, H, I, dtype=torch.float64)
        A = torch.randn(r * E, I, dtype=torch.float64, requires_grad=True)
        B = torch.randn(H, r * E, dtype=torch.float64, requires_grad=True)
        W_eff = MoELoRAMaterialize.apply(W, A, B, 1.0)
        assert W_eff.shape == (E, H, I)
        loss = W_eff.sum()
        loss.backward()
        assert A.grad.shape == A.shape
        assert B.grad.shape == B.shape


# =============================================================================
# Tests: materialize_expert_lora
# =============================================================================


class TestMaterializeExpertLora:
    def test_none_passthrough(self):
        W = torch.randn(4, 32, 16)
        result = materialize_expert_lora(W, None)
        assert result is W

    def test_with_lora(self):
        E, dim1, dim2, r = 4, 32, 16, 4
        W = torch.randn(E, dim1, dim2)
        A = torch.randn(r * E, dim2, requires_grad=True)
        B = torch.randn(dim1, r * E, requires_grad=True)
        result = materialize_expert_lora(W, (A, B, 0.5))
        assert result.shape == W.shape
        assert not torch.equal(result, W)
