import importlib.util
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn

from axolotl.kernels.lora import get_lora_parameters

PEFT_AVAILABLE = importlib.util.find_spec("peft") is not None


class TestLoRAParameterFreezing:
    """Test suite for LoRA parameter freezing validation."""

    def setup_method(self):
        self.dtype = torch.float32

    def create_mock_lora_layer(
        self, has_adapters=True, adapters_disabled=False, merged=False
    ):
        """Create a mock LoRA layer for testing."""
        mock_layer = Mock()

        base_layer = Mock()
        base_layer.weight = torch.randn(512, 256, dtype=self.dtype)
        base_layer.bias = torch.randn(512, dtype=self.dtype)

        if has_adapters:
            mock_layer.base_layer = base_layer
            mock_layer.disable_adapters = adapters_disabled
            mock_layer.merged = merged

            mock_layer.active_adapters = ["default"]
            mock_layer.lora_A = {"default": Mock()}
            mock_layer.lora_B = {"default": Mock()}
            mock_layer.scaling = {"default": 0.1}

            mock_layer.lora_A["default"].weight = torch.randn(16, 256, dtype=self.dtype)
            mock_layer.lora_B["default"].weight = torch.randn(512, 16, dtype=self.dtype)
        else:
            mock_layer.weight = base_layer.weight
            mock_layer.bias = base_layer.bias

        return mock_layer

    def test_parameter_freezing_adapters_disabled(self):
        """Test that LoRA parameters are None when adapters are disabled."""
        layer = self.create_mock_lora_layer(has_adapters=True, adapters_disabled=True)

        W, b, A, B, s = get_lora_parameters(layer)

        # Base parameters should be returned
        assert W is not None
        assert b is not None
        # LoRA parameters should be None (frozen)
        assert A is None
        assert B is None
        assert s is None

    def test_parameter_freezing_adapters_merged(self):
        """Test that LoRA parameters are None when adapters are merged."""
        layer = self.create_mock_lora_layer(has_adapters=True, merged=True)

        W, b, A, B, s = get_lora_parameters(layer)

        # Base parameters should be returned
        assert W is not None
        assert b is not None

        # LoRA parameters should be None (frozen)
        assert A is None
        assert B is None
        assert s is None

    def test_parameter_freezing_no_adapters(self):
        """Test parameter behavior when no adapters are present."""
        layer = self.create_mock_lora_layer(has_adapters=False)

        W, b, A, B, s = get_lora_parameters(layer)

        # Base parameters should be returned
        assert W is not None
        assert b is not None

        # LoRA parameters should be None (frozen)
        assert A is None
        assert B is None
        assert s is None

    def test_parameter_active_adapters_enabled(self):
        """Test that LoRA parameters are returned when adapters are active."""
        layer = self.create_mock_lora_layer(
            has_adapters=True, adapters_disabled=False, merged=False
        )

        W, b, A, B, s = get_lora_parameters(layer)

        # All parameters should be returned
        assert W is not None
        assert b is not None
        assert A is not None
        assert B is not None
        assert s is not None
        assert s == 0.1

    def test_parameter_shapes_consistency(self):
        """Test that parameter shapes are consistent when active."""
        layer = self.create_mock_lora_layer(
            has_adapters=True, adapters_disabled=False, merged=False
        )

        W, b, A, B, s = get_lora_parameters(layer)

        # Check shape consistency
        assert W.shape == (512, 256)
        assert b.shape == (512,)
        assert A.shape == (16, 256)
        assert B.shape == (512, 16)

    def test_parameter_dtypes_consistency(self):
        """Test that parameter dtypes are consistent."""
        layer = self.create_mock_lora_layer(
            has_adapters=True, adapters_disabled=False, merged=False
        )

        W, b, quant_state, A, B, s = get_lora_parameters(layer)

        assert W.dtype == self.dtype
        assert b.dtype == self.dtype
        assert A.dtype == self.dtype
        assert B.dtype == self.dtype

    def test_quantization_state_handling(self):
        """Test that quantization state is properly handled."""
        layer = self.create_mock_lora_layer(has_adapters=True)

        quant_state_mock = Mock()
        layer.base_layer.weight.quant_state = quant_state_mock

        W, b, quant_state, A, B, s = get_lora_parameters(layer)

        assert quant_state == quant_state_mock

    def test_multiple_adapters_active_adapter_selection(self):
        """Test that the correct adapter is selected when multiple adapters exist."""
        layer = self.create_mock_lora_layer(
            has_adapters=True, adapters_disabled=False, merged=False
        )

        layer.lora_A["adapter2"] = Mock()
        layer.lora_B["adapter2"] = Mock()
        layer.scaling["adapter2"] = 0.2

        layer.lora_A["adapter2"].weight = torch.randn(16, 256, dtype=self.dtype)
        layer.lora_B["adapter2"].weight = torch.randn(512, 16, dtype=self.dtype)

        layer.active_adapters = ["adapter2"]

        W, b, A, B, s = get_lora_parameters(layer)

        assert s == 0.2
        assert torch.equal(A, layer.lora_A["adapter2"].weight)
        assert torch.equal(B, layer.lora_B["adapter2"].weight)


class TestLoRAParameterFreezingIntegration:
    """Integration tests for parameter freezing with actual LoRA layers."""

    @pytest.mark.skipif(
        not PEFT_AVAILABLE, reason="PEFT not available for integration tests"
    )
    def test_parameter_freezing_with_real_lora_layer(self):
        """Test parameter freezing with actual PEFT LoRA layer."""
        from peft import LoraConfig, get_peft_model

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(256, 512)

            def forward(self, x):
                return self.linear(x)

        base_model = SimpleModel()
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["linear"],
            lora_dropout=0.1,
        )
        model = get_peft_model(base_model, lora_config)
        lora_layer = model.base_model.model.linear
        # Test with adapters enabled
        W, b, A, B, s = get_lora_parameters(lora_layer)
        assert A is not None
        assert B is not None
        assert s is not None
        # Test with adapters disabled
        model.disable_adapter_layers()
        W, b, A, B, s = get_lora_parameters(lora_layer)
        assert A is None
        assert B is None
        assert s is None

    @pytest.mark.skipif(
        not PEFT_AVAILABLE, reason="PEFT not available for integration tests"
    )
    def test_parameter_freezing_gradient_behavior(self):
        """Test that frozen parameters don't receive gradients."""
        from peft import LoraConfig, get_peft_model

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(256, 512)

            def forward(self, x):
                return self.linear(x)

        base_model = SimpleModel()
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["linear"],
            lora_dropout=0.1,
        )
        model = get_peft_model(base_model, lora_config)
        x = torch.randn(1, 256)
        target = torch.randn(1, 512)
        model.enable_adapter_layers()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        lora_layer = model.base_model.model.linear
        has_lora_grads = any(
            param.grad is not None
            for name, param in lora_layer.named_parameters()
            if "lora_" in name
        )
        assert has_lora_grads, (
            "LoRA parameters should have gradients when adapters are enabled"
        )
        model.zero_grad()
        model.disable_adapter_layers()
        output = model(x)
        loss = nn.MSELoss()(output, target)
        any_requires_grad = any(param.requires_grad for param in model.parameters())
        if any_requires_grad:
            loss.backward()
        has_lora_grads_disabled = any(
            param.grad is not None
            for name, param in lora_layer.named_parameters()
            if "lora_" in name
        )
        assert not has_lora_grads_disabled, (
            "LoRA parameters should not have gradients when adapters are disabled"
        )
        model.zero_grad()
        del model, base_model, lora_layer, x, target, output, loss
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
