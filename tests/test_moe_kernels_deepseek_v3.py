"""
Unit tests for DeepSeek v3 MoE kernel optimizations.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from axolotl.integrations.moe_kernels.plugin import (
    MoeOptimizedPlugin,
    apply_moe_kernel_patches,
    patch_deepseek_v3_moe_forward_optimized,
)
from axolotl.utils.dict import DictDefault


class TestDeepSeekV3MoeKernels(unittest.TestCase):
    """Test cases for DeepSeek v3 MoE kernel optimizations."""

    def test_patch_deepseek_v3_function(self):
        """Test that the DeepSeek v3 patch function works correctly."""

        # Mock the DeepseekV3MoE class
        mock_module = MagicMock()
        mock_moe_class = MagicMock()
        mock_module.DeepseekV3MoE = mock_moe_class

        with patch.dict(
            "sys.modules",
            {"transformers.models.deepseek_v3.modeling_deepseek_v3": mock_module},
        ):
            patch_deepseek_v3_moe_forward_optimized()

        # Verify methods were patched
        self.assertIsNotNone(mock_moe_class.forward)
        self.assertIsNotNone(mock_moe_class.moe)

    def test_apply_moe_patches_includes_deepseek_v3(self):
        """Test that apply_moe_kernel_patches includes DeepSeek v3."""

        with patch(
            "axolotl.integrations.moe_kernels.plugin.patch_deepseek_v3_moe_forward_optimized"
        ) as mock_patch:
            apply_moe_kernel_patches(models=["deepseek_v3"])
            mock_patch.assert_called_once()

    def test_plugin_detects_deepseek_v3_base_model(self):
        """Test plugin detection for DeepSeek v3 base models."""

        plugin = MoeOptimizedPlugin()

        # Test various DeepSeek v3 model configurations
        test_configs = [
            {"base_model": "deepseek-ai/DeepSeek-V3", "moe_kernels": True},
            {"base_model": "axolotl-ai-co/DeepSeek-V3-11M", "moe_kernels": True},
            {"base_model": "some-org/deepseek-v3-custom", "moe_kernels": True},
        ]

        for cfg in test_configs:
            plugin.register(cfg)
            self.assertTrue(plugin.enabled, f"Plugin should be enabled for {cfg}")

    def test_plugin_detects_deepseek_v3_model_type(self):
        """Test plugin detection for DeepSeek v3 model type."""

        plugin = MoeOptimizedPlugin()
        cfg = DictDefault({"model_type": "deepseek_v3", "moe_kernels": True})

        plugin.register(cfg)
        self.assertTrue(plugin.enabled)

        # Mock the pre_model_load to test auto-detection
        with patch(
            "axolotl.integrations.moe_kernels.plugin.apply_moe_kernel_patches"
        ) as mock_apply:
            plugin.pre_model_load(cfg)
            mock_apply.assert_called_once()
            call_args = mock_apply.call_args
            self.assertIn("deepseek_v3", call_args[1]["models"])

    def test_plugin_explicit_model_list(self):
        """Test that explicit model list includes DeepSeek v3."""

        plugin = MoeOptimizedPlugin()
        cfg = {"moe_kernels": True, "moe_kernel_models": ["deepseek_v3", "mixtral"]}

        plugin.register(cfg)
        self.assertEqual(plugin.models, ["deepseek_v3", "mixtral"])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA not available")
    def test_deepseek_v3_kernel_integration(self):
        """Integration test for DeepSeek v3 kernel execution (requires CUDA)."""

        from axolotl.kernels.moe import cg_grouped_gemm_forward

        # Create test tensors
        batch_size, hidden_dim = 8, 256
        num_experts, expert_dim = 256, 512  # DeepSeek v3 has 256 experts

        inputs = torch.randn(batch_size, hidden_dim, dtype=torch.float16, device="cuda")
        expert_weights = torch.randn(
            num_experts, expert_dim, hidden_dim, dtype=torch.float16, device="cuda"
        )
        expert_indices = torch.randint(0, num_experts, (batch_size,), device="cuda")

        # Execute kernel
        output = cg_grouped_gemm_forward(
            inputs, expert_weights, expert_indices, group_size_m=64
        )

        # Verify output shape
        self.assertEqual(output.shape, (batch_size, expert_dim))
        self.assertEqual(output.dtype, torch.float16)
        self.assertEqual(output.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
