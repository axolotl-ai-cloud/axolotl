"""Unit tests for GRPO trainer selection and FP8 quantization helpers."""

import unittest
from unittest.mock import MagicMock

import torch


class TestGRPOStrategyConflict(unittest.TestCase):
    """Tests for context_parallel + async_grpo conflict detection."""

    def test_raises_on_both_enabled(self):
        from axolotl.core.trainers.grpo import GRPOStrategy

        with self.assertRaises(ValueError) as ctx:
            GRPOStrategy.get_trainer_class(context_parallel=True, async_grpo=True)
        self.assertIn("context_parallel", str(ctx.exception))
        self.assertIn("async_grpo", str(ctx.exception))

    def test_sequence_parallel_only(self):
        from axolotl.core.trainers.grpo import GRPOStrategy
        from axolotl.core.trainers.grpo.trainer import (
            AxolotlGRPOContextParallelTrainer,
        )

        cls = GRPOStrategy.get_trainer_class(context_parallel=True, async_grpo=False)
        self.assertIs(cls, AxolotlGRPOContextParallelTrainer)

    def test_async_only(self):
        from axolotl.core.trainers.grpo import GRPOStrategy
        from axolotl.core.trainers.grpo.trainer import AxolotlAsyncGRPOTrainer

        cls = GRPOStrategy.get_trainer_class(context_parallel=False, async_grpo=True)
        self.assertIs(cls, AxolotlAsyncGRPOTrainer)

    def test_neither(self):
        from axolotl.core.trainers.grpo import GRPOStrategy
        from axolotl.core.trainers.grpo.trainer import AxolotlGRPOTrainer

        cls = GRPOStrategy.get_trainer_class(context_parallel=False, async_grpo=False)
        self.assertIs(cls, AxolotlGRPOTrainer)


class TestQuantizeFP8Roundtrip(unittest.TestCase):
    def _roundtrip(self, shape, scale_shape):
        import torch

        from axolotl.kernels.quantize import dequantize_fp8, quantize_fp8

        W = torch.randn(*shape) * 3
        q, scale_inv = quantize_fp8(W, torch.ones(scale_shape))
        self.assertEqual(q.dtype, torch.float8_e4m3fn)
        self.assertEqual(scale_inv.shape, torch.Size(scale_shape))
        back = dequantize_fp8(q, scale_inv, torch.float32)
        torch.testing.assert_close(back, W, rtol=0.07, atol=0.05)

    def test_block_scales(self):
        self._roundtrip((8, 8), (2, 4))

    def test_tail_blocks(self):
        self._roundtrip((7, 9), (2, 3))

    def test_per_tensor_scale(self):
        self._roundtrip((4, 6), (1,))


class TestDequantizeFP8TailBlocks(unittest.TestCase):
    """Tests for FP8 dequantization with non-divisible dimensions."""

    def test_exact_divisible_shape(self):
        from axolotl.kernels.quantize import dequantize_fp8

        W = torch.randn(256, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_inv = torch.ones(2, 1, dtype=torch.bfloat16)
        result = dequantize_fp8(W, scale_inv)
        self.assertEqual(result.shape, (256, 128))
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_non_divisible_rows(self):
        from axolotl.kernels.quantize import dequantize_fp8

        # 130 rows, scale has 2 blocks (block_size ~65 for exact div, but with
        # tail blocks: first block=65 rows, second=65 rows, 130%2=0 actually).
        # Use 131 rows with 2 scale blocks to trigger tail handling.
        W = torch.ones(131, 128, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_inv = torch.tensor([[2.0], [3.0]], dtype=torch.bfloat16)
        result = dequantize_fp8(W, scale_inv)
        self.assertEqual(result.shape, (131, 128))
        self.assertEqual(result.dtype, torch.bfloat16)

    def test_non_divisible_cols(self):
        from axolotl.kernels.quantize import dequantize_fp8

        W = torch.ones(128, 200, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_inv = torch.ones(1, 2, dtype=torch.bfloat16)
        result = dequantize_fp8(W, scale_inv)
        self.assertEqual(result.shape, (128, 200))

    def test_scalar_scale(self):
        from axolotl.kernels.quantize import dequantize_fp8

        W = torch.ones(64, 64, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        scale_inv = torch.tensor(2.0, dtype=torch.bfloat16)
        result = dequantize_fp8(W, scale_inv)
        self.assertEqual(result.shape, (64, 64))


class TestLoraFP8Guard(unittest.TestCase):
    """Tests that get_lora_parameters only uses weight_scale_inv for FP8 weights."""

    def test_non_fp8_weight_skips_scale_inv(self):
        """Non-FP8 weight should NOT pick up weight_scale_inv as quant_state."""
        from axolotl.kernels.lora import get_lora_parameters

        proj = MagicMock()
        proj.disable_adapters = True
        base_layer = MagicMock(spec=[])  # empty spec to control attrs precisely

        # Use a real tensor for weight (bf16, no quant_state attr)
        base_layer.weight = torch.randn(64, 64, dtype=torch.bfloat16)
        base_layer.bias = None
        base_layer.weight_scale_inv = torch.ones(1)  # should NOT be used for bf16

        proj.base_layer = base_layer

        W, b, quant_state, A, B, s, *_ = get_lora_parameters(proj)
        # quant_state should be None since weight is bf16, not FP8
        self.assertIsNone(quant_state)

    def test_fp8_weight_uses_scale_inv(self):
        """FP8 weight should pick up weight_scale_inv as quant_state."""
        from axolotl.kernels.lora import get_lora_parameters

        proj = MagicMock()
        proj.disable_adapters = True
        base_layer = MagicMock()
        proj.base_layer = base_layer

        # FP8 weight
        base_layer.weight = torch.randn(64, 64, dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        )
        base_layer.bias = None
        scale_inv = torch.ones(1)
        base_layer.weight_scale_inv = scale_inv

        W, b, quant_state, A, B, s, *_ = get_lora_parameters(proj)
        self.assertIs(quant_state, scale_inv)


if __name__ == "__main__":
    unittest.main()
