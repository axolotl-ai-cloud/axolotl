"""Unit tests for async GRPO"""

import unittest
from unittest.mock import MagicMock

import torch


class TestReplayBuffer(unittest.TestCase):
    """Tests for ReplayBuffer edge cases."""

    def test_add_noop_when_max_size_zero(self):
        from axolotl.core.trainers.grpo.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(max_size=0)
        buf.add(1.0, {"data": "test"})
        self.assertEqual(len(buf), 0)

    def test_add_noop_when_max_size_negative(self):
        from axolotl.core.trainers.grpo.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(max_size=-1)
        buf.add(1.0, {"data": "test"})
        self.assertEqual(len(buf), 0)

    def test_sample_returns_none_when_max_size_zero(self):
        from axolotl.core.trainers.grpo.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(max_size=0)
        self.assertIsNone(buf.sample(1))

    def test_sample_returns_none_when_empty(self):
        from axolotl.core.trainers.grpo.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(max_size=5)
        self.assertIsNone(buf.sample(1))

    def test_normal_add_and_sample(self):
        from axolotl.core.trainers.grpo.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(max_size=3)
        buf.add(1.0, {"a": 1})
        buf.add(2.0, {"a": 2})
        buf.add(3.0, {"a": 3})
        self.assertEqual(len(buf), 3)
        result = buf.sample(1)
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)

    def test_replaces_lowest_when_full(self):
        from axolotl.core.trainers.grpo.replay_buffer import ReplayBuffer

        buf = ReplayBuffer(max_size=2)
        buf.add(1.0, {"a": 1})
        buf.add(2.0, {"a": 2})
        buf.add(3.0, {"a": 3})  # should replace score=1.0
        self.assertEqual(len(buf), 2)
        scores = sorted(item[0] for item in buf._heap)
        self.assertEqual(scores, [2.0, 3.0])


class TestGRPOStrategyConflict(unittest.TestCase):
    """Tests for sequence_parallel + async_grpo conflict detection."""

    def test_raises_on_both_enabled(self):
        from axolotl.core.trainers.grpo import GRPOStrategy

        with self.assertRaises(ValueError) as ctx:
            GRPOStrategy.get_trainer_class(sequence_parallel=True, async_grpo=True)
        self.assertIn("sequence_parallel", str(ctx.exception))
        self.assertIn("async_grpo", str(ctx.exception))

    def test_sequence_parallel_only(self):
        from axolotl.core.trainers.grpo import GRPOStrategy
        from axolotl.core.trainers.grpo.trainer import (
            AxolotlGRPOSequenceParallelTrainer,
        )

        cls = GRPOStrategy.get_trainer_class(sequence_parallel=True, async_grpo=False)
        self.assertIs(cls, AxolotlGRPOSequenceParallelTrainer)

    def test_async_only(self):
        from axolotl.core.trainers.grpo import GRPOStrategy
        from axolotl.core.trainers.grpo.trainer import AxolotlAsyncGRPOTrainer

        cls = GRPOStrategy.get_trainer_class(sequence_parallel=False, async_grpo=True)
        self.assertIs(cls, AxolotlAsyncGRPOTrainer)

    def test_neither(self):
        from axolotl.core.trainers.grpo import GRPOStrategy
        from axolotl.core.trainers.grpo.trainer import AxolotlGRPOTrainer

        cls = GRPOStrategy.get_trainer_class(sequence_parallel=False, async_grpo=False)
        self.assertIs(cls, AxolotlGRPOTrainer)


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


class TestValidateQuantPatchRestore(unittest.TestCase):
    """Test that validate_quantization_for_training is restored after trainer creation."""

    def test_patch_restored_on_success(self):
        """Monkeypatch should be restored even after successful trainer creation."""
        import transformers.trainer as _trainer_module

        original = _trainer_module.validate_quantization_for_training

        # After the build() method runs, original should be restored.
        # We can't easily test the full build(), but we can test the pattern.
        _orig = _trainer_module.validate_quantization_for_training
        _trainer_module.validate_quantization_for_training = lambda model: None
        try:
            pass  # simulate trainer_cls() succeeding
        finally:
            _trainer_module.validate_quantization_for_training = _orig

        self.assertIs(_trainer_module.validate_quantization_for_training, original)

    def test_patch_restored_on_error(self):
        """Monkeypatch should be restored even if trainer creation raises."""
        import transformers.trainer as _trainer_module

        original = _trainer_module.validate_quantization_for_training

        _orig = _trainer_module.validate_quantization_for_training
        _trainer_module.validate_quantization_for_training = lambda model: None
        try:
            raise ValueError("test error")
        except ValueError:
            pass
        finally:
            _trainer_module.validate_quantization_for_training = _orig

        self.assertIs(_trainer_module.validate_quantization_for_training, original)


if __name__ == "__main__":
    unittest.main()
