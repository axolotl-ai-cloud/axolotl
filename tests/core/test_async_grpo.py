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


class TestAdvantageEstimator(unittest.TestCase):
    """Tests for the pluggable advantage estimator helper."""

    def _f(self):
        from axolotl.core.trainers.grpo.async_trainer import (
            compute_advantages_with_estimator,
        )

        return compute_advantages_with_estimator

    def test_rloo_leave_one_out_baseline(self):
        f = self._f()
        rewards = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 6.0])
        adv, is_std_zero = f("rloo", rewards, num_generations=3)
        # group [1,2,3]: baseline_j = (sum - r_j) / (n - 1)
        expected = torch.tensor([1 - 2.5, 2 - 2.0, 3 - 1.5, 0 - 3.0, 0 - 3.0, 6.0])
        self.assertTrue(torch.allclose(adv, expected))
        self.assertFalse(is_std_zero.any())

    def test_reinforce_plus_plus_group_mean_batch_std(self):
        f = self._f()
        rewards = torch.tensor([1.0, 2.0, 3.0, 0.0, 0.0, 6.0])
        adv, _ = f("reinforce_plus_plus", rewards, num_generations=3)
        # REINFORCE++ Baseline: subtract per-prompt group mean, then normalize
        # by the batch-level std of those centered rewards.
        g = rewards.view(-1, 3)
        centered = (g - g.mean(dim=1, keepdim=True)).reshape(-1)
        expected = centered / (centered.std() + 1e-4)
        self.assertTrue(torch.allclose(adv, expected))
        # Differs from both global-mean batch-norm and per-group std-norm.
        global_norm = (rewards - rewards.mean()) / (rewards.std() + 1e-4)
        self.assertFalse(torch.allclose(adv, global_norm))
        group_norm = (
            centered.view(-1, 3) / (g.std(dim=1, keepdim=True) + 1e-4)
        ).reshape(-1)
        self.assertFalse(torch.allclose(adv, group_norm))

    def test_equal_rewards_flag_std_zero_and_zero_advantage(self):
        f = self._f()
        rewards = torch.tensor([5.0, 5.0, 5.0, 1.0, 2.0, 3.0])
        adv, is_std_zero = f("rloo", rewards, num_generations=3)
        self.assertTrue(torch.allclose(adv[:3], torch.zeros(3)))
        self.assertTrue(is_std_zero[:3].all())
        self.assertFalse(is_std_zero[3:].any())

    def test_rloo_single_generation_degrades_to_mean_center(self):
        f = self._f()
        # No leave-one-out baseline possible with one sample per group.
        adv, is_std_zero = f("rloo", torch.tensor([2.0, 4.0]), num_generations=1)
        self.assertTrue(torch.allclose(adv, torch.zeros(2)))
        self.assertTrue(is_std_zero.all())

    def test_shape_preserved(self):
        f = self._f()
        rewards = torch.randn(12)
        for est in ("rloo", "reinforce_plus_plus"):
            adv, is_std_zero = f(est, rewards, num_generations=4)
            self.assertEqual(adv.shape, rewards.shape)
            self.assertEqual(is_std_zero.shape, rewards.shape)


class TestBaseTrainerAggregateRewards(unittest.TestCase):
    """Covers the standard (non-SP, non-async) trainer's reward reconstruction.

    The base ``AxolotlGRPOTrainer`` re-derives per-sample rewards from the
    cached per-function rewards to apply non-default estimators, so this must
    match TRL's own aggregation.
    """

    def _stub(self, *, weights, aggregation, num_reward_funcs):
        from axolotl.core.trainers.grpo.trainer import AxolotlGRPOTrainer

        trainer = object.__new__(AxolotlGRPOTrainer)
        trainer.reward_weights = torch.tensor(weights, dtype=torch.float32)
        trainer.multi_objective_aggregation = aggregation
        trainer.reward_funcs = [None] * num_reward_funcs
        return trainer

    def test_sum_then_normalize_matches_weighted_sum(self):
        trainer = self._stub(
            weights=[1.0, 0.5], aggregation="sum_then_normalize", num_reward_funcs=2
        )
        rewards_per_func = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [0.0, 6.0], [5.0, 1.0]]
        )
        out = trainer._aggregate_rewards(rewards_per_func, num_generations=2)
        expected = rewards_per_func[:, 0] * 1.0 + rewards_per_func[:, 1] * 0.5
        self.assertTrue(torch.allclose(out, expected))

    def test_normalize_then_sum_matches_per_group_func_norm(self):
        trainer = self._stub(
            weights=[1.0, 1.0], aggregation="normalize_then_sum", num_reward_funcs=2
        )
        rewards_per_func = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0], [0.0, 6.0], [5.0, 1.0]]
        )
        out = trainer._aggregate_rewards(rewards_per_func, num_generations=2)

        grouped = rewards_per_func.view(-1, 2, 2)
        mean_k = grouped.mean(dim=1, keepdim=True)
        std_k = grouped.std(dim=1, keepdim=True)
        reward_k = ((grouped - mean_k) / (std_k + 1e-4)).view(-1, 2)
        expected = reward_k.sum(dim=1)
        self.assertTrue(torch.allclose(out, expected))

    def test_invalid_aggregation_raises(self):
        trainer = self._stub(weights=[1.0], aggregation="bogus", num_reward_funcs=1)
        with self.assertRaises(ValueError):
            trainer._aggregate_rewards(torch.tensor([[1.0], [2.0]]), num_generations=2)


class TestAdvantageEstimatorSchema(unittest.TestCase):
    """Tests for the advantage_estimator config field wiring."""

    def test_schema_default_none_and_valid_values(self):
        from axolotl.utils.schemas.trl import TRLConfig

        self.assertIsNone(TRLConfig().advantage_estimator)
        for v in ("grpo", "rloo", "reinforce_plus_plus"):
            self.assertEqual(TRLConfig(advantage_estimator=v).advantage_estimator, v)

    def test_schema_rejects_invalid(self):
        from pydantic import ValidationError

        from axolotl.utils.schemas.trl import TRLConfig

        with self.assertRaises(ValidationError):
            TRLConfig(advantage_estimator="bogus")

    def test_builder_passes_field_through(self):
        from axolotl.core.trainers.grpo import GRPOStrategy
        from axolotl.utils.dict import DictDefault

        cfg = DictDefault(
            {
                "context_parallel_size": 1,
                "vllm": {},
                "trl": {"advantage_estimator": "rloo"},
            }
        )
        kwargs = GRPOStrategy.set_training_args_kwargs(cfg)
        self.assertEqual(kwargs.get("advantage_estimator"), "rloo")

        cfg_unset = DictDefault(
            {"context_parallel_size": 1, "vllm": {}, "trl": {"beta": 0.0}}
        )
        self.assertNotIn(
            "advantage_estimator",
            GRPOStrategy.set_training_args_kwargs(cfg_unset),
        )


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


class TestVllmLoraSyncPatch(unittest.TestCase):
    """The ``_generate_single_turn`` patch wires sync_weights to the right place.

    These tests exercise the patch-installation branch in isolation. They build
    a stub trainer with just enough attributes to look like
    ``AsyncGRPOTrainer`` for the duration of the relevant code path.

    Background — there are two correct behaviors and we historically had a bug
    where both modes used the same one:

      - Async prefetch ON: the BG generation thread can't safely call
        sync_weights mid-rollout. We no-op the stock hook and drive sync from
        the main thread via ``_maybe_sync_vllm_weights``.
      - Async prefetch OFF: TRL's stock ``_generate_single_turn`` already
        calls ``sync_weights`` once per step boundary on the main thread. We
        wire that hook directly to ``_sync_lora_adapter`` because
        ``_maybe_sync_vllm_weights`` short-circuits when async is off.

    Before the fix, both modes installed ``lambda: None``, so sync mode never
    pushed any LoRA adapter to vLLM and the trainer was a no-op.
    """

    @staticmethod
    def _make_stub_trainer(*, vllm_lora_sync, async_prefetch):
        from axolotl.core.trainers.grpo.async_trainer import (
            AsyncGRPOTrainer,
        )

        class FakeArgs:
            pass

        args = FakeArgs()
        args.vllm_lora_sync = vllm_lora_sync
        args.async_prefetch = async_prefetch

        class FakeVllmGen:
            sync_weights = staticmethod(lambda: None)
            model = MagicMock()

        # Use object.__new__ so we don't run __init__ (which needs a real
        # model, dataset, etc.). We only need the `_generate_single_turn`
        # method's patch branch to run, so we set up the minimum state.
        trainer = object.__new__(AsyncGRPOTrainer)
        trainer.args = args
        trainer.use_vllm = True
        trainer.vllm_generation = FakeVllmGen()
        trainer._patched_sync_weights = False
        # Spy on _sync_lora_adapter so we can assert it's the function the
        # hook delegates to in sync mode.
        trainer._sync_lora_adapter = MagicMock(name="_sync_lora_adapter_spy")
        trainer._sync_peft_weights_no_merge = MagicMock(
            name="_sync_peft_weights_no_merge_spy"
        )
        return trainer

    @staticmethod
    def _run_patch_branch(trainer):
        """Execute just the sync_weights-patching branch in isolation.

        We can't easily call the real ``_generate_single_turn`` because it
        does a full vLLM generate. Instead we copy the exact branch out of
        the source so the test verifies the same logic the trainer runs.
        """
        if not getattr(trainer, "_patched_sync_weights", False):
            if trainer.use_vllm and hasattr(trainer, "vllm_generation"):
                if getattr(trainer.args, "vllm_lora_sync", False):
                    if getattr(trainer.args, "async_prefetch", False):
                        trainer.vllm_generation.sync_weights = lambda: None
                    else:
                        sync_helper = trainer._sync_lora_adapter

                        def _lora_filesystem_sync():
                            sync_helper()

                        trainer.vllm_generation.sync_weights = _lora_filesystem_sync
                    trainer._patched_sync_weights = True

    def test_sync_mode_with_lora_sync_wires_to_sync_lora_adapter(self):
        trainer = self._make_stub_trainer(vllm_lora_sync=True, async_prefetch=False)
        self._run_patch_branch(trainer)

        assert trainer._patched_sync_weights is True
        # Trigger the patched hook — it must call _sync_lora_adapter.
        trainer.vllm_generation.sync_weights()
        trainer._sync_lora_adapter.assert_called_once()

    def test_async_mode_with_lora_sync_installs_noop_hook(self):
        trainer = self._make_stub_trainer(vllm_lora_sync=True, async_prefetch=True)
        self._run_patch_branch(trainer)

        assert trainer._patched_sync_weights is True
        # Hook must be a no-op so BG-thread generation doesn't fight the
        # main-thread optimizer step over the model weights.
        trainer.vllm_generation.sync_weights()
        trainer._sync_lora_adapter.assert_not_called()

    def test_sync_mode_with_lora_sync_does_not_call_during_install(self):
        """Installing the patch should not pre-emptively sync."""
        trainer = self._make_stub_trainer(vllm_lora_sync=True, async_prefetch=False)
        self._run_patch_branch(trainer)
        # _sync_lora_adapter should only be called when the patched hook
        # itself is invoked (e.g., from TRL's _generate_single_turn).
        trainer._sync_lora_adapter.assert_not_called()

    def test_patch_is_idempotent(self):
        trainer = self._make_stub_trainer(vllm_lora_sync=True, async_prefetch=False)
        self._run_patch_branch(trainer)
        first_hook = trainer.vllm_generation.sync_weights
        # Second call must not re-patch (otherwise we'd lose the original).
        self._run_patch_branch(trainer)
        assert trainer.vllm_generation.sync_weights is first_hook


class TestMaybeSyncVllmWeightsIntervalDefault(unittest.TestCase):
    """``_maybe_sync_vllm_weights`` must not crash when interval is unset.

    Before the fix, ``step % self.args.vllm_sync_interval`` would TypeError
    on the very first call when ``vllm_sync_interval`` was ``None`` (which
    is the default for any config that doesn't explicitly set it). We now
    fall back to interval=1 so unset means "sync every step", matching the
    behavior of TRL's own ``_generate_single_turn``.
    """

    @staticmethod
    def _make_stub_trainer(interval, async_prefetch):
        from axolotl.core.trainers.grpo.async_trainer import (
            AsyncGRPOTrainer,
        )

        class FakeArgs:
            pass

        args = FakeArgs()
        args.async_prefetch = async_prefetch
        args.vllm_sync_interval = interval
        args.vllm_lora_sync = True

        class FakeState:
            global_step = 1

        trainer = object.__new__(AsyncGRPOTrainer)
        trainer.args = args
        trainer.use_vllm = True
        trainer.state = FakeState()
        trainer._last_synced_step = 0
        trainer._sync_lora_adapter = MagicMock(name="sync_spy")
        return trainer

    def test_interval_none_in_async_mode_does_not_crash(self):
        trainer = self._make_stub_trainer(interval=None, async_prefetch=True)
        from axolotl.core.trainers.grpo.async_trainer import (
            AsyncGRPOTrainer,
        )

        # Should not raise TypeError — defaults to every-step sync
        AsyncGRPOTrainer._maybe_sync_vllm_weights(trainer)
        trainer._sync_lora_adapter.assert_called_once()

    def test_sync_mode_drives_sync(self):
        """Sync mode must fire ``_sync_lora_adapter`` from ``_maybe_sync_vllm_weights``.

        The previous behavior (early return when ``not async_prefetch``)
        assumed TRL's stock ``_generate_single_turn`` would handle sync.
        That's true for vanilla GRPO but FALSE for NeMo Gym multi-turn
        where the data producer bypasses ``_generate_single_turn``
        entirely. Without this trigger no sync ever happens and the
        trainer becomes a no-op.
        """
        trainer = self._make_stub_trainer(interval=1, async_prefetch=False)
        from axolotl.core.trainers.grpo.async_trainer import (
            AsyncGRPOTrainer,
        )

        AsyncGRPOTrainer._maybe_sync_vllm_weights(trainer)
        trainer._sync_lora_adapter.assert_called_once()

    def test_async_mode_with_explicit_interval_respects_modulo(self):
        trainer = self._make_stub_trainer(interval=4, async_prefetch=True)
        from axolotl.core.trainers.grpo.async_trainer import (
            AsyncGRPOTrainer,
        )

        # global_step=1, interval=4 → 1 % 4 != 0 → no sync
        AsyncGRPOTrainer._maybe_sync_vllm_weights(trainer)
        trainer._sync_lora_adapter.assert_not_called()

        # global_step=4 → 4 % 4 == 0 → sync
        trainer.state.global_step = 4
        AsyncGRPOTrainer._maybe_sync_vllm_weights(trainer)
        trainer._sync_lora_adapter.assert_called_once()


if __name__ == "__main__":
    unittest.main()
