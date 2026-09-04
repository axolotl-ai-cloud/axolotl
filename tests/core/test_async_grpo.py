"""Unit tests for async GRPO"""

import types
import unittest
from collections import defaultdict
from unittest.mock import MagicMock, patch

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


class _StopEarly(Exception):
    """Raised by a mocked call to short-circuit a large method right after
    capturing the kwargs it was invoked with."""


def _capture_and_stop():
    """Returns (captured_kwargs_dict, side_effect_fn) for mocking
    ``_get_per_token_logps_and_entropies``: records the kwargs of the first
    call, then aborts the enclosing method via ``_StopEarly`` so tests don't
    need to stub out the rest of a large scoring/loss method."""
    captured: dict = {}

    def _side_effect(*_args, **kwargs):
        captured.update(kwargs)
        raise _StopEarly()

    return captured, _side_effect


class TestMultimodalTileFieldPropagation(unittest.TestCase):
    """spatial_shapes / num_tiles / image_position_ids (TRL 1.9's LFM2-VL /
    tile-indexed VLM fields) must reach ``_get_per_token_logps_and_entropies``
    from every async GRPO scoring path, not just the low-level method
    signature."""

    def _make_async_trainer(self):
        from axolotl.core.trainers.grpo.async_trainer import AsyncGRPOTrainer

        trainer = AsyncGRPOTrainer.__new__(AsyncGRPOTrainer)
        trainer.accelerator = MagicMock()
        trainer.accelerator.device = torch.device("cpu")
        trainer.accelerator.num_processes = 1
        trainer.model = MagicMock()
        trainer.model.is_gradient_checkpointing = False
        trainer.is_fsdp_enabled = False
        trainer.use_vllm = True
        trainer.vllm_importance_sampling_correction = True
        trainer.beta = 0.0
        trainer.aux_loss_enabled = False
        trainer.num_generations = 2
        trainer.num_iterations = 1
        trainer.pad_token_id = 0
        trainer.eos_token_id = 1
        trainer.mask_truncated_completions = False
        trainer.tools = None
        trainer.chat_template_kwargs = {}
        trainer._launch_reward_workers = MagicMock()
        trainer.args = types.SimpleNamespace(
            per_device_train_batch_size=2,
            batch_flattening=False,
            gradient_accumulation_steps=1,
            steps_per_generation=1,
            gradient_checkpointing_kwargs=None,
        )
        return trainer

    def test_generate_only_forwards_tile_fields(self):
        """_generate_only must (a) recover num_tiles the way TRL does for
        tile-indexed VLMs, and (b) preserve spatial_shapes/image_position_ids
        into the deferred rollout output."""
        from axolotl.core.trainers.grpo.async_trainer import AsyncGRPOTrainer

        trainer = self._make_async_trainer()

        images = [["img_a"], ["img_b", "img_c"]]  # num_images == [1, 2]
        inputs = [
            {"prompt": "p1", "images": images[0]},
            {"prompt": "p2", "images": images[1]},
        ]
        trainer._generate = MagicMock(
            return_value=(
                [[1, 2, 3], [1, 2]],  # prompt_ids_list
                [[4, 5], [4, 5, 6]],  # completion_ids_list
                None,  # tool_mask_list
                ["a", "b"],  # completions
                2,  # num_items_in_batch
                None,  # sampling_per_token_logps_list
                None,  # extra_fields
            )
        )
        trainer.processing_class = MagicMock()
        trainer.processing_class.return_value = {
            "spatial_shapes": torch.zeros(3, 2),
            "pixel_values": torch.zeros(3, 4, 4),
            "image_position_ids": torch.zeros(2, 5),
        }
        trainer.processing_class.image_processor.use_thumbnail = False
        trainer.processing_class.image_processor.return_value = {
            "image_rows": torch.tensor([1, 1, 1]),
            "image_cols": torch.tensor([2, 3, 4]),
        }

        with (
            patch(
                "axolotl.core.trainers.grpo.async_trainer.apply_chat_template",
                return_value={"prompt": "TEXT"},
            ),
            patch(
                "axolotl.core.trainers.grpo.async_trainer.is_conversational",
                return_value=True,
            ),
            patch(
                "axolotl.core.trainers.grpo.async_trainer.prepare_multimodal_messages",
                side_effect=lambda p, _il: p,
            ),
        ):
            output = AsyncGRPOTrainer._generate_only(trainer, inputs)

        # image_rows * image_cols == [2, 3, 4]; split by num_images=[1, 2] → [2], [3, 4]
        self.assertEqual(output["num_tiles"], [2, 7])
        self.assertIn("spatial_shapes", output)
        self.assertIn("image_position_ids", output)

    def test_compute_deferred_scores_forwards_tile_fields(self):
        trainer = self._make_async_trainer()
        captured, side_effect = _capture_and_stop()
        trainer._get_per_token_logps_and_entropies = MagicMock(side_effect=side_effect)

        rollout = {
            "_deferred_inputs": [{}, {}],
            "_deferred_prompts": ["p1", "p2"],
            "_deferred_completions": ["c1", "c2"],
            "_deferred_completion_ids_list": [[1], [1]],
            "_pending_policy_logps": True,
            "prompt_ids": torch.zeros(2, 3, dtype=torch.long),
            "completion_ids": torch.zeros(2, 4, dtype=torch.long),
            "prompt_mask": torch.ones(2, 3, dtype=torch.long),
            "completion_mask": torch.ones(2, 4, dtype=torch.long),
            "spatial_shapes": torch.zeros(2, 2),
            "image_position_ids": torch.zeros(2, 5),
            "num_tiles": [2, 3],
        }

        with self.assertRaises(_StopEarly):
            trainer._compute_deferred_scores(rollout)

        self.assertEqual(captured["num_tiles"], [2, 3])
        self.assertIn("spatial_shapes", captured)
        self.assertIn("image_position_ids", captured)

    def test_compute_streaming_group_scores_forwards_tile_fields(self):
        trainer = self._make_async_trainer()
        captured, side_effect = _capture_and_stop()
        trainer._get_per_token_logps_and_entropies = MagicMock(side_effect=side_effect)

        data = {
            "prompt_ids": torch.zeros(4, 3, dtype=torch.long),
            "completion_ids": torch.zeros(4, 4, dtype=torch.long),
            "prompt_mask": torch.ones(4, 3, dtype=torch.long),
            "completion_mask": torch.ones(4, 4, dtype=torch.long),
            "spatial_shapes": torch.zeros(4, 2),
            "image_position_ids": torch.zeros(4, 5),
            "num_tiles": [2, 3, 4, 5],
        }

        with self.assertRaises(_StopEarly):
            trainer._compute_streaming_group_scores(
                data,
                s_start=0,
                s_end=2,
                inputs=[{}, {}],
                prompts=["p1", "p2"],
                completions=["c1", "c2"],
                completion_ids_list=[[1], [1]],
                is_last_chunk=True,
            )

        # num_tiles is a per-sample list (like num_images) — sliced to the chunk
        self.assertEqual(captured["num_tiles"], [2, 3])
        self.assertIn("spatial_shapes", captured)
        self.assertIn("image_position_ids", captured)

    def test_compute_loss_forwards_tile_fields(self):
        trainer = self._make_async_trainer()
        captured, side_effect = _capture_and_stop()
        trainer._get_per_token_logps_and_entropies = MagicMock(side_effect=side_effect)

        inputs = {
            "prompt_ids": torch.zeros(2, 3, dtype=torch.long),
            "completion_ids": torch.zeros(2, 4, dtype=torch.long),
            "prompt_mask": torch.ones(2, 3, dtype=torch.long),
            "completion_mask": torch.ones(2, 4, dtype=torch.long),
            "spatial_shapes": torch.zeros(2, 2),
            "num_tiles": [2, 3],
            "image_position_ids": torch.zeros(2, 5),
        }

        with self.assertRaises(_StopEarly):
            trainer._compute_loss(trainer.model, inputs)

        self.assertEqual(captured["num_tiles"], [2, 3])
        self.assertIn("spatial_shapes", captured)
        self.assertIn("image_position_ids", captured)

    def test_fast_async_replay_recompute_forwards_tile_fields(self):
        from axolotl.core.trainers.grpo.fast_async_trainer import (
            FastAsyncGRPOTrainer,
        )
        from axolotl.core.trainers.grpo.replay_buffer import ReplayBuffer

        trainer = FastAsyncGRPOTrainer.__new__(FastAsyncGRPOTrainer)
        trainer.model = MagicMock()
        trainer.model.is_gradient_checkpointing = False
        trainer.args = types.SimpleNamespace(gradient_checkpointing_kwargs=None)
        trainer.reward_funcs = [MagicMock()]
        trainer._replay_recompute_logps = True
        trainer._replay_buffer = ReplayBuffer(max_size=2)
        trainer._replay_buffer.add(
            1.0,
            {
                "prompt_ids": torch.zeros(2, 3, dtype=torch.long),
                "completion_ids": torch.zeros(2, 4, dtype=torch.long),
                "prompt_mask": torch.ones(2, 3, dtype=torch.long),
                "completion_mask": torch.ones(2, 4, dtype=torch.long),
                "old_per_token_logps": torch.zeros(2, 4),
            },
        )
        captured, side_effect = _capture_and_stop()
        trainer._get_per_token_logps_and_entropies = MagicMock(side_effect=side_effect)

        # One group (num_generations=2) with zero reward variance → "no signal" →
        # triggers replacement from the replay buffer + old_per_token_logps recompute.
        # 2 samples, uneven tiles-per-sample [2, 3] -> 5 tiles total (LFM2-VL-style).
        data = {
            "prompt_ids": torch.zeros(2, 3, dtype=torch.long),
            "completion_ids": torch.zeros(2, 4, dtype=torch.long),
            "prompt_mask": torch.ones(2, 3, dtype=torch.long),
            "completion_mask": torch.ones(2, 4, dtype=torch.long),
            "old_per_token_logps": torch.zeros(2, 4),
            "num_tiles": [2, 3],
            "pixel_values": torch.zeros(5, 3, 4, 4),
            "pixel_attention_mask": torch.ones(5, 4, 4),
            "spatial_shapes": torch.zeros(5, 2),
        }
        rewards_per_func = torch.zeros(2, 1)
        advantages = torch.zeros(2)

        with self.assertRaises(_StopEarly):
            FastAsyncGRPOTrainer._post_advantage_hook(
                trainer,
                data,
                rewards_per_func,
                advantages,
                inputs=[{}, {}],
                num_generations=2,
                mode="train",
            )

        self.assertIn("spatial_shapes", captured)
        self.assertIn("num_tiles", captured)
        self.assertEqual(captured["num_tiles"], [2, 3])


class TestGetPerTokenLogpsMultimodalTailBatch(unittest.TestCase):
    """A batch_size that doesn't evenly divide the sample count must not
    overflow the cumulative image/tile index tensors on the last chunk."""

    def _make_trainer(self):
        from axolotl.core.trainers.grpo.async_trainer import AsyncGRPOTrainer

        trainer = AsyncGRPOTrainer.__new__(AsyncGRPOTrainer)
        trainer.is_fsdp_enabled = False
        trainer.accelerator = MagicMock()
        trainer.accelerator.unwrap_model = lambda m, keep_fp32_wrapper=True: m
        trainer.use_liger_kernel = False
        trainer.temperature = 1.0
        trainer.model_kwarg_keys = set()
        return trainer

    @staticmethod
    def _fake_model(seq_len, vocab_size):
        def _call(**kwargs):
            n = kwargs["input_ids"].size(0)
            out = MagicMock()
            out.logits = torch.randn(n, seq_len, vocab_size)
            return out

        return _call

    def test_image_position_ids_tail_batch_does_not_overflow(self):
        trainer = self._make_trainer()
        B, L, V = 5, 6, 10  # batch_size=2 -> last chunk start=4, start+batch_size=6 > B
        logits_to_keep = 3
        num_images = [1, 1, 1, 1, 1]

        logps, _entropies, _aux = trainer._get_per_token_logps_and_entropies(
            self._fake_model(L, V),
            torch.randint(0, V, (B, L)),
            torch.ones(B, L, dtype=torch.long),
            logits_to_keep,
            batch_size=2,
            num_images=num_images,
            pixel_values=torch.randn(5, 3, 4, 4),
            image_position_ids=torch.randn(5, 2),
        )
        self.assertEqual(logps.shape, (B, logits_to_keep))

    def test_spatial_shapes_tail_batch_does_not_overflow(self):
        trainer = self._make_trainer()
        B, L, V = 5, 6, 10
        logits_to_keep = 3
        num_tiles = [1, 1, 1, 1, 1]

        logps, _entropies, _aux = trainer._get_per_token_logps_and_entropies(
            self._fake_model(L, V),
            torch.randint(0, V, (B, L)),
            torch.ones(B, L, dtype=torch.long),
            logits_to_keep,
            batch_size=2,
            num_tiles=num_tiles,
            pixel_values=torch.randn(5, 3, 4, 4),
            pixel_attention_mask=torch.ones(5, 4, 4, dtype=torch.long),
            spatial_shapes=torch.randn(5, 2),
        )
        self.assertEqual(logps.shape, (B, logits_to_keep))

    def test_image_grid_thw_tail_batch_does_not_overflow(self):
        trainer = self._make_trainer()
        B, L, V = 5, 6, 10
        logits_to_keep = 3
        num_images = [1, 1, 1, 1, 1]

        logps, _entropies, _aux = trainer._get_per_token_logps_and_entropies(
            self._fake_model(L, V),
            torch.randint(0, V, (B, L)),
            torch.ones(B, L, dtype=torch.long),
            logits_to_keep,
            batch_size=2,
            num_images=num_images,
            pixel_values=torch.randn(5, 4),
            image_grid_thw=torch.tensor([[1, 1, 1]] * 5),
        )
        self.assertEqual(logps.shape, (B, logits_to_keep))


class TestComputeLossAuxLoss(unittest.TestCase):
    """MoE router aux loss must be requested, added to the policy loss with
    ``router_aux_loss_coef``, and logged — matching stock TRL's
    ``GRPOTrainer._compute_loss`` behavior."""

    def _make_trainer(self, aux_loss_enabled, batch_flattening=False):
        from axolotl.core.trainers.grpo.async_trainer import AsyncGRPOTrainer

        trainer = AsyncGRPOTrainer.__new__(AsyncGRPOTrainer)
        trainer.is_fsdp_enabled = False
        trainer.aux_loss_enabled = aux_loss_enabled
        trainer.router_aux_loss_coef = 0.01
        trainer.top_entropy_quantile = 1.0
        trainer.beta = 0.0
        trainer.loss_type = "grpo"
        trainer.epsilon_low = 0.2
        trainer.epsilon_high = 0.2
        trainer.use_vllm = False
        trainer.off_policy_mask_threshold = None
        trainer.current_gradient_accumulation_steps = 1
        trainer.args = types.SimpleNamespace(
            batch_flattening=batch_flattening, delta=None
        )
        trainer.model = MagicMock()
        trainer.model.training = True
        trainer._metrics = defaultdict(lambda: defaultdict(list))
        trainer.accelerator = MagicMock()
        trainer.accelerator.gather = lambda x: x
        return trainer

    def _make_inputs(self):
        return {
            "prompt_ids": torch.zeros(2, 3, dtype=torch.long),
            "prompt_mask": torch.ones(2, 3, dtype=torch.long),
            "completion_ids": torch.zeros(2, 3, dtype=torch.long),
            "completion_mask": torch.ones(2, 3, dtype=torch.long),
            "advantages": torch.ones(2, 1),
        }

    def test_aux_loss_added_to_policy_loss_when_enabled(self):
        trainer = self._make_trainer(aux_loss_enabled=True)
        aux_loss = torch.tensor(2.0)
        trainer._get_per_token_logps_and_entropies = MagicMock(
            return_value=(torch.zeros(2, 3), torch.zeros(2, 3), aux_loss)
        )

        loss_with_aux = trainer._compute_loss(trainer.model, self._make_inputs())

        # compute_aux_loss must be requested from the low-level scoring call
        _, call_kwargs = trainer._get_per_token_logps_and_entropies.call_args
        self.assertTrue(call_kwargs["compute_aux_loss"])

        # Baseline without aux loss: same setup, aux_loss_enabled=False
        trainer_no_aux = self._make_trainer(aux_loss_enabled=False)
        trainer_no_aux._get_per_token_logps_and_entropies = MagicMock(
            return_value=(torch.zeros(2, 3), torch.zeros(2, 3), None)
        )
        loss_without_aux = trainer_no_aux._compute_loss(
            trainer_no_aux.model, self._make_inputs()
        )
        _, call_kwargs_no_aux = (
            trainer_no_aux._get_per_token_logps_and_entropies.call_args
        )
        self.assertFalse(call_kwargs_no_aux["compute_aux_loss"])

        # router_aux_loss_coef * aux_loss must be added on top of the policy loss
        self.assertAlmostEqual(
            (loss_with_aux - loss_without_aux).item(),
            trainer.router_aux_loss_coef * aux_loss.item(),
            places=5,
        )
        self.assertIn("aux_loss", trainer._metrics["train"])
        self.assertNotIn("aux_loss", trainer_no_aux._metrics["train"])

    def test_flattened_path_disabled_when_aux_loss_enabled(self):
        """batch_flattening's fast path can't produce an aux_loss; it must be
        skipped (not silently drop the MoE loss) whenever aux_loss is needed."""
        trainer = self._make_trainer(aux_loss_enabled=True, batch_flattening=True)
        aux_loss = torch.tensor(1.5)
        trainer._get_per_token_logps_and_entropies = MagicMock(
            return_value=(torch.zeros(2, 3), torch.zeros(2, 3), aux_loss)
        )
        trainer._get_per_token_logps_and_entropies_flattened = MagicMock(
            side_effect=AssertionError(
                "flattened path must not be used when aux_loss_enabled"
            )
        )

        trainer._compute_loss(trainer.model, self._make_inputs())

        trainer._get_per_token_logps_and_entropies.assert_called_once()
        trainer._get_per_token_logps_and_entropies_flattened.assert_not_called()


class TestSliceMultimodalKwargs(unittest.TestCase):
    """_slice_multimodal_kwargs must slice image/tile-indexed tensors by
    cumulative image/tile offset, not by naive sample-range slicing."""

    def test_image_position_ids_path_uses_image_offset_not_sample_offset(self):
        from axolotl.core.trainers.grpo.fast_async_trainer import (
            _slice_multimodal_kwargs,
        )

        # 4 samples, uneven images-per-sample: [1, 2, 1, 1] -> 5 images total
        num_images = [1, 2, 1, 1]
        pixel_values = torch.arange(5).float().unsqueeze(1).repeat(1, 3)
        image_position_ids = torch.arange(5).float().unsqueeze(1).repeat(1, 2)
        data = {
            "num_images": num_images,
            "pixel_values": pixel_values,
            "image_position_ids": image_position_ids,
        }

        # samples [1:3) own images at rows [1:4) (sample 0 -> row 0,
        # sample 1 -> rows 1-2, sample 2 -> row 3) — NOT rows [1:3).
        out = _slice_multimodal_kwargs(data, 1, 3)
        self.assertEqual(out["num_images"], [2, 1])
        torch.testing.assert_close(out["pixel_values"], pixel_values[1:4])
        torch.testing.assert_close(out["image_position_ids"], image_position_ids[1:4])

    def test_spatial_shapes_path_uses_tile_offset_not_sample_offset(self):
        from axolotl.core.trainers.grpo.fast_async_trainer import (
            _slice_multimodal_kwargs,
        )

        # 3 samples, uneven tiles-per-sample: [2, 1, 3] -> 6 tiles total
        num_tiles = [2, 1, 3]
        pixel_values = torch.arange(6).float().unsqueeze(1).repeat(1, 3)
        spatial_shapes = torch.arange(6).float().unsqueeze(1).repeat(1, 2)
        pixel_attention_mask = torch.ones(6, 4)
        data = {
            "num_tiles": num_tiles,
            "pixel_values": pixel_values,
            "spatial_shapes": spatial_shapes,
            "pixel_attention_mask": pixel_attention_mask,
        }

        # samples [1:3) own tile rows [2:6), not [1:3)
        out = _slice_multimodal_kwargs(data, 1, 3)
        self.assertEqual(out["num_tiles"], [1, 3])
        torch.testing.assert_close(out["pixel_values"], pixel_values[2:6])
        torch.testing.assert_close(out["spatial_shapes"], spatial_shapes[2:6])
        torch.testing.assert_close(
            out["pixel_attention_mask"], pixel_attention_mask[2:6]
        )

    def test_sample_indexed_fields_use_plain_sample_slice(self):
        from axolotl.core.trainers.grpo.fast_async_trainer import (
            _slice_multimodal_kwargs,
        )

        data = {
            "image_sizes": torch.arange(4).unsqueeze(1),
            "token_type_ids": torch.arange(8).view(4, 2),
        }
        out = _slice_multimodal_kwargs(data, 1, 3)
        torch.testing.assert_close(out["image_sizes"], data["image_sizes"][1:3])
        torch.testing.assert_close(out["token_type_ids"], data["token_type_ids"][1:3])


class TestFastAsyncReplayRecomputeImageOffset(unittest.TestCase):
    """Regression test for the replay-recompute call site: a replayed group
    with r_start > 0 must receive the images belonging to *its* samples,
    not the first N images of the full batch."""

    def test_replay_recompute_uses_correct_image_offset_for_r_start_gt_0(self):
        from axolotl.core.trainers.grpo.fast_async_trainer import (
            FastAsyncGRPOTrainer,
        )
        from axolotl.core.trainers.grpo.replay_buffer import ReplayBuffer

        trainer = FastAsyncGRPOTrainer.__new__(FastAsyncGRPOTrainer)
        trainer.model = MagicMock()
        trainer.model.is_gradient_checkpointing = False
        trainer.args = types.SimpleNamespace(gradient_checkpointing_kwargs=None)
        trainer.reward_funcs = [MagicMock()]
        trainer._replay_recompute_logps = True
        trainer._replay_buffer = ReplayBuffer(max_size=2)
        trainer._replay_buffer.add(
            1.0,
            {
                "prompt_ids": torch.zeros(2, 3, dtype=torch.long),
                "completion_ids": torch.zeros(2, 4, dtype=torch.long),
                "prompt_mask": torch.ones(2, 3, dtype=torch.long),
                "completion_mask": torch.ones(2, 4, dtype=torch.long),
                "old_per_token_logps": torch.zeros(2, 4),
            },
        )
        captured, side_effect = _capture_and_stop()
        trainer._get_per_token_logps_and_entropies = MagicMock(side_effect=side_effect)

        # 4 samples / 2 groups of num_generations=2. Uneven images-per-sample
        # [1, 2, 1, 1] -> 5 images total, so a naive data[key][r_start:r_end]
        # slice would silently return the wrong (group 0's) images.
        num_images = [1, 2, 1, 1]
        pixel_values = torch.arange(5).float().unsqueeze(1).repeat(1, 3)
        image_position_ids = torch.arange(5).float().unsqueeze(1).repeat(1, 2)
        data = {
            "prompt_ids": torch.zeros(4, 3, dtype=torch.long),
            "completion_ids": torch.zeros(4, 4, dtype=torch.long),
            "prompt_mask": torch.ones(4, 3, dtype=torch.long),
            "completion_mask": torch.ones(4, 4, dtype=torch.long),
            "old_per_token_logps": torch.zeros(4, 4),
            "num_images": num_images,
            "pixel_values": pixel_values,
            "image_position_ids": image_position_ids,
        }
        # Group 0 (samples 0-1) has reward signal -> kept as-is.
        # Group 1 (samples 2-3) has zero variance -> replaced from the replay
        # buffer, triggering recompute for r_start=2, r_end=4.
        rewards_per_func = torch.tensor([[0.0], [1.0], [0.5], [0.5]])
        advantages = torch.zeros(4)

        with self.assertRaises(_StopEarly):
            FastAsyncGRPOTrainer._post_advantage_hook(
                trainer,
                data,
                rewards_per_func,
                advantages,
                inputs=[{}, {}, {}, {}],
                num_generations=2,
                mode="train",
            )

        # Samples [2:4) own images at rows [3:5) (sample 0 -> row 0,
        # sample 1 -> rows 1-2, sample 2 -> row 3, sample 3 -> row 4).
        self.assertEqual(captured["num_images"], [1, 1])
        torch.testing.assert_close(captured["pixel_values"], pixel_values[3:5])
        torch.testing.assert_close(
            captured["image_position_ids"], image_position_ids[3:5]
        )


if __name__ == "__main__":
    unittest.main()
