"""Unit tests for SCOPE-RL helpers"""

import unittest
from collections import defaultdict
from unittest.mock import MagicMock

import torch

from axolotl.core.trainers.grpo.scope import (
    scope_aux_indices,
    scope_temperature,
    scope_weights,
)


class TestScopeTemperature(unittest.TestCase):
    """Entropy feedback controller: T = clip(1 + H0 - H, t_min, t_max)."""

    def test_at_target_is_neutral(self):
        self.assertEqual(scope_temperature(0.5, 0.5, 0.8, 1.2), 1.0)

    def test_below_target_raises_temperature(self):
        self.assertAlmostEqual(scope_temperature(0.4, 0.5, 0.8, 1.2), 1.1)

    def test_above_target_lowers_temperature(self):
        self.assertAlmostEqual(scope_temperature(0.6, 0.5, 0.8, 1.2), 0.9)

    def test_clipped_at_bounds(self):
        self.assertEqual(scope_temperature(0.0, 0.5, 0.8, 1.2), 1.2)
        self.assertEqual(scope_temperature(5.0, 0.5, 0.8, 1.2), 0.8)


class TestScopeAuxIndices(unittest.TestCase):
    """Auxiliary sample selection."""

    def test_paper_default_ratio(self):
        # 512 prompts x 8 generations, alpha = 1/64 -> 8 groups = 64 rows
        idx = scope_aux_indices(512 * 8, 8, 1 / 64, seed=0)
        self.assertEqual(len(idx), 64)

    def test_whole_groups_only(self):
        idx = scope_aux_indices(64, 8, 1 / 4, seed=0)
        groups = {i // 8 for i in idx}
        self.assertEqual(len(idx), len(groups) * 8)
        self.assertEqual(sorted(idx), idx)

    def test_at_least_one_group(self):
        self.assertEqual(len(scope_aux_indices(64, 8, 1e-6, seed=0)), 8)

    def test_disabled_and_empty_cases(self):
        self.assertEqual(scope_aux_indices(64, 8, 0.0, seed=0), [])
        self.assertEqual(scope_aux_indices(4, 8, 0.5, seed=0), [])

    def test_deterministic_for_a_seed(self):
        self.assertEqual(
            scope_aux_indices(256, 8, 0.25, seed=3),
            scope_aux_indices(256, 8, 0.25, seed=3),
        )


class TestScopeWeights(unittest.TestCase):
    """Row weights reproduce mean(main) + alpha * mean(aux)."""

    def _weighted_mean(self, losses, mask, alpha):
        return (losses * scope_weights(mask, alpha)).mean()

    def test_matches_two_term_objective(self):
        losses = torch.arange(10, dtype=torch.float)
        mask = torch.tensor([0.0] * 8 + [1.0] * 2)
        expected = losses[:8].mean() + 0.25 * losses[8:].mean()
        self.assertAlmostEqual(
            self._weighted_mean(losses, mask, 0.25).item(), expected.item(), places=5
        )

    def test_invariant_to_micro_batch_split(self):
        losses = torch.arange(16, dtype=torch.float)
        mask = torch.tensor([0.0] * 12 + [1.0] * 4)
        weights = scope_weights(mask, 1 / 8)
        whole = (losses * weights).mean()
        halves = [(losses[s] * weights[s]).mean() for s in (slice(0, 8), slice(8, 16))]
        self.assertAlmostEqual(whole.item(), (sum(halves) / 2).item(), places=5)

    def test_non_positive_rows_stay_in_the_denominator(self):
        """Eq. 11 averages the aux term over every resampled row, not just the positives."""
        losses = torch.tensor([0.0] * 8 + [5.0, 0.0, 0.0, 0.0])
        mask = torch.tensor([0.0] * 8 + [1.0] * 4)
        weighted = (losses * scope_weights(mask, 0.25)).mean()
        self.assertAlmostEqual(weighted.item(), 0.25 * 5.0 / 4.0, places=5)

    def test_no_aux_rows_is_plain_mean(self):
        losses = torch.arange(4, dtype=torch.float)
        mask = torch.zeros(4)
        self.assertAlmostEqual(
            self._weighted_mean(losses, mask, 0.5).item(), losses.mean().item()
        )


class TestScopeGenerate(unittest.TestCase):
    """Temperature override around the auxiliary generation call."""

    def _trainer(self, entropy, side_effect):
        trainer = MagicMock()
        trainer._scope_entropy = entropy
        trainer.vllm_generation.temperature = 1.0
        trainer.args.scope_target_entropy = 0.5
        trainer.args.scope_temperature_min = 0.8
        trainer.args.scope_temperature_max = 1.2
        trainer._metrics = {"train": defaultdict(list)}
        trainer._generate.side_effect = side_effect
        return trainer

    def test_collapsed_entropy_samples_hotter_then_restores(self):
        from axolotl.core.trainers.grpo.async_trainer import AsyncGRPOTrainer

        seen = []
        trainer = self._trainer(
            0.1, lambda _: seen.append(trainer.vllm_generation.temperature)
        )
        AsyncGRPOTrainer._scope_generate(trainer, ["p"], rank0_only=False)

        self.assertEqual(seen, [1.2])
        self.assertEqual(trainer.vllm_generation.temperature, 1.0)
        self.assertEqual(trainer._metrics["train"]["scope/temperature"], [1.2])

    def test_temperature_is_relative_to_the_sampling_temperature(self):
        from axolotl.core.trainers.grpo.async_trainer import AsyncGRPOTrainer

        seen = []
        trainer = self._trainer(
            1.0, lambda _: seen.append(trainer.vllm_generation.temperature)
        )
        trainer.vllm_generation.temperature = 0.5
        AsyncGRPOTrainer._scope_generate(trainer, ["p"], rank0_only=False)

        self.assertAlmostEqual(seen[0], 0.5 * 0.8)
        self.assertEqual(trainer.vllm_generation.temperature, 0.5)

    def test_restores_temperature_on_failure(self):
        from axolotl.core.trainers.grpo.async_trainer import AsyncGRPOTrainer

        trainer = self._trainer(0.1, RuntimeError("vllm down"))
        with self.assertRaises(RuntimeError):
            AsyncGRPOTrainer._scope_generate(trainer, ["p"], rank0_only=False)
        self.assertEqual(trainer.vllm_generation.temperature, 1.0)


if __name__ == "__main__":
    unittest.main()
