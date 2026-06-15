"""
Tests for the GRPO advantage estimators (group_mean, rloo, reinforce_plus_plus)
"""

import pytest
import torch

from axolotl.core.trainers.grpo.advantages import (
    ADVANTAGE_ESTIMATORS,
    compute_advantages,
)
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.validation import RLValidationMixin

# 2 groups of 4 generations with distinct group statistics
REWARDS = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 10.0, 14.0, 18.0])
NUM_GENERATIONS = 4


class TestComputeAdvantages:
    """Unit tests for compute_advantages."""

    def test_group_mean_matches_grpo_formula(self):
        """group_mean reproduces TRL's GRPO advantage computation exactly."""
        advantages, _, _ = compute_advantages(
            REWARDS, NUM_GENERATIONS, "group_mean", scale_rewards="group"
        )
        grouped = REWARDS.view(-1, NUM_GENERATIONS)
        mean = grouped.mean(dim=1).repeat_interleave(NUM_GENERATIONS)
        std = grouped.std(dim=1).repeat_interleave(NUM_GENERATIONS)
        expected = (REWARDS - mean) / (std + 1e-4)
        torch.testing.assert_close(advantages, expected)

    def test_group_mean_scale_none(self):
        """scale_rewards='none' subtracts the group mean without dividing by std."""
        advantages, _, _ = compute_advantages(
            REWARDS, NUM_GENERATIONS, "group_mean", scale_rewards="none"
        )
        grouped = REWARDS.view(-1, NUM_GENERATIONS)
        mean = grouped.mean(dim=1).repeat_interleave(NUM_GENERATIONS)
        torch.testing.assert_close(advantages, REWARDS - mean)

    def test_group_mean_scale_batch(self):
        """scale_rewards='batch' divides by the std of the whole batch."""
        advantages, _, _ = compute_advantages(
            REWARDS, NUM_GENERATIONS, "group_mean", scale_rewards="batch"
        )
        grouped = REWARDS.view(-1, NUM_GENERATIONS)
        mean = grouped.mean(dim=1).repeat_interleave(NUM_GENERATIONS)
        expected = (REWARDS - mean) / (REWARDS.std() + 1e-4)
        torch.testing.assert_close(advantages, expected)

    def test_rloo_leave_one_out_baseline(self):
        """RLOO baselines each reward against the mean of the other G-1 rewards."""
        advantages, _, _ = compute_advantages(
            REWARDS, NUM_GENERATIONS, "rloo", scale_rewards="none"
        )
        grouped = REWARDS.view(-1, NUM_GENERATIONS)
        baseline = (grouped.sum(dim=1, keepdim=True) - grouped) / (NUM_GENERATIONS - 1)
        torch.testing.assert_close(advantages, (grouped - baseline).reshape(-1))

    def test_rloo_is_scaled_group_mean(self):
        """Unscaled RLOO advantages equal group_mean advantages times G/(G-1)."""
        rloo, _, _ = compute_advantages(
            REWARDS, NUM_GENERATIONS, "rloo", scale_rewards="none"
        )
        group_mean, _, _ = compute_advantages(
            REWARDS, NUM_GENERATIONS, "group_mean", scale_rewards="none"
        )
        torch.testing.assert_close(
            rloo, group_mean * NUM_GENERATIONS / (NUM_GENERATIONS - 1)
        )

    def test_reinforce_plus_plus_global_baseline(self):
        """REINFORCE++ with batch scaling is the global z-score of the rewards."""
        advantages, _, _ = compute_advantages(
            REWARDS, NUM_GENERATIONS, "reinforce_plus_plus", scale_rewards="batch"
        )
        expected = (REWARDS - REWARDS.mean()) / (REWARDS.std() + 1e-4)
        torch.testing.assert_close(advantages, expected)

    def test_bool_scale_rewards_accepted(self):
        """Bool scale_rewards values map to 'group' (True) and 'none' (False)."""
        scaled, _, _ = compute_advantages(REWARDS, NUM_GENERATIONS, scale_rewards=True)
        grouped, _, _ = compute_advantages(
            REWARDS, NUM_GENERATIONS, scale_rewards="group"
        )
        torch.testing.assert_close(scaled, grouped)
        unscaled, _, _ = compute_advantages(
            REWARDS, NUM_GENERATIONS, scale_rewards=False
        )
        none, _, _ = compute_advantages(REWARDS, NUM_GENERATIONS, scale_rewards="none")
        torch.testing.assert_close(unscaled, none)

    def test_zero_variance_group_flagged(self):
        """Constant-reward groups are reported via is_std_zero."""
        rewards = torch.tensor([1.0, 1.0, 2.0, 4.0])
        _, _, is_std_zero = compute_advantages(rewards, 2, scale_rewards="group")
        assert is_std_zero.tolist() == [True, True, False, False]

    def test_invalid_estimator_raises(self):
        """Unknown estimators raise a clear ValueError."""
        with pytest.raises(ValueError, match="Invalid advantage_estimator"):
            compute_advantages(REWARDS, NUM_GENERATIONS, "ppo")

    def test_invalid_scale_rewards_raises(self):
        """Unknown scale_rewards values raise a clear ValueError."""
        with pytest.raises(ValueError, match="Invalid scale_rewards"):
            compute_advantages(REWARDS, NUM_GENERATIONS, scale_rewards="global")

    def test_rloo_requires_two_generations(self):
        """RLOO needs at least two generations for a leave-one-out baseline."""
        with pytest.raises(ValueError, match="num_generations >= 2"):
            compute_advantages(torch.tensor([1.0, 2.0]), 1, "rloo")

    def test_all_estimators_zero_mean_per_batch(self):
        """Every estimator produces advantages that sum to ~0 over the batch."""
        for estimator in ADVANTAGE_ESTIMATORS:
            advantages, _, _ = compute_advantages(
                REWARDS, NUM_GENERATIONS, estimator, scale_rewards="none"
            )
            assert abs(advantages.sum().item()) < 1e-4, estimator


class TestTrainerOverride:
    """The AxolotlGRPOTrainer hook swaps advantages for the configured estimator."""

    def test_override_replaces_advantages_and_logs(self, monkeypatch):
        """With advantage_estimator=rloo, the batch advantages and the logged
        advantages both reflect the leave-one-out baseline instead of TRL's
        group-mean values."""
        from collections import deque
        from unittest.mock import MagicMock

        from trl import GRPOTrainer

        from axolotl.core.trainers.grpo.trainer import AxolotlGRPOTrainer

        rewards_per_func = REWARDS.unsqueeze(1)

        def fake_super_gen(self, inputs):
            """Simulate TRL's parent method: stash rewards and return/log
            group-mean advantages."""
            self._axolotl_rewards_per_func = rewards_per_func
            adv, _, _ = compute_advantages(
                REWARDS, NUM_GENERATIONS, "group_mean", "none"
            )
            self._logs["advantages"].extend(adv.tolist())
            return {"advantages": adv}

        monkeypatch.setattr(
            GRPOTrainer, "_generate_and_score_completions", fake_super_gen
        )

        trainer = object.__new__(AxolotlGRPOTrainer)
        trainer.args = MagicMock(advantage_estimator="rloo")
        trainer.multi_objective_aggregation = "sum_then_normalize"
        trainer.reward_weights = torch.ones(1)
        trainer.scale_rewards = "none"
        trainer.num_generations = NUM_GENERATIONS
        trainer.num_generations_eval = NUM_GENERATIONS
        trainer.model = MagicMock(training=True)
        trainer.accelerator = MagicMock(process_index=0)
        trainer._logs = {"advantages": deque(maxlen=64)}

        inputs = [{} for _ in range(len(REWARDS))]
        output = AxolotlGRPOTrainer._generate_and_score_completions(trainer, inputs)

        expected, _, _ = compute_advantages(REWARDS, NUM_GENERATIONS, "rloo", "none")
        torch.testing.assert_close(output["advantages"], expected)
        assert list(trainer._logs["advantages"]) == expected.tolist()

    def test_override_noop_for_group_mean(self, monkeypatch):
        """With the default estimator, the parent output passes through untouched."""
        from unittest.mock import MagicMock

        from trl import GRPOTrainer

        from axolotl.core.trainers.grpo.trainer import AxolotlGRPOTrainer

        sentinel = {"advantages": torch.tensor([1.0])}
        monkeypatch.setattr(
            GRPOTrainer,
            "_generate_and_score_completions",
            lambda self, inputs: sentinel,
        )

        trainer = object.__new__(AxolotlGRPOTrainer)
        trainer.args = MagicMock(advantage_estimator="group_mean")
        assert (
            AxolotlGRPOTrainer._generate_and_score_completions(trainer, []) is sentinel
        )


class TestAdvantageEstimatorConfig:
    """Config plumbing and validation for trl.advantage_estimator."""

    def test_strategy_passes_estimator_to_training_args(self):
        """GRPOStrategy forwards advantage_estimator into the trainer kwargs."""
        from axolotl.core.trainers.grpo import GRPOStrategy

        cfg = DictDefault(
            {
                "trl": {"advantage_estimator": "rloo"},
                "vllm": {},
                "context_parallel_size": 1,
            }
        )
        kwargs = GRPOStrategy.set_training_args_kwargs(cfg)
        assert kwargs["advantage_estimator"] == "rloo"

    def test_validator_rejects_non_grpo(self):
        """advantage_estimator with a non-GRPO trainer is a config error."""
        data = {"rl": "dpo", "trl": {"advantage_estimator": "rloo"}}
        with pytest.raises(ValueError, match="only supported with `rl: grpo`"):
            RLValidationMixin.check_grpo_advantage_estimator(data)

    def test_validator_rejects_normalize_then_sum(self):
        """advantage_estimator with GDPO-style aggregation is a config error."""
        data = {
            "rl": "grpo",
            "trl": {
                "advantage_estimator": "reinforce_plus_plus",
                "multi_objective_aggregation": "normalize_then_sum",
            },
        }
        with pytest.raises(ValueError, match="normalize_then_sum"):
            RLValidationMixin.check_grpo_advantage_estimator(data)

    def test_validator_allows_valid_config(self):
        """A valid GRPO + rloo config passes validation unchanged."""
        data = {
            "rl": "grpo",
            "trl": {"advantage_estimator": "rloo", "scale_rewards": "none"},
        }
        assert RLValidationMixin.check_grpo_advantage_estimator(data) is data

    def test_validator_ignores_unset(self):
        """An unset estimator passes for any trainer type."""
        data = {"rl": "dpo", "trl": {}}
        assert RLValidationMixin.check_grpo_advantage_estimator(data) is data

    def test_validator_rejects_explicit_group_mean_non_grpo(self):
        """Explicitly setting the estimator (even to the default) requires rl: grpo."""
        data = {"rl": "dpo", "trl": {"advantage_estimator": "group_mean"}}
        with pytest.raises(ValueError, match="only supported with `rl: grpo`"):
            RLValidationMixin.check_grpo_advantage_estimator(data)

    def test_validator_allows_group_mean_with_normalize_then_sum(self):
        """group_mean is a no-op, so it stays valid with GDPO-style aggregation."""
        data = {
            "rl": "grpo",
            "trl": {
                "advantage_estimator": "group_mean",
                "multi_objective_aggregation": "normalize_then_sum",
            },
        }
        assert RLValidationMixin.check_grpo_advantage_estimator(data) is data
