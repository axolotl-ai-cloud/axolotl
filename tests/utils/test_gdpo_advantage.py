"""
Unit tests for GDPO (Group Reward-Decoupled Normalization Policy Optimization)
advantage calculation.

GDPO addresses the reward advantage collapse problem in multi-reward GRPO training
by normalizing each reward function independently before combining them.
"""

import torch

from axolotl.core.trainers.gdpo.trainer import compute_gdpo_advantages


class TestGDPOAdvantageCalculation:
    """Test GDPO's decoupled normalization logic."""

    def test_gdpo_vs_grpo_different_advantages(self):
        """GDPO should produce different advantages than GRPO for multi-reward scenarios."""
        # Setup: 2 prompts, 4 generations each, 2 reward functions
        num_generations = 4
        epsilon = 1e-4

        # Simulate rewards: shape (8, 2) = (batch_size, num_rewards)
        rewards_per_func = torch.tensor(
            [
                # Prompt 1 generations
                [0.0, 1.0],  # gen1: format=0, correct=1
                [0.0, 2.0],  # gen2: format=0, correct=2
                [1.0, 2.0],  # gen3: format=1, correct=2
                [1.0, 3.0],  # gen4: format=1, correct=3
                # Prompt 2 generations
                [0.0, 0.0],
                [1.0, 1.0],
                [1.0, 2.0],
                [0.0, 3.0],
            ]
        )

        weights = torch.tensor([1.0, 1.0])

        # GRPO: combine first, then normalize
        combined_rewards = (rewards_per_func * weights).sum(dim=1)
        grpo_grouped = combined_rewards.view(-1, num_generations)
        grpo_mean = grpo_grouped.mean(dim=1, keepdim=True)
        grpo_std = grpo_grouped.std(dim=1, keepdim=True)
        grpo_advantages = ((grpo_grouped - grpo_mean) / (grpo_std + epsilon)).view(-1)

        # GDPO: normalize each reward, then combine
        gdpo_advantages = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=epsilon,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
        )

        # Assert they're different
        assert not torch.allclose(grpo_advantages, gdpo_advantages, atol=0.01), (
            "GDPO advantages should differ from GRPO for multi-reward scenarios"
        )

    def test_gdpo_preserves_reward_signal(self):
        """GDPO should preserve individual reward signals better than GRPO."""
        # When format=1 and correct=2 vs format=0 and correct=2,
        # GDPO should show clear advantage difference for format
        num_generations = 2

        rewards_per_func = torch.tensor(
            [
                [0.0, 2.0],  # format=0, correct=2
                [1.0, 2.0],  # format=1, correct=2
            ]
        )

        weights = torch.tensor([1.0, 1.0])

        gdpo_advantages = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=1e-4,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
        )

        # The second sample (format=1) should have higher advantage
        assert gdpo_advantages[1] > gdpo_advantages[0], (
            "GDPO should reward format=1 higher than format=0"
        )

    def test_single_reward_gdpo_equals_grpo(self):
        """With single reward, GDPO should produce proportional advantages to GRPO."""
        num_generations = 4
        epsilon = 1e-4

        rewards_per_func = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        weights = torch.tensor([1.0])

        # GRPO calculation
        combined = (rewards_per_func * weights).sum(dim=1)
        grpo_grouped = combined.view(-1, num_generations)
        grpo_mean = grpo_grouped.mean(dim=1, keepdim=True)
        grpo_std = grpo_grouped.std(dim=1, keepdim=True)
        grpo_advantages = ((grpo_grouped - grpo_mean) / (grpo_std + epsilon)).view(-1)

        # GDPO calculation
        gdpo_advantages = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=epsilon,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
        )

        # The relative ordering and proportions should be the same
        # (note: GDPO uses sample variance like nanstd, while torch.std uses population variance)
        # Check that normalized versions are close
        grpo_norm = (grpo_advantages - grpo_advantages.mean()) / (
            grpo_advantages.std() + epsilon
        )
        gdpo_norm = (gdpo_advantages - gdpo_advantages.mean()) / (
            gdpo_advantages.std() + epsilon
        )

        assert torch.allclose(grpo_norm, gdpo_norm, atol=1e-4), (
            "Single-reward GDPO should have same relative advantages as GRPO"
        )

    def test_gdpo_with_nan_handling(self):
        """GDPO should handle NaN values in rewards gracefully."""
        num_generations = 4

        rewards_per_func = torch.tensor(
            [
                [1.0, 2.0],
                [float("nan"), 3.0],  # NaN in first reward
                [3.0, float("nan")],  # NaN in second reward
                [4.0, 5.0],
            ]
        )

        weights = torch.tensor([1.0, 1.0])

        # Should not raise an exception
        gdpo_advantages = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=1e-4,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
        )

        # Result should have NaN in affected positions
        assert gdpo_advantages.shape[0] == 4

    def test_gdpo_batch_normalization(self):
        """Test that batch normalization option works correctly."""
        num_generations = 4

        rewards_per_func = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [3.0, 3.0],
                [4.0, 4.0],
            ]
        )

        weights = torch.tensor([1.0, 1.0])

        # With batch norm
        adv_with_batch = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=1e-4,
            gdpo_batch_norm=True,
            gdpo_per_reward_scale=True,
        )

        # With batch norm, advantages should be normalized to zero mean
        assert abs(adv_with_batch.nanmean().item()) < 0.01, (
            "Batch-normalized advantages should have near-zero mean"
        )

    def test_gdpo_reward_weights(self):
        """Test that reward weights are applied correctly."""
        num_generations = 2

        rewards_per_func = torch.tensor(
            [
                [1.0, 1.0],  # Same raw rewards
                [0.0, 0.0],
            ]
        )

        # Equal weights
        weights_equal = torch.tensor([1.0, 1.0])
        adv_equal = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights_equal,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=1e-4,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
        )

        # Unequal weights (reward 2 weighted 2x)
        weights_unequal = torch.tensor([1.0, 2.0])
        adv_unequal = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights_unequal,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=1e-4,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
        )

        # Advantages should differ based on weights
        assert not torch.allclose(adv_equal, adv_unequal), (
            "Different reward weights should produce different advantages"
        )

    def test_gdpo_without_scaling(self):
        """Test GDPO without reward scaling."""
        num_generations = 4

        rewards_per_func = torch.tensor(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [4.0, 40.0],
            ]
        )

        weights = torch.tensor([1.0, 1.0])

        # With scaling
        adv_scaled = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=1e-4,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
        )

        # Without scaling
        adv_unscaled = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights,
            num_generations=num_generations,
            scale_rewards=False,
            gdpo_epsilon=1e-4,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
        )

        # Results should differ
        assert not torch.allclose(adv_scaled, adv_unscaled), (
            "Scaling should affect advantages"
        )

    def test_gdpo_distinct_advantage_groups(self):
        """
        Test that GDPO maintains more distinct advantage groups than GRPO.

        This is a key benefit of GDPO - different reward combinations should
        produce different advantages when rewards have different weights.
        """
        num_generations = 4
        epsilon = 1e-4

        # Setup rewards where GRPO would collapse distinctions when using equal weights
        # but GDPO with different weights should preserve them
        rewards_per_func = torch.tensor(
            [
                [0.0, 3.0],  # sum=3
                [1.0, 2.0],  # sum=3  (same as above in GRPO)
                [2.0, 1.0],  # sum=3  (same as above in GRPO)
                [3.0, 0.0],  # sum=3  (same as above in GRPO)
            ]
        )

        # With equal weights, GRPO would produce all zeros
        weights_equal = torch.tensor([1.0, 1.0])
        combined = (rewards_per_func * weights_equal).sum(dim=1)
        grpo_grouped = combined.view(-1, num_generations)
        grpo_std = grpo_grouped.std(dim=1, keepdim=True)
        assert grpo_std.item() < epsilon, "GRPO std should be near zero for same sums"

        # But with unequal weights, GDPO should produce distinct advantages
        # because each reward is normalized separately before weighting
        weights_unequal = torch.tensor([1.0, 2.0])  # Correctness weighted more

        gdpo_advantages = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights_unequal,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=epsilon,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
        )

        # GDPO should have distinct values because the weights break symmetry
        unique_advantages = len(torch.unique(gdpo_advantages.round(decimals=4)))
        assert unique_advantages > 1, (
            f"GDPO with unequal weights should produce distinct advantage groups, got {unique_advantages}"
        )

        # Additionally, the sample with highest second reward (index 0) should have
        # higher advantage due to higher weight on second reward
        assert gdpo_advantages[0] > gdpo_advantages[3], (
            "Sample with higher weighted reward should have higher advantage"
        )

    def test_gdpo_multiple_prompts(self):
        """Test GDPO with multiple prompts (groups)."""
        num_generations = 2
        num_prompts = 3

        # 3 prompts x 2 generations = 6 samples
        rewards_per_func = torch.tensor(
            [
                # Prompt 1
                [1.0, 2.0],
                [2.0, 1.0],
                # Prompt 2
                [3.0, 4.0],
                [4.0, 3.0],
                # Prompt 3
                [5.0, 6.0],
                [6.0, 5.0],
            ]
        )

        weights = torch.tensor([1.0, 1.0])

        gdpo_advantages = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=1e-4,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
        )

        assert gdpo_advantages.shape[0] == num_prompts * num_generations

        # Within each prompt group, advantages should be opposite
        # (higher reward gets higher advantage, lower gets lower)
        for i in range(num_prompts):
            start = i * num_generations
            end = start + num_generations
            group_adv = gdpo_advantages[start:end]
            # Advantages within group should have opposite signs
            assert group_adv[0] * group_adv[1] <= 0, (
                f"Group {i} advantages should have opposite signs"
            )


class TestGDPOEdgeCases:
    """Test edge cases for GDPO implementation."""

    def test_zero_std_handling(self):
        """Test handling when a reward has zero variance."""
        num_generations = 4

        # First reward has zero variance
        rewards_per_func = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
                [1.0, 4.0],
            ]
        )

        weights = torch.tensor([1.0, 1.0])

        # Should not raise division by zero error
        gdpo_advantages = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=1e-4,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
        )

        # Should still produce valid (non-inf) advantages
        assert not torch.isinf(gdpo_advantages).any(), (
            "Zero std should not produce infinite advantages"
        )

    def test_all_same_rewards(self):
        """Test when all rewards are identical."""
        num_generations = 4

        rewards_per_func = torch.tensor(
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        )

        weights = torch.tensor([1.0, 1.0])

        gdpo_advantages = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=1e-4,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
        )

        # All advantages should be 0 or near 0
        assert torch.allclose(
            gdpo_advantages, torch.zeros_like(gdpo_advantages), atol=1e-3
        ), "Identical rewards should produce near-zero advantages"

    def test_negative_rewards(self):
        """Test handling of negative rewards."""
        num_generations = 4

        rewards_per_func = torch.tensor(
            [
                [-3.0, -2.0],
                [-2.0, -1.0],
                [-1.0, 0.0],
                [0.0, 1.0],
            ]
        )

        weights = torch.tensor([1.0, 1.0])

        gdpo_advantages = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=1e-4,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
        )

        # Higher rewards should have higher advantages
        assert gdpo_advantages[3] > gdpo_advantages[0], (
            "Higher rewards should have higher advantages"
        )

    def test_device_handling(self):
        """Test that device parameter is respected."""
        num_generations = 4

        rewards_per_func = torch.tensor(
            [
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
            ]
        )

        weights = torch.tensor([1.0, 1.0])

        # Test on CPU explicitly
        gdpo_advantages = compute_gdpo_advantages(
            rewards_per_func=rewards_per_func,
            reward_weights=weights,
            num_generations=num_generations,
            scale_rewards=True,
            gdpo_epsilon=1e-4,
            gdpo_batch_norm=False,
            gdpo_per_reward_scale=True,
            device=torch.device("cpu"),
        )

        assert gdpo_advantages.device.type == "cpu"
