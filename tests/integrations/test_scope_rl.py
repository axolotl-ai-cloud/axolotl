"""Unit tests for the SCOPE-RL plugin"""

import unittest
from collections import defaultdict
from unittest.mock import MagicMock

import pytest
import torch

from axolotl.integrations.scope_rl import ScopeRLPlugin
from axolotl.integrations.scope_rl.args import TRAINER_CLS, ScopeRLArgs
from axolotl.integrations.scope_rl.scope import (
    scope_aux_indices,
    scope_temperature,
    scope_weights,
)
from axolotl.utils.dict import DictDefault


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
        trainer._last_entropy = entropy
        trainer.vllm_generation.temperature = 1.0
        trainer.args.scope_target_entropy = 0.5
        trainer.args.scope_temperature_min = 0.8
        trainer.args.scope_temperature_max = 1.2
        trainer._metrics = {"train": defaultdict(list)}
        trainer._generate.side_effect = side_effect
        return trainer

    def test_collapsed_entropy_samples_hotter_then_restores(self):
        from axolotl.integrations.scope_rl.trainer import ScopeRLAsyncGRPOTrainer

        seen = []
        trainer = self._trainer(
            0.1, lambda _: seen.append(trainer.vllm_generation.temperature)
        )
        ScopeRLAsyncGRPOTrainer._scope_generate(trainer, ["p"], rank0_only=False)

        self.assertEqual(seen, [1.2])
        self.assertEqual(trainer.vllm_generation.temperature, 1.0)
        self.assertEqual(trainer._metrics["train"]["scope/temperature"], [1.2])

    def test_temperature_is_relative_to_the_sampling_temperature(self):
        from axolotl.integrations.scope_rl.trainer import ScopeRLAsyncGRPOTrainer

        seen = []
        trainer = self._trainer(
            1.0, lambda _: seen.append(trainer.vllm_generation.temperature)
        )
        trainer.vllm_generation.temperature = 0.5
        ScopeRLAsyncGRPOTrainer._scope_generate(trainer, ["p"], rank0_only=False)

        self.assertAlmostEqual(seen[0], 0.5 * 0.8)
        self.assertEqual(trainer.vllm_generation.temperature, 0.5)

    def test_first_rollout_runs_at_the_sampling_temperature(self):
        """No optimizer step has logged entropy yet, so the branch starts neutral."""
        from axolotl.integrations.scope_rl.trainer import ScopeRLAsyncGRPOTrainer

        seen = []
        trainer = self._trainer(
            None, lambda _: seen.append(trainer.vllm_generation.temperature)
        )
        ScopeRLAsyncGRPOTrainer._scope_generate(trainer, ["p"], rank0_only=False)

        self.assertEqual(seen, [1.0])

    def test_restores_temperature_on_failure(self):
        from axolotl.integrations.scope_rl.trainer import ScopeRLAsyncGRPOTrainer

        trainer = self._trainer(0.1, RuntimeError("vllm down"))
        with self.assertRaises(RuntimeError):
            ScopeRLAsyncGRPOTrainer._scope_generate(trainer, ["p"], rank0_only=False)
        self.assertEqual(trainer.vllm_generation.temperature, 1.0)


class TestScopeHooks(unittest.TestCase):
    """Advantage rewrite and loss weighting on the trainer hooks."""

    @staticmethod
    def _trainer(alpha=0.5, threshold=1.0):
        from axolotl.integrations.scope_rl.trainer import ScopeRLAsyncGRPOTrainer

        # Bypass __init__: the hooks only touch args and metrics.
        trainer = ScopeRLAsyncGRPOTrainer.__new__(ScopeRLAsyncGRPOTrainer)
        trainer.args = MagicMock()
        trainer.args.scope_alpha = alpha
        trainer.args.scope_positive_threshold = threshold
        trainer._metrics = {"train": defaultdict(list)}
        return trainer

    def test_aux_rows_keep_positives_at_advantage_one(self):
        trainer = self._trainer()
        advantages = torch.tensor([0.25, -0.25, 0.75, -0.75])
        rewards = torch.tensor([1.0, 0.0, 1.0, 0.0])
        data = {
            "aux_mask": torch.tensor([0.0, 0.0, 1.0, 1.0]),
            "importance_sampling_ratio": torch.full((4, 3), 2.0),
        }
        trainer._apply_scope(data, advantages, rewards, slice(0, 4), "train")

        self.assertEqual(data["advantages"].tolist(), [0.25, -0.25, 1.0, 0.0])
        self.assertEqual(
            data["importance_sampling_ratio"][:, 0].tolist(), [2.0, 2.0, 1.0, 1.0]
        )
        self.assertEqual(
            data["scope_weight"].tolist(),
            scope_weights(data["aux_mask"], 0.5).tolist(),
        )
        self.assertEqual(trainer._metrics["train"]["scope/positive_frac"], [0.5])

    def test_full_batch_mask_is_sliced_to_this_rank(self):
        trainer = self._trainer()
        advantages = torch.tensor([0.5, 0.5])
        rewards = torch.tensor([0.0, 1.0, 1.0, 0.0])
        data = {"aux_mask": torch.tensor([0.0, 0.0, 1.0, 1.0])}
        trainer._apply_scope(data, advantages, rewards, slice(2, 4), "train")

        self.assertEqual(data["aux_mask"].tolist(), [1.0, 1.0])
        self.assertEqual(data["advantages"].tolist(), [1.0, 0.0])

    def test_kl_penalty_skips_aux_rows(self):
        trainer = self._trainer()
        loss = torch.ones(4, 2)
        kl = torch.ones(4, 2)
        aux_mask = torch.tensor([0.0, 0.0, 1.0, 1.0])
        inputs = {"aux_mask": aux_mask, "scope_weight": scope_weights(aux_mask, 0.5)}
        loss, kl = trainer._weight_per_token_loss(loss, kl, inputs)

        self.assertEqual(loss[:, 0].tolist(), inputs["scope_weight"].tolist())
        self.assertEqual(kl[:, 0].tolist(), [2.0, 2.0, 0.0, 0.0])

    def test_loss_hook_is_identity_without_scope_rows(self):
        trainer = self._trainer()
        loss = torch.ones(2, 2)
        out_loss, out_kl = trainer._weight_per_token_loss(loss, None, {})
        self.assertIs(out_loss, loss)
        self.assertIsNone(out_kl)


class TestScopeRLValidator:
    """SCOPE-RL needs `rl: grpo` on the async trainer."""

    @staticmethod
    def _check(data):
        return ScopeRLArgs.check_scope_rl(data)

    @staticmethod
    def _cfg(**kw):
        trl = {"async_prefetch": True, "loss_type": "grpo"}
        for key in ("async_prefetch", "use_data_producer", "loss_type"):
            if key in kw:
                trl[key] = kw.pop(key)
        return {"rl": "grpo", "scope_rl": True, "trl": trl, **kw}

    def test_async_prefetch_passes(self):
        data = self._cfg()
        assert self._check(data) is data

    def test_without_prefetch_raises(self):
        """use_data_producer alone takes the scoring path, where the branch never runs."""
        with pytest.raises(ValueError, match="async_prefetch"):
            self._check(self._cfg(async_prefetch=False, use_data_producer=True))

    def test_token_normalised_loss_type_raises(self):
        with pytest.raises(ValueError, match="loss_type"):
            self._check(self._cfg(loss_type="dapo"))

    def test_unset_loss_type_raises(self):
        """TRL defaults to dapo, so leaving it unset is the common broken case."""
        with pytest.raises(ValueError, match="loss_type"):
            self._check(
                {"rl": "grpo", "scope_rl": True, "trl": {"async_prefetch": True}}
            )

    def test_inverted_temperature_range_raises(self):
        """`min > max` makes the clip collapse to `max`, ignoring the feedback signal."""
        with pytest.raises(ValueError, match="scope_temperature_min"):
            self._check(self._cfg(scope_temperature_min=1.2, scope_temperature_max=0.8))

    def test_non_grpo_raises(self):
        with pytest.raises(ValueError, match="rl: grpo"):
            self._check(
                {"rl": "dpo", "scope_rl": True, "trl": {"async_prefetch": True}}
            )

    def test_foreign_trainer_cls_raises(self):
        with pytest.raises(ValueError, match="trainer_cls"):
            self._check(self._cfg(trainer_cls="my.module.Trainer"))

    def test_plugin_trainer_cls_passes(self):
        data = self._cfg(trainer_cls=TRAINER_CLS)
        assert self._check(data) is data

    def test_disabled_is_ignored(self):
        data = {"rl": "dpo", "scope_rl": False, "trl": {}}
        assert self._check(data) is data


class TestScopeRLPlugin:
    """Plugin wiring: trainer routing and training-arg forwarding."""

    def test_register_routes_to_plugin_trainer(self):
        cfg = DictDefault({"scope_rl": True})
        ScopeRLPlugin().register(cfg)
        assert cfg["trainer_cls"] == TRAINER_CLS

    def test_register_keeps_user_trainer_cls(self):
        cfg = DictDefault({"scope_rl": True, "trainer_cls": "my.Trainer"})
        ScopeRLPlugin().register(cfg)
        assert cfg["trainer_cls"] == "my.Trainer"

    def test_register_is_noop_when_disabled(self):
        cfg = DictDefault({"rl": "grpo"})
        ScopeRLPlugin().register(cfg)
        assert "trainer_cls" not in cfg

    def test_training_args_forward_only_set_keys(self):
        cfg = DictDefault(
            {"scope_rl": True, "scope_alpha": 0.25, "scope_target_entropy": None}
        )
        assert ScopeRLPlugin().get_training_args(cfg) == {
            "scope_rl": True,
            "scope_alpha": 0.25,
        }

    def test_training_args_empty_when_disabled(self):
        cfg = DictDefault({"scope_rl": None, "scope_alpha": 0.25})
        assert ScopeRLPlugin().get_training_args(cfg) == {}

    def test_config_flow_routes_and_forwards_args(self):
        """register -> validate -> get_training_args, as `load_cfg` drives it."""
        from axolotl.cli.config import prepare_plugins
        from axolotl.integrations.base import PluginManager
        from axolotl.utils.config import validate_config

        cfg = DictDefault(
            {
                "base_model": "HuggingFaceTB/SmolLM2-135M",
                "learning_rate": 1e-5,
                "micro_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "datasets": [{"path": "x", "type": "chat_template"}],
                "plugins": ["axolotl.integrations.scope_rl.ScopeRLPlugin"],
                "rl": "grpo",
                "scope_rl": True,
                "scope_alpha": 0.25,
                "trl": {
                    "use_vllm": True,
                    "async_prefetch": True,
                    "loss_type": "grpo",
                    "reward_funcs": ["r"],
                },
            }
        )
        prepare_plugins(cfg)
        cfg = validate_config(cfg)

        assert cfg.trainer_cls == TRAINER_CLS
        assert PluginManager.get_instance().get_training_args(cfg) == {
            "scope_rl": True,
            "scope_alpha": 0.25,
        }


if __name__ == "__main__":
    unittest.main()
