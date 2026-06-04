"""Unit tests for axolotl.core.builders RL (preference/GRPO) trainer builders."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from axolotl.core.builders import HFRLTrainerBuilder
from axolotl.utils.data import prepare_preference_datasets
from axolotl.utils.dict import DictDefault

from tests.constants import ALPACA_MESSAGES_CONFIG_REVISION


class TestHFRLTrainerBuilder:
    """
    TestCase class for RLHF trainer builders
    """

    def _test_common_training_arguments(self, training_arguments, rl: str):
        """Helper to test common arguments across all variants"""
        # Basic training settings
        if rl == "grpo":
            # grpo_cfg's micro_batch_size is diff from others
            assert training_arguments.per_device_train_batch_size == 4
        else:
            assert training_arguments.per_device_train_batch_size == 2
        assert training_arguments.gradient_accumulation_steps == 1
        assert training_arguments.max_steps == 100

        # Optimizer settings
        assert training_arguments.learning_rate == 0.00005
        assert training_arguments.weight_decay == 0.01
        assert training_arguments.adam_beta1 == 0.998
        assert training_arguments.adam_beta2 == 0.9
        assert training_arguments.adam_epsilon == 0.00001
        assert training_arguments.max_grad_norm == 1.0

        # LR scheduler settings
        assert training_arguments.lr_scheduler_type == "cosine"
        assert training_arguments.warmup_steps == 10
        assert training_arguments.cosine_min_lr_ratio == 0.1
        assert training_arguments.cosine_constant_lr_ratio == 0.2

        # Other settings
        assert training_arguments.dataloader_num_workers == 1
        assert training_arguments.dataloader_pin_memory is True

        # TODO(wing): restore once trl releases 0.22.0
        # assert training_arguments.gradient_checkpointing is True

    def test_dpo_training_arguments(self, dpo_cfg, model, tokenizer):
        dpo_cfg["precompute_ref_log_probs"] = True
        builder = HFRLTrainerBuilder(dpo_cfg, model, tokenizer)
        training_arguments, _ = builder._build_training_arguments(100)

        self._test_common_training_arguments(training_arguments, rl=dpo_cfg.rl)
        # DPO specific
        assert training_arguments.beta == 0.1
        assert hasattr(training_arguments, "use_weighting")
        assert training_arguments.use_weighting is True
        assert training_arguments.label_smoothing == 0.1
        assert training_arguments.precompute_ref_log_probs is True
        assert training_arguments.loss_type == ["sigmoid", "sft"]
        assert training_arguments.loss_weights == [1.0, 0.5]

    def test_orpo_training_arguments(self, orpo_cfg, model, tokenizer):
        builder = HFRLTrainerBuilder(orpo_cfg, model, tokenizer)
        training_arguments, _ = builder._build_training_arguments(100)

        self._test_common_training_arguments(training_arguments, rl=orpo_cfg.rl)
        # ORPO specific
        assert training_arguments.beta == 0.1  # maps from orpo_alpha

    def test_kto_training_arguments(self, kto_cfg, model, tokenizer):
        builder = HFRLTrainerBuilder(kto_cfg, model, tokenizer)
        training_arguments, _ = builder._build_training_arguments(100)

        self._test_common_training_arguments(training_arguments, rl=kto_cfg.rl)
        # KTO specific
        assert training_arguments.desirable_weight == 1.0
        assert training_arguments.undesirable_weight == 1.0

    def _write_rewards_file(self, rewards_dir: Path):
        """
        Writes reward function to local tmp path to be loaded on trainer building
        """
        # Create rewards.py in a directory we can import from
        rewards_dir.mkdir()
        rewards_file = rewards_dir / "rewards.py"
        rewards_file.write_text(
            """import random
def rand_reward_func(prompts, completions) -> list[float]:
    return [random.uniform(0, 1) for _ in completions]
"""
        )

    def test_grpo_training_arguments(self, grpo_cfg, model, tokenizer, tmp_path):
        rewards_dir = tmp_path / "rewards_test"
        self._write_rewards_file(rewards_dir)

        # Add the directory to Python path so we can import the module
        sys.path.insert(0, str(rewards_dir))

        try:
            builder = HFRLTrainerBuilder(grpo_cfg, model, tokenizer)
            training_arguments, _ = builder._build_training_arguments(100)
            builder.train_dataset = MagicMock()

            self._test_common_training_arguments(training_arguments, rl=grpo_cfg.rl)
            # GRPO specific
            assert training_arguments.beta == 0.001
            assert training_arguments.max_completion_length == 256
            assert training_arguments.use_vllm is False
            # assert training_arguments.vllm_device == "auto"
            # assert training_arguments.vllm_gpu_memory_utilization == 0.15
            assert training_arguments.num_generations == 4

            # Test trainer creation to verify reward_funcs
            trainer = builder.build(100)

            # Verify reward functions are properly loaded
            assert len(trainer.reward_funcs) == 1
            assert trainer.reward_funcs[0].__module__ == "rewards"
            assert trainer.reward_funcs[0].__name__ == "rand_reward_func"
        finally:
            # remove imported module from path
            if str(rewards_dir) in sys.path:
                sys.path.remove(str(rewards_dir))

    def test_ipo_training_arguments(self, ipo_cfg, model, tokenizer):
        builder = HFRLTrainerBuilder(ipo_cfg, model, tokenizer)
        training_arguments, _ = builder._build_training_arguments(100)

        self._test_common_training_arguments(training_arguments, rl=ipo_cfg.rl)
        # IPO specific
        assert training_arguments.beta == 0.1
        assert training_arguments.loss_type == ["ipo"]
        assert training_arguments.label_smoothing == 0

    def test_simpo_training_arguments(self, simpo_cfg, model, tokenizer):
        builder = HFRLTrainerBuilder(simpo_cfg, model, tokenizer)
        training_arguments, _ = builder._build_training_arguments(100)

        self._test_common_training_arguments(training_arguments, rl=simpo_cfg.rl)
        # SIMPO specific
        assert training_arguments.beta == 0.2
        assert training_arguments.cpo_alpha == 0.9
        assert training_arguments.simpo_gamma == 0.4

    @pytest.mark.parametrize(
        ("cfg_string", "dataset_name"),
        [
            (
                "dpo_cfg",
                "dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff",
            ),
            (
                "ipo_cfg",
                "dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff",
            ),
            (
                "grpo_cfg",
                "dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff",
            ),
            ("orpo_cfg", None),  # don't use fixture for orpo to use smaller split
            ("kto_cfg", None),  # no fixture for kto
            # (
            #     "simpo_cfg",
            #     "dataset_fozziethebeat_alpaca_messages_2k_dpo_test_rev_ea82cff",
            # ),
        ],
    )
    def test_custom_optimizer_cls_and_kwargs(
        self,
        request,
        cfg_string,
        dataset_name,
        tmp_path,
        model,
        tokenizer,
    ):
        cfg = request.getfixturevalue(cfg_string)

        builder = HFRLTrainerBuilder(cfg, model, tokenizer)
        cfg["optimizer"] = "muon"

        if cfg_string in ["dpo_cfg", "ipo_cfg", "grpo_cfg", "simpo_cfg"]:
            cfg["datasets"] = [DictDefault(ALPACA_MESSAGES_CONFIG_REVISION)]
        elif cfg_string == "kto_cfg":
            cfg["datasets"] = [
                DictDefault(
                    {
                        "path": "argilla/ultrafeedback-binarized-preferences-cleaned-kto",
                        "type": "llama3.ultra",
                        "split": "train[:1%]",
                    }
                )
            ]
        elif cfg_string == "orpo_cfg":
            cfg["datasets"] = [
                DictDefault(
                    {
                        "path": "argilla/ultrafeedback-binarized-preferences-cleaned",
                        "type": "chat_template.argilla",
                        "split": "train[:1%]",
                    }
                )
            ]
        else:
            raise ValueError(f"Unhandled cfg_string: {cfg_string}")
        cfg["dataset_num_proc"] = 1

        if cfg_string == "grpo_cfg":
            rewards_dir = tmp_path / "rewards_test"
            self._write_rewards_file(rewards_dir)

            # Add the directory to Python path so we can import the module
            sys.path.insert(0, str(rewards_dir))

        try:
            # Only use mock for the commented out configs
            if dataset_name is not None:
                with patch(
                    "axolotl.utils.data.rl.load_dataset_with_config"
                ) as mock_load_dataset:
                    mock_load_dataset.return_value = request.getfixturevalue(
                        dataset_name
                    )
                    train_dataset, eval_dataset = prepare_preference_datasets(
                        cfg, tokenizer
                    )
            else:
                # Load actual datasets for orpo_cfg and kto_cfg
                train_dataset, eval_dataset = prepare_preference_datasets(
                    cfg, tokenizer
                )

            builder.train_dataset = train_dataset
            builder.eval_dataset = eval_dataset

            trainer = builder.build(100)

            assert trainer.optimizer_cls_and_kwargs is not None

            from axolotl.contribs.mit.muon import MuonOptimizerFactory
            from axolotl.contribs.mit.muon.muon import Muon

            optimizer_cls, optimizer_kwargs = trainer.optimizer_cls_and_kwargs
            assert optimizer_cls is MuonOptimizerFactory
            assert optimizer_kwargs["lr"] == 0.00005
            assert optimizer_kwargs["weight_decay"] == 0.01
            assert optimizer_kwargs["betas"] == (0.998, 0.9)
            assert optimizer_kwargs["eps"] == 0.00001

            # Ensure optimizer is created with correct class
            optim = trainer.create_optimizer()
            assert isinstance(optim, Muon)

        finally:
            # remove imported module from path
            if cfg_string == "grpo_cfg" and str(rewards_dir) in sys.path:
                sys.path.remove(str(rewards_dir))
