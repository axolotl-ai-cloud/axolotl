"""Unit tests for axolotl.core.builders"""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from torch.optim import AdamW

from axolotl.common.datasets import load_datasets
from axolotl.core.builders import HFCausalTrainerBuilder, HFRLTrainerBuilder
from axolotl.loaders import ModelLoader, load_tokenizer
from axolotl.utils.config import normalize_config
from axolotl.utils.data import prepare_preference_datasets
from axolotl.utils.dict import DictDefault
from axolotl.utils.schemas.enums import RLType

from tests.constants import ALPACA_MESSAGES_CONFIG_REVISION


@pytest.fixture(name="base_cfg")
def fixture_base_cfg():
    """
    Base config with all common arguments between SFT and RLHF
    """
    cfg = DictDefault(
        {
            # Model and tokenizer settings
            "base_model": "HuggingFaceTB/SmolLM2-135M-Instruct",
            "sequence_len": 2048,
            "model_config_type": "llama",  # example type
            # Basic training settings
            "micro_batch_size": 2,
            "eval_batch_size": 2,
            "num_epochs": 1,
            "gradient_accumulation_steps": 1,
            "max_steps": 100,
            "val_set_size": 0,
            # Optimizer settings
            "optimizer": "adamw_torch_fused",
            "learning_rate": 0.00005,
            "weight_decay": 0.01,
            "adam_beta1": 0.998,
            "adam_beta2": 0.9,
            "adam_epsilon": 0.00001,
            "max_grad_norm": 1.0,
            # LR scheduler settings
            "lr_scheduler": "cosine",
            "lr_scheduler_kwargs": {"foo": "bar"},
            "warmup_steps": 10,
            "warmup_ratio": None,
            "cosine_min_lr_ratio": 0.1,
            "cosine_constant_lr_ratio": 0.2,
            # Checkpointing and saving
            "save_steps": 100,
            "output_dir": "./model-out",
            "save_safetensors": True,
            "save_total_limit": 4,
            "save_only_model": False,
            # Hardware/performance settings
            "gradient_checkpointing": False,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "dataloader_num_workers": 1,
            "dataloader_pin_memory": True,
            "dataloader_prefetch_factor": 2,
            "context_parallel_size": 1,
            "tensor_parallel_size": 1,
            # Dtype
            "fp16": False,
            "bf16": False,
            "tf32": False,
            # Logging and evaluation
            "logging_steps": 10,
            "eval_steps": 50,
            "eval_strategy": "steps",
            "save_strategy": "steps",
            "include_tokens_per_second": True,
            # Other common settings
            "seed": 42,
            "remove_unused_columns": True,
            "ddp_timeout": 1800,
            "ddp_bucket_cap_mb": 25,
            "ddp_broadcast_buffers": False,
            "dataset_processes": 4,
        }
    )

    normalize_config(cfg)
    return cfg


@pytest.fixture(name="dpo_cfg")
def fixture_dpo_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": RLType.DPO,
            "dpo_use_weighting": True,
            "dpo_use_logits_to_keep": True,
            "dpo_label_smoothing": 0.1,
            "beta": 0.1,  # DPO beta
        }
    )
    return cfg


@pytest.fixture(name="orpo_cfg")
def fixture_orpo_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": RLType.ORPO,
            "orpo_alpha": 0.1,
            "max_prompt_len": 512,
        }
    )
    return cfg


@pytest.fixture(name="kto_cfg")
def fixture_kto_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": RLType.KTO,
            "kto_desirable_weight": 1.0,
            "kto_undesirable_weight": 1.0,
            "max_prompt_len": 512,
        }
    )
    return cfg


@pytest.fixture(name="grpo_cfg")
def fixture_grpo_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": RLType.GRPO,
            "trl": DictDefault(
                {
                    "beta": 0.001,
                    "max_completion_length": 256,
                    "use_vllm": False,  # run on CPU
                    # "vllm_device": "auto",
                    # "vllm_gpu_memory_utilization": 0.15,
                    "num_generations": 4,
                    "reward_funcs": ["rewards.rand_reward_func"],
                }
            ),
            # Must be evenly divisible by num_generations
            "micro_batch_size": 4,
        }
    )
    return cfg


@pytest.fixture(name="ipo_cfg")
def fixture_ipo_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": RLType.IPO,
            "dpo_label_smoothing": 0,
            "beta": 0.1,
        }
    )
    return cfg


@pytest.fixture(name="simpo_cfg")
def fixture_simpo_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": RLType.SIMPO,
            "rl_beta": 0.2,
            "cpo_alpha": 0.9,
            "simpo_gamma": 0.4,
        }
    )
    return cfg


@pytest.fixture(name="sft_cfg")
def fixture_sft_cfg(base_cfg):
    cfg = base_cfg.copy()
    cfg.update(
        {
            "rl": None,
            "sample_packing": False,
            "eval_sample_packing": False,
            "flash_attention": False,
        }
    )
    return cfg


@pytest.fixture(name="rm_cfg")
def fixture_rm_cfg(sft_cfg):
    cfg = sft_cfg.copy()
    cfg.update(
        DictDefault(
            {
                "reward_model": True,
                "datasets": [
                    {
                        "path": "argilla/distilabel-intel-orca-dpo-pairs",
                        "type": "bradley_terry.chat_template",
                        "split": "train[:1%]",
                    }
                ],
            }
        )
    )
    return cfg


@pytest.fixture(name="prm_cfg")
def fixture_prm_cfg(sft_cfg):
    cfg = sft_cfg.copy()
    cfg.update(
        DictDefault(
            {
                "process_reward_model": True,
                "datasets": [
                    {
                        "path": "trl-lib/math_shepherd",
                        "type": "stepwise_supervised",
                        "split": "train[:1%]",
                    }
                ],
            }
        )
    )
    return cfg


@pytest.fixture(name="tokenizer")
def fixture_tokenizer(base_cfg):
    return load_tokenizer(base_cfg)


@pytest.fixture(name="model")
def fixture_model(base_cfg, tokenizer):
    model, _ = ModelLoader(base_cfg, tokenizer).load()
    return model


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
        builder = HFRLTrainerBuilder(dpo_cfg, model, tokenizer)
        training_arguments, _ = builder._build_training_arguments(100)

        self._test_common_training_arguments(training_arguments, rl=dpo_cfg.rl)
        # DPO specific
        assert training_arguments.beta == 0.1
        assert hasattr(training_arguments, "use_weighting")
        assert training_arguments.use_weighting is True
        assert training_arguments.label_smoothing == 0.1

    def test_orpo_training_arguments(self, orpo_cfg, model, tokenizer):
        builder = HFRLTrainerBuilder(orpo_cfg, model, tokenizer)
        training_arguments, _ = builder._build_training_arguments(100)

        self._test_common_training_arguments(training_arguments, rl=orpo_cfg.rl)
        # ORPO specific
        assert training_arguments.beta == 0.1  # maps from orpo_alpha
        assert training_arguments.max_prompt_length == 512

    def test_kto_training_arguments(self, kto_cfg, model, tokenizer):
        builder = HFRLTrainerBuilder(kto_cfg, model, tokenizer)
        training_arguments, _ = builder._build_training_arguments(100)

        self._test_common_training_arguments(training_arguments, rl=kto_cfg.rl)
        # KTO specific
        assert training_arguments.desirable_weight == 1.0
        assert training_arguments.undesirable_weight == 1.0
        assert training_arguments.max_prompt_length == 512

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
        assert training_arguments.loss_type == "ipo"
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
        cfg["dataset_num_proc"] = 4

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

            optimizer_cls, optimizer_kwargs = trainer.optimizer_cls_and_kwargs
            assert optimizer_cls is AdamW
            assert optimizer_kwargs["lr"] == 0.00005
            assert optimizer_kwargs["weight_decay"] == 0.01
            assert optimizer_kwargs["betas"] == (0.998, 0.9)
            assert optimizer_kwargs["eps"] == 0.00001

            # Ensure optimizer is created with correct class
            optim = trainer.create_optimizer()
            assert isinstance(optim, AdamW)

        finally:
            # remove imported module from path
            if cfg_string == "grpo_cfg" and str(rewards_dir) in sys.path:
                sys.path.remove(str(rewards_dir))


class TestHFCausalTrainerBuilder:
    """
    TestCase class for SFT trainer builder
    """

    def test_training_arguments(self, sft_cfg, model, tokenizer):
        builder = HFCausalTrainerBuilder(sft_cfg, model, tokenizer)
        trainer = builder.build(100)
        training_arguments = trainer.args

        # Test common arguments
        assert training_arguments.per_device_train_batch_size == 2
        assert training_arguments.gradient_accumulation_steps == 1
        assert training_arguments.max_steps == 100

        assert training_arguments.learning_rate == 0.00005
        assert training_arguments.weight_decay == 0.01
        assert training_arguments.adam_beta1 == 0.998
        assert training_arguments.adam_beta2 == 0.9
        assert training_arguments.adam_epsilon == 0.00001
        assert training_arguments.max_grad_norm == 1.0

        assert training_arguments.lr_scheduler_type == "cosine"
        assert training_arguments.warmup_steps == 10
        assert training_arguments.cosine_min_lr_ratio == 0.1

        assert training_arguments.dataloader_num_workers == 1
        assert training_arguments.dataloader_pin_memory is True
        assert training_arguments.gradient_checkpointing is False

        # SFT specific
        assert training_arguments.sample_packing is False
        assert training_arguments.eval_sample_packing is False

    @pytest.mark.parametrize(
        "cfg_string",
        [
            "sft_cfg",
            "rm_cfg",
            "prm_cfg",
        ],
    )
    def test_custom_optimizer_cls_and_kwargs(
        self, request, cfg_string, model, tokenizer
    ):
        cfg = request.getfixturevalue(cfg_string)
        builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
        cfg["optimizer"] = "muon"

        # need to load datasets for reward model and process reward model trainer
        if cfg_string in ["rm_cfg", "prm_cfg"]:
            dataset_meta = load_datasets(cfg=cfg)

            builder.train_dataset = dataset_meta.train_dataset
            builder.eval_dataset = dataset_meta.eval_dataset

        trainer = builder.build(100)

        assert trainer.optimizer_cls_and_kwargs is not None

        optimizer_cls, optimizer_kwargs = trainer.optimizer_cls_and_kwargs
        assert optimizer_cls is AdamW
        assert optimizer_kwargs["lr"] == 0.00005
        assert optimizer_kwargs["weight_decay"] == 0.01
        assert optimizer_kwargs["betas"] == (0.998, 0.9)
        assert optimizer_kwargs["eps"] == 0.00001

        # Ensure optimizer is created with correct class
        optim = trainer.create_optimizer()
        assert isinstance(optim, AdamW)

    def test_muon_optimizer_skips_muon_params(self, sft_cfg, model, tokenizer):
        cfg = sft_cfg.copy()
        cfg["optimizer"] = "muon"
        builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
        trainer = builder.build(10)

        optimizer = trainer.create_optimizer()
        muon_param_ids = {
            id(param)
            for param in trainer.model.parameters()
            if getattr(param, "use_muon", False)
        }
        assert muon_param_ids

        for group in optimizer.param_groups:
            contains_muon = any(id(p) in muon_param_ids for p in group["params"])
            if contains_muon:
                assert group["lr"] == 0.0
                assert group.get("weight_decay", 0.0) == 0.0
            else:
                assert group["lr"] > 0


class TestTrainerClsPlugin:
    """
    TestCase class for trainer builder with plugin
    """

    def test_trainer_cls_is_not_none_with_plugin(self, kto_cfg, model, tokenizer):
        """
        Test that the trainer cls is not none with plugin

        Fixes #2693
        """
        cfg = kto_cfg.copy()
        cfg.plugins = ["axolotl.integrations.liger.LigerPlugin"]

        # Expected AttributeError as we don't pass regular model configs to RL trainer builder
        # If it throws `TypeError: None is not a callable object`, trainer_cls could be None
        try:
            builder = HFRLTrainerBuilder(cfg, model, tokenizer)

            builder.build(100)
        except TypeError as e:
            # Error raised if trainer_cls is None
            assert "'tuple' object has no attribute 'config'" not in str(e)
        except Exception:
            # Another error happens, so we passed trainer_cls to builder
            pass
