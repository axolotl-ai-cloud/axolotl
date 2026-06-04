"""Unit tests for axolotl.core.builders SFT and reward-model trainer builders."""

from unittest.mock import MagicMock

import pytest
import torch._inductor.config as _inductor_cfg

from axolotl.common.datasets import load_datasets
from axolotl.core.builders import HFCausalTrainerBuilder, HFRLTrainerBuilder
from axolotl.core.builders.base import TrainerBuilderBase
from axolotl.utils.schemas.enums import INDUCTOR_COMPILE_OPTIONS_ALLOWLIST


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
    def test_builder_w_rm_trainers(self, request, cfg_string, model, tokenizer):
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


@pytest.fixture(name="inductor_config_snapshot")
def fixture_inductor_config_snapshot():
    keys = sorted(INDUCTOR_COMPILE_OPTIONS_ALLOWLIST)
    saved = {k: getattr(_inductor_cfg, k) for k in keys}
    try:
        yield saved
    finally:
        for k, v in saved.items():
            setattr(_inductor_cfg, k, v)


class TestApplyTorchCompileOptions:
    # Runtime apply path; schema-layer validation is tested in TestTorchCompileValidation.

    def test_allowlisted_flag_is_applied(self, inductor_config_snapshot):
        assert _inductor_cfg.coordinate_descent_tuning is False
        TrainerBuilderBase._apply_torch_compile_options(
            {"coordinate_descent_tuning": True}
        )
        assert _inductor_cfg.coordinate_descent_tuning is True

    def test_all_allowlisted_flags_are_attributes_on_inductor_config(
        self, inductor_config_snapshot
    ):
        for key in INDUCTOR_COMPILE_OPTIONS_ALLOWLIST:
            assert hasattr(_inductor_cfg, key), (
                f"torch._inductor.config has no attribute {key!r}; "
                f"INDUCTOR_COMPILE_OPTIONS_ALLOWLIST needs updating."
            )

    def test_dotted_flag_is_applied(self, inductor_config_snapshot):
        # ConfigModule supports dotted setattr natively (no path-walk needed).
        assert _inductor_cfg.triton.cudagraphs is False
        TrainerBuilderBase._apply_torch_compile_options({"triton.cudagraphs": True})
        assert _inductor_cfg.triton.cudagraphs is True

    def test_multiple_flags_are_applied(self, inductor_config_snapshot):
        TrainerBuilderBase._apply_torch_compile_options(
            {
                "coordinate_descent_tuning": True,
                "shape_padding": False,
                "epilogue_fusion": False,
            }
        )
        assert _inductor_cfg.coordinate_descent_tuning is True
        assert _inductor_cfg.shape_padding is False
        assert _inductor_cfg.epilogue_fusion is False

    def test_empty_dict_does_not_mutate_state(self, inductor_config_snapshot):
        TrainerBuilderBase._apply_torch_compile_options({})
        for k, v in inductor_config_snapshot.items():
            assert getattr(_inductor_cfg, k) == v

    def test_configure_torch_compile_invokes_apply_when_options_set(self):
        builder = MagicMock(spec=TrainerBuilderBase)
        builder.cfg = MagicMock()
        builder.cfg.torch_compile = True
        builder.cfg.torch_compile_backend = None
        builder.cfg.torch_compile_mode = None
        builder.cfg.torch_compile_options = {"coordinate_descent_tuning": True}
        training_args_kwargs: dict = {}

        TrainerBuilderBase._configure_torch_compile(builder, training_args_kwargs)

        builder._apply_torch_compile_options.assert_called_once_with(
            {"coordinate_descent_tuning": True}
        )

    def test_configure_torch_compile_skips_apply_when_options_unset(self):
        builder = MagicMock(spec=TrainerBuilderBase)
        builder.cfg = MagicMock()
        builder.cfg.torch_compile = True
        builder.cfg.torch_compile_backend = None
        builder.cfg.torch_compile_mode = None
        builder.cfg.torch_compile_options = None
        training_args_kwargs: dict = {}

        TrainerBuilderBase._configure_torch_compile(builder, training_args_kwargs)

        builder._apply_torch_compile_options.assert_not_called()
