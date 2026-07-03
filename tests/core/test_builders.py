"""Unit tests for axolotl.core.builders SFT and reward-model trainer builders."""

from types import SimpleNamespace

import pytest
from datasets import Dataset

from axolotl.core.builders import HFCausalTrainerBuilder, HFRLTrainerBuilder
from axolotl.core.builders.base import TrainerBuilderBase


def _gradient_checkpointing_kwargs(cfg):
    training_args_kwargs = {}
    TrainerBuilderBase._configure_gradient_checkpointing(
        SimpleNamespace(cfg=cfg), training_args_kwargs
    )
    return training_args_kwargs


class TestGradientCheckpointingConfig:
    def test_hidden_states_offload_uses_non_reentrant_trainer_path(self):
        training_args_kwargs = _gradient_checkpointing_kwargs(
            SimpleNamespace(
                layer_offloading=False,
                activation_offloading="hidden_states",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": False},
            )
        )

        assert training_args_kwargs["gradient_checkpointing"] is True
        assert training_args_kwargs["gradient_checkpointing_kwargs"] == {
            "use_reentrant": False
        }
        assert training_args_kwargs["activation_offloading"] == "hidden_states"

    def test_hidden_states_offload_with_reentrant_stays_on_model_loader_path(self):
        training_args_kwargs = _gradient_checkpointing_kwargs(
            SimpleNamespace(
                layer_offloading=False,
                activation_offloading="hidden_states",
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": True},
            )
        )

        assert training_args_kwargs["gradient_checkpointing"] is True
        assert training_args_kwargs["gradient_checkpointing_kwargs"] == {
            "use_reentrant": True
        }
        assert "activation_offloading" not in training_args_kwargs


def _reward_dataset():
    return Dataset.from_list(
        [
            {
                "chosen_ids": [1, 2, 3],
                "rejected_ids": [1, 4],
            }
        ]
    )


def _prm_dataset():
    return Dataset.from_list(
        [
            {
                "input_ids": [1, 2, 3],
                "labels": [-100, -100, 1],
            }
        ]
    )


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

        if cfg_string == "rm_cfg":
            builder.train_dataset = _reward_dataset()
        elif cfg_string == "prm_cfg":
            builder.train_dataset = _prm_dataset()

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

    def test_sinkgd_optimizer(self, sft_cfg, model, tokenizer):
        cfg = sft_cfg.copy()
        cfg["optimizer"] = "sinkgd"
        cfg["optim_args"] = {"sinkhorn_iters": 5, "sinkgd_lr_scale": 0.05}

        builder = HFCausalTrainerBuilder(cfg, model, tokenizer)
        trainer = builder.build(100)

        assert trainer.optimizer_cls_and_kwargs is not None

        from axolotl.utils.optimizers.sinkgd import SinkGD, SinkGDOptimizerFactory

        optimizer_cls, optimizer_kwargs = trainer.optimizer_cls_and_kwargs
        assert optimizer_cls is SinkGDOptimizerFactory
        assert optimizer_kwargs["lr"] == 0.00005
        assert optimizer_kwargs["weight_decay"] == 0.01
        assert optimizer_kwargs["sinkhorn_iters"] == 5
        assert optimizer_kwargs["sinkgd_lr_scale"] == 0.05

        # OptimizerMixin must resolve the factory into a concrete SinkGD instance
        optim = trainer.create_optimizer()
        assert isinstance(optim, SinkGD)
        # linear weight matrices must be in the stateless SR-Sinkhorn group
        assert any(group.get("use_sinkgd") for group in optim.param_groups)


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
