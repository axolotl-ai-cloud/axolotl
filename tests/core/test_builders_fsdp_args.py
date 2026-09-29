"""Tests for the FSDP training arguments set by the base trainer builder."""

from unittest.mock import MagicMock

from axolotl.core.builders.base import TrainerBuilderBase
from axolotl.utils.dict import DictDefault


def _base_training_args(cfg):
    builder = MagicMock(spec=TrainerBuilderBase)
    builder.cfg = cfg
    training_args_kwargs, _ = TrainerBuilderBase._set_base_training_args(builder, 10)
    return training_args_kwargs


def test_fsdp_config_sets_fsdp_true():
    fsdp_config = DictDefault({"reshard_after_forward": True})
    training_args_kwargs = _base_training_args(
        DictDefault({"micro_batch_size": 1, "fsdp_config": fsdp_config})
    )
    assert training_args_kwargs["fsdp"] is True
    assert training_args_kwargs["fsdp_config"] == fsdp_config


def test_no_fsdp_config_leaves_fsdp_unset():
    training_args_kwargs = _base_training_args(DictDefault({"micro_batch_size": 1}))
    assert "fsdp" not in training_args_kwargs
    assert "fsdp_config" not in training_args_kwargs
