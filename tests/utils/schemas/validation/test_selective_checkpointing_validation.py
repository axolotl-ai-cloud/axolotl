"""Test for config validation for selective activation checkpointing."""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


class TestSelectiveCheckpointing:
    """
    Test cases for selective_checkpointing schema validation
    """

    def test_bool_shorthand_normalizes_to_attention(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                selective_checkpointing=True,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.selective_checkpointing["save"] == ["attention"]
        assert cfg.selective_checkpointing["save_sliding_window"] is False

    def test_false_normalizes_to_none(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                selective_checkpointing=False,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.selective_checkpointing is None

    def test_custom_save_list_preserved(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                selective_checkpointing={"save": ["attention", "aten::mm"]},
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.selective_checkpointing["save"] == ["attention", "aten::mm"]

    def test_requires_gradient_checkpointing(self, min_base_cfg):
        cfg = (
            DictDefault(
                selective_checkpointing=True,
            )
            | min_base_cfg
        )

        with pytest.raises(ValueError, match="requires gradient_checkpointing"):
            validate_config(cfg)

    def test_rejects_reentrant(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": True},
                selective_checkpointing=True,
            )
            | min_base_cfg
        )

        with pytest.raises(ValueError, match="non-reentrant"):
            validate_config(cfg)

    def test_rejects_trl_activation_offloading(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                activation_offloading=True,
                selective_checkpointing=True,
            )
            | min_base_cfg
        )

        with pytest.raises(ValueError, match="hidden_states"):
            validate_config(cfg)

    def test_composes_with_hidden_states_offload(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                activation_offloading="hidden_states",
                selective_checkpointing=True,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.selective_checkpointing["save"] == ["attention"]
        assert cfg.activation_offloading == "hidden_states"

    def test_rejects_matmul_save_with_adapter(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                adapter="lora",
                lora_r=16,
                lora_alpha=32,
                lora_target_linear=True,
                selective_checkpointing={"save": ["attention", "aten::mm"]},
            )
            | min_base_cfg
        )

        with pytest.raises(ValueError, match="in-place"):
            validate_config(cfg)

    def test_allows_matmul_save_without_adapter(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                selective_checkpointing={"save": ["attention", "aten::mm"]},
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.selective_checkpointing["save"] == ["attention", "aten::mm"]
