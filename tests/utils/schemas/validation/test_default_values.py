"""Tests for default values for configurations"""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


class TestDefaultConfigValues:
    """Tests for default values for configurations"""

    def test_pad_to_sequence_len(self, min_base_cfg):
        """Tests that sample packing automatically sets pad_to_sequence_len to True"""
        cfg = (
            DictDefault(
                sample_packing=True,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)

        assert cfg.pad_to_sequence_len is True

    def test_pad_to_sequence_len_auto(self, min_base_cfg):
        """`auto` is kept without packing, coerced to True with packing, rejected for streaming"""
        cfg = validate_config(DictDefault(pad_to_sequence_len="auto") | min_base_cfg)
        assert cfg.pad_to_sequence_len == "auto"

        cfg = validate_config(
            DictDefault(pad_to_sequence_len="auto", sample_packing=True) | min_base_cfg
        )
        assert cfg.pad_to_sequence_len is True

        with pytest.raises(ValueError, match="map-style dataset"):
            validate_config(
                DictDefault(pad_to_sequence_len="auto", streaming=True) | min_base_cfg
            )
