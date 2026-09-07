"""Tests for default values for configurations"""

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
