"""Tests for default values for configurations"""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


class TestDefaultConfigValues:
    """Tests for default values for configurations"""

    @pytest.mark.parametrize(
        "sample_packing",
        [None, True, False],
    )
    def test_pad_to_sequence_len(self, sample_packing, min_base_cfg):
        cfg = (
            DictDefault(
                sample_packing=sample_packing,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)

        assert cfg.pad_to_sequence_len == sample_packing
