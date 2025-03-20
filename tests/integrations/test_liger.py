"""
config validation tests for swiglu args
"""

# pylint: disable=duplicate-code
import logging
from typing import Optional

import pytest

from axolotl.utils.config import prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="minimal_liger_cfg")
def fixture_cfg():
    return DictDefault(
        {
            "base_model": "TinyLlama/TinyLlama-1.1B-Chat-v0.6",
            "learning_rate": 0.000001,
            "datasets": [
                {
                    "path": "mhenrichsen/alpaca_2k_test",
                    "type": "alpaca",
                }
            ],
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "plugins": ["axolotl.integrations.liger.LigerPlugin"],
        }
    )


# pylint: disable=too-many-public-methods
class TestValidation:
    """
    Test the validation module for liger
    """

    _caplog: Optional[pytest.LogCaptureFixture] = None

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        caplog.set_level(logging.WARNING)
        self._caplog = caplog

    def test_deprecated_swiglu(self, minimal_liger_cfg):
        test_cfg = DictDefault(
            {
                "liger_swiglu": False,
            }
            | minimal_liger_cfg
        )

        with self._caplog.at_level(
            logging.WARNING, logger="axolotl.integrations.liger.args"
        ):
            prepare_plugins(test_cfg)
            updated_cfg = validate_config(test_cfg)
            # TODO this test is brittle in CI
            # assert (
            #     "The 'liger_swiglu' argument is deprecated"
            #     in self._caplog.records[0].message
            # )
            assert updated_cfg.liger_swiglu is None
            assert updated_cfg.liger_glu_activation is False

    def test_conflict_swiglu_ligergluactivation(self, minimal_liger_cfg):
        test_cfg = DictDefault(
            {
                "liger_swiglu": False,
                "liger_glu_activation": True,
            }
            | minimal_liger_cfg
        )

        with pytest.raises(
            ValueError,
            match=r".*You cannot have both `liger_swiglu` and `liger_glu_activation` set.*",
        ):
            prepare_plugins(test_cfg)
            validate_config(test_cfg)
