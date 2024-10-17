"""
config validation tests for swiglu args
"""
# pylint: disable=duplicate-code
import logging
from typing import Optional

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="minimal_base_cfg")
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
        }
    )


class BaseValidation:
    """
    Base validation module to setup the log capture
    """

    _caplog: Optional[pytest.LogCaptureFixture] = None

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog


# pylint: disable=too-many-public-methods
class TestValidation(BaseValidation):
    """
    Test the validation module for liger
    """

    def test_deprecated_swiglu(self, minimal_cfg):
        test_cfg = DictDefault(
            {
                "liger_swiglu": False,
            }
            | minimal_cfg
        )

        with self._caplog.at_level(logging.WARNING):
            updated_cfg = validate_config(test_cfg)
            assert (
                "The 'liger_swiglu' argument is deprecated"
                in self._caplog.records[0].message
            )
            assert updated_cfg.liger_swiglu is None
            assert updated_cfg.liger_glu_activations is False

    def test_conflict_swiglu_ligergluactivation(self, minimal_cfg):
        test_cfg = DictDefault(
            {
                "liger_swiglu": False,
                "liger_glu_activations": True,
            }
            | minimal_cfg
        )

        with pytest.raises(
            ValueError,
            match=r".*You cannot have both `liger_swiglu` and `liger_glu_activation` set.*",
        ):
            validate_config(test_cfg)
