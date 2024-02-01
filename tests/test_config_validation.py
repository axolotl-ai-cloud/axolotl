import pytest

from axolotl.utils.config.models.input.v0_4_1 import AxolotlInputConfig
from axolotl.utils.dict import DictDefault


@pytest.fixture(name="min_cfg")
def fixture_min_cfg():
    return DictDefault(
        {
            "base_model": "lorem/ipsum",
            "learning_rate": 1.0e-6,
            "datasets": [{"path": "dolor/sit", "type": "sharegpt"}],
            "sequence_len": 1024,
            "gradient_accumulation_steps": 1,
            "micro_batch_size": 1,
            "num_epochs": 4,
            "output_dir": "./model-out",
        }
    )


class TestPydanticConfigValidation:
    def test_something(self, min_cfg):
        cfg = AxolotlInputConfig(**min_cfg.to_dict())
