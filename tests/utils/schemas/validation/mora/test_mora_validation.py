"""Validation tests for the MoRA / ReMoRA integration."""

import pytest

from axolotl.utils.config import prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault


class TestMoraValidation:
    """MoRA-specific config validation."""

    def test_mora_block_round_trips(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            {
                "adapter": "mora",
                "plugins": ["axolotl.integrations.mora.MoraPlugin"],
                "mora": {
                    "use_mora": True,
                    "mora_type": 6,
                },
            }
        )

        prepare_plugins(cfg)
        validated = validate_config(cfg)

        assert validated.adapter == "mora"
        assert validated.mora.use_mora is True
        assert validated.mora.mora_type == 6

    def test_remora_maps_to_existing_relora_fields(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            {
                "adapter": "mora",
                "plugins": ["axolotl.integrations.mora.MoraPlugin"],
                "mora": {
                    "use_mora": True,
                    "mora_type": 6,
                    "use_relora": True,
                    "use_relora_step": 2000,
                },
            }
        )

        prepare_plugins(cfg)
        validated = validate_config(cfg)

        assert validated.relora is True
        assert validated.jagged_restart_steps == 2000
        assert validated.mora.use_relora is True
        assert validated.mora.use_relora_step == 2000

    def test_remora_requires_step(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            {
                "adapter": "mora",
                "plugins": ["axolotl.integrations.mora.MoraPlugin"],
                "mora": {
                    "use_mora": True,
                    "use_relora": True,
                },
            }
        )

        prepare_plugins(cfg)
        with pytest.raises(ValueError, match="use_relora_step"):
            validate_config(cfg)

    def test_mora_rejects_quantized_base_model(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            {
                "adapter": "mora",
                "plugins": ["axolotl.integrations.mora.MoraPlugin"],
                "load_in_4bit": True,
                "mora": {
                    "use_mora": True,
                    "mora_type": 6,
                },
            }
        )

        prepare_plugins(cfg)
        with pytest.raises(ValueError, match="full-precision base model"):
            validate_config(cfg)
