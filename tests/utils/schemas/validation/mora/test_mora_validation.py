"""Validation tests for the MoRA / ReMoRA integration."""

import pytest

from axolotl.integrations.mora import MoraType
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
                    "mora_type": "rope",
                },
            }
        )

        prepare_plugins(cfg)
        validated = validate_config(cfg)

        assert validated.adapter == "mora"
        assert validated.mora.use_mora is True
        assert validated.mora.mora_type == MoraType.ROPE

    def test_mora_type_accepts_legacy_supported_numbers(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            {
                "adapter": "mora",
                "plugins": ["axolotl.integrations.mora.MoraPlugin"],
                "mora": {
                    "use_mora": True,
                    "mora_type": 1,
                },
            }
        )

        prepare_plugins(cfg)
        validated = validate_config(cfg)

        assert validated.mora.mora_type == MoraType.SHARING

    def test_mora_rejects_unsupported_variant_numbers(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            {
                "adapter": "mora",
                "plugins": ["axolotl.integrations.mora.MoraPlugin"],
                "mora": {
                    "use_mora": True,
                    "mora_type": 2,
                },
            }
        )

        prepare_plugins(cfg)
        with pytest.raises(ValueError, match="mora_type"):
            validate_config(cfg)

    def test_remora_uses_core_relora_fields(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            {
                "adapter": "mora",
                "plugins": ["axolotl.integrations.mora.MoraPlugin"],
                "relora": True,
                "jagged_restart_steps": 2000,
                "mora": {
                    "use_mora": True,
                    "mora_type": "rope",
                },
            }
        )

        prepare_plugins(cfg)
        validated = validate_config(cfg)

        assert validated.relora is True
        assert validated.jagged_restart_steps == 2000

    def test_remora_still_requires_core_restart_steps(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            {
                "adapter": "mora",
                "plugins": ["axolotl.integrations.mora.MoraPlugin"],
                "relora": True,
                "mora": {
                    "use_mora": True,
                    "mora_type": "rope",
                },
            }
        )

        prepare_plugins(cfg)
        with pytest.raises(ValueError, match="jagged_restart_steps"):
            validate_config(cfg)
