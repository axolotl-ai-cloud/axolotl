"""Test for config validation for EBFT."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from axolotl.prompt_strategies.ebft import load as load_ebft
from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault

EBFT_EXAMPLES_DIR = Path(__file__).parents[4] / "examples" / "ebft"


class TestEBFTValidation:
    """
    Test cases for ebft schema validation
    """

    @pytest.mark.parametrize("mode", ["structured", "strided"])
    def test_ebft_config_validates(self, min_base_cfg, mode):
        cfg = min_base_cfg | DictDefault(rl="ebft", ebft={"mode": mode})

        cfg = validate_config(cfg)
        assert cfg.ebft.mode == mode

    @pytest.mark.parametrize("ebft", [None, {}])
    def test_ebft_section_required(self, min_base_cfg, ebft):
        cfg = min_base_cfg | DictDefault(rl="ebft", ebft=ebft)

        with pytest.raises(ValidationError, match="`ebft` config section is required"):
            validate_config(cfg)

    def test_strided_flex_attention_forces_reentrant(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            rl="ebft",
            ebft={"mode": "strided"},
            attn_implementation="flex_attention",
            gradient_checkpointing=True,
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing_kwargs["use_reentrant"] is True

    def test_strided_flex_attention_rejects_activation_offloading(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            rl="ebft",
            ebft={"mode": "strided"},
            attn_implementation="flex_attention",
            gradient_checkpointing=True,
            activation_offloading=True,
        )

        with pytest.raises(ValidationError, match="incompatible with"):
            validate_config(cfg)

    @pytest.mark.parametrize(
        "config_path",
        sorted(EBFT_EXAMPLES_DIR.glob("*.yaml")),
        ids=lambda path: path.name,
    )
    def test_example_configs_validate(self, config_path):
        validate_config(DictDefault(yaml.safe_load(config_path.read_text())))

    @pytest.mark.parametrize(
        "config_path",
        sorted(EBFT_EXAMPLES_DIR.glob("*.yaml")),
        ids=lambda path: path.name,
    )
    def test_example_dataset_types_resolve(self, config_path):
        cfg = DictDefault(yaml.safe_load(config_path.read_text()))

        for dataset in cfg.datasets:
            assert load_ebft(dataset["type"], cfg, dataset_idx=0) is not None, (
                f"{dataset['type']} does not resolve to a packaged ebft strategy"
            )
