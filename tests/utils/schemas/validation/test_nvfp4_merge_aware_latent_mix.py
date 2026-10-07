"""Config validation for nvfp4_merge_aware_latent_mix without the kernels plugin."""

from unittest.mock import patch

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


def test_latent_mix_accepted_on_core_lora_config(min_base_cfg):
    cfg = validate_config(
        DictDefault(adapter="lora", nvfp4_merge_aware_latent_mix=0.5) | min_base_cfg
    )
    assert cfg.nvfp4_merge_aware_latent_mix == 0.5


def test_latent_mix_unset_by_default(min_base_cfg):
    cfg = validate_config(DictDefault(adapter="lora") | min_base_cfg)
    assert cfg.nvfp4_merge_aware_latent_mix is None


@pytest.mark.parametrize("mix", [-0.5, 1.0, True])
def test_latent_mix_out_of_range_rejected(min_base_cfg, mix):
    with pytest.raises(ValueError, match="nvfp4_merge_aware_latent_mix"):
        validate_config(
            DictDefault(adapter="lora", nvfp4_merge_aware_latent_mix=mix) | min_base_cfg
        )


def test_latent_mix_ignored_when_merge_aware_disabled(min_base_cfg):
    with patch("axolotl.utils.schemas.peft.LOG.warning") as warning:
        cfg = validate_config(
            DictDefault(
                adapter="lora",
                nvfp4_merge_aware=False,
                nvfp4_merge_aware_latent_mix=0.5,
            )
            | min_base_cfg
        )
    assert cfg.nvfp4_merge_aware_latent_mix is None
    assert any(
        "nvfp4_merge_aware_latent_mix" in call.args[0]
        for call in warning.call_args_list
    )


def test_latent_mix_ignored_for_rl():
    from axolotl.utils.schemas.peft import validate_nvfp4_merge_aware_latent_mix

    with patch("axolotl.utils.schemas.peft.LOG.warning") as warning:
        data = validate_nvfp4_merge_aware_latent_mix(
            {"adapter": "lora", "rl": "dpo", "nvfp4_merge_aware_latent_mix": 0.5}
        )
    assert data["nvfp4_merge_aware_latent_mix"] is None
    assert "rl" in warning.call_args.args[0]
