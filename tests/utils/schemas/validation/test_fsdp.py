"""
tests for pydantic fsdp validation
"""

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


class TestFSDPValidation:
    """
    test class for pydantic fsdp validation
    """

    def test_fsdp_version_in_fsdp_config(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fsdp_config={
                "fsdp_version": 2,
            },
        )
        cfg = validate_config(
            cfg,
        )
        assert cfg.fsdp_version == 2
        assert cfg.fsdp_config.fsdp_version is None

    def test_fsdp_offload_w_8bit_optim(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fsdp_config={
                "offload_params": True,
            },
            optimizer="adamw_8bit",
            fsdp_version=1,
        )
        with pytest.raises(
            ValueError, match="FSDP Offload not compatible with adamw_8bit"
        ):
            validate_config(cfg)

    def test_fsdp2_w_8bit_optim(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fsdp_config={
                "offload_params": True,
            },
            optimizer="adamw_8bit",
            fsdp_version=2,
        )
        with pytest.raises(
            ValueError,
            match="FSDP2 not compatible with adamw_8bit, use `adamw_torch_8bit` instead",
        ):
            validate_config(cfg)

    def test_fsdp2_w_cpu_ram_efficient_loading(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            load_in_8bit=True,
            adapter="lora",
            fsdp_config={
                "cpu_ram_efficient_loading": True,
            },
            fsdp_version=2,
        )
        validated_cfg = validate_config(cfg)
        assert validated_cfg.fsdp_version == 2
        assert validated_cfg.fsdp_config.cpu_ram_efficient_loading is True

    def test_fsdp2_cpu_offload_pin_memory_requires_offload_params(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fsdp_config={
                "cpu_offload_pin_memory": False,
                "offload_params": False,
            },
            fsdp_version=2,
        )
        with pytest.raises(
            ValueError,
            match="disabling cpu_offload_pin_memory requires enabling offload_params",
        ):
            validate_config(cfg)

    def test_fsdp1_cpu_offload_pin_memory_not_supported(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fsdp_config={
                "cpu_offload_pin_memory": False,
                "offload_params": True,
            },
            fsdp_version=1,
        )
        with pytest.raises(
            ValueError,
            match="FSDP1 does not support disabling cpu_offload_pin_memory, please set `fsdp_version` to 2",
        ):
            validate_config(cfg)

    def test_fsdp2_cpu_offload_pin_memory_w_offload_params(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fsdp_config={
                "cpu_offload_pin_memory": False,
                "offload_params": True,
            },
            fsdp_version=2,
        )
        validated_cfg = validate_config(cfg)
        assert validated_cfg.fsdp_config.cpu_offload_pin_memory is False
        assert validated_cfg.fsdp_config.offload_params is True

    def test_fsdp_prefixes_removed(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fsdp_config={
                "fsdp_version": 2,
                "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
                "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
                "fsdp_reshard_after_forward": True,
            }
        )
        cfg = validate_config(cfg)
        assert cfg.fsdp_version == 2
        assert cfg.fsdp_config.fsdp_version is None
        for keys in cfg.fsdp_config.keys():
            assert not keys.startswith("fsdp_")
        assert cfg.fsdp_config.auto_wrap_policy == "TRANSFORMER_BASED_WRAP"
        assert cfg.fsdp_config.transformer_layer_cls_to_wrap == "LlamaDecoderLayer"
        assert cfg.fsdp_config.reshard_after_forward is True

    @pytest.mark.parametrize(
        "rl",
        [
            "dpo",
            "kto",
            "orpo",
            "ipo",
        ],
    )
    def test_fsdp2_dpo(self, min_base_cfg, rl):
        cfg = min_base_cfg | DictDefault(
            fsdp_version=2,
            fsdp_config={
                "reshard_after_forward": True,
            },
            rl=rl,
            load_in_8bit=True,
            adapter="lora",
            remove_unused_columns=False,
        )
        with pytest.raises(
            ValueError,
            match="FSDP2 does not support load_in_8bit or load_in_4bit with ",
        ):
            validate_config(cfg)
