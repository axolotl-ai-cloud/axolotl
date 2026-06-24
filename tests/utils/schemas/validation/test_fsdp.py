"""
tests for pydantic fsdp validation
"""

import logging

import pytest

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


class TestFSDPValidation:
    """
    test class for pydantic fsdp validation
    """

    def test_fsdp_version_from_fsdp_config(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fsdp_config={
                "version": 2,
            },
        )
        cfg = validate_config(
            cfg,
        )
        assert cfg.fsdp_version == 2

    def test_fsdp_version_in_fsdp_config(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fsdp_version=2,
            fsdp_config={
                "reshard_after_forward": True,
            },
        )
        cfg = validate_config(
            cfg,
        )
        assert cfg.fsdp_version == 2
        assert cfg.fsdp_config.fsdp_version == 2

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
        assert cfg.fsdp_config.fsdp_version == 2
        for key in cfg.fsdp_config.keys():
            if key != "fsdp_version":
                assert not key.startswith("fsdp_")
        assert cfg.fsdp_config.auto_wrap_policy == "TRANSFORMER_BASED_WRAP"
        assert cfg.fsdp_config.transformer_layer_cls_to_wrap == "LlamaDecoderLayer"
        assert cfg.fsdp_config.reshard_after_forward is True

    def test_fp32_norms_requires_fsdp_config(self, min_base_cfg):
        # fsdp_config is the canonical "is_fsdp" signal; fp32_norms requires it.
        cfg = min_base_cfg | DictDefault(
            fp32_norms=True,
            fsdp_version=2,
        )
        with pytest.raises(ValueError, match="fp32_norms requires FSDP to be enabled"):
            validate_config(cfg)

    def test_fp32_norms_requires_fsdp2(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fp32_norms=True,
            fsdp_version=1,
            fsdp_config={"reshard_after_forward": True},
        )
        with pytest.raises(ValueError, match="fp32_norms requires fsdp_version: 2"):
            validate_config(cfg)

    def test_fp32_norms_cpu_ram_efficient_loading_ok(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fp32_norms=True,
            fsdp_version=2,
            fsdp_config={
                "reshard_after_forward": True,
                "cpu_ram_efficient_loading": True,
            },
        )
        validated_cfg = validate_config(cfg)
        assert validated_cfg.fp32_norms is True
        assert validated_cfg.fsdp_config.cpu_ram_efficient_loading is True

    def test_fp32_norms_tensor_parallel_ok(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fp32_norms=True,
            fsdp_version=2,
            tensor_parallel_size=2,
            fsdp_config={"reshard_after_forward": True},
        )
        validated_cfg = validate_config(cfg)
        assert validated_cfg.fp32_norms is True
        assert validated_cfg.tensor_parallel_size == 2

    def test_fp32_norms_fsdp2_ok(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fp32_norms=True,
            fp32_norm_classes=["AfmoeRMSNorm"],
            fsdp_version=2,
            fsdp_config={"reshard_after_forward": True},
        )
        validated_cfg = validate_config(cfg)
        assert validated_cfg.fp32_norms is True
        assert validated_cfg.fp32_norm_classes == ["AfmoeRMSNorm"]

    def test_fp32_norm_classes_without_fp32_norms_warns(self, min_base_cfg, caplog):
        cfg = min_base_cfg | DictDefault(
            fp32_norm_classes=["AfmoeRMSNorm"],
        )
        # axolotl.cli.configure_logging() sets propagate=False on the `axolotl`
        # logger, so pytest caplog (attached to root) can't see records by
        # default. Temporarily re-enable propagation for this assertion.
        ax_logger = logging.getLogger("axolotl")
        old_propagate = ax_logger.propagate
        ax_logger.propagate = True
        try:
            with caplog.at_level("WARNING", logger="axolotl"):
                validated_cfg = validate_config(cfg)
        finally:
            ax_logger.propagate = old_propagate
        assert not validated_cfg.fp32_norms
        assert "fp32_norm_classes is set but fp32_norms is not enabled" in caplog.text

    def test_muon_fsdp1_rejected(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            optimizer="muon",
            fsdp_version=1,
            fsdp_config={"reshard_after_forward": True},
        )
        with pytest.raises(
            ValueError, match="Muon optimizer is only compatible with FSDP2"
        ):
            validate_config(cfg)

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

    def test_size_based_wrap_requires_min_num_params(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(
            fsdp_version=2,
            fsdp_config={
                "auto_wrap_policy": "SIZE_BASED_WRAP",
                "reshard_after_forward": True,
            },
        )
        with pytest.raises(
            ValueError,
            match="min_num_params is required when auto_wrap_policy is SIZE_BASED_WRAP",
        ):
            validate_config(cfg)
