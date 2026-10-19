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
        with pytest.raises(ValueError, match="fsdp_version: 1 is no longer supported"):
            validate_config(cfg)

    def test_fsdp_version_defaults_to_2(self, min_base_cfg):
        cfg = validate_config(
            min_base_cfg | DictDefault(fsdp_config={"reshard_after_forward": True})
        )
        assert cfg.fsdp_version == 2
        assert cfg.fsdp_config.fsdp_version == 2

    def test_fsdp_version_defaults_to_2_without_fsdp_config(self, min_base_cfg):
        assert validate_config(min_base_cfg).fsdp_version == 2

    @pytest.mark.parametrize(
        "overrides",
        [
            {"fsdp_version": 1},
            {"fsdp_config": {"fsdp_version": 1}},
            {"fsdp_config": {"version": 1}},
            {"fsdp_config": {"fsdp_version": 1, "fsdp_offload_params": True}},
        ],
    )
    def test_fsdp1_rejected(self, min_base_cfg, overrides):
        cfg = min_base_cfg | DictDefault(overrides)
        with pytest.raises(ValueError, match="fsdp_version: 1 is no longer supported"):
            validate_config(cfg)

    @pytest.mark.parametrize("key", ["sharding_strategy", "fsdp_sharding_strategy"])
    def test_fsdp1_only_key_rejected_with_fsdp2_equivalent(self, min_base_cfg, key):
        cfg = min_base_cfg | DictDefault(fsdp_config={key: "FULL_SHARD"})
        with pytest.raises(
            ValueError,
            match="fsdp_config.sharding_strategy .*Use `reshard_after_forward` instead",
        ):
            validate_config(cfg)

    def test_fsdp1_only_key_without_equivalent_rejected(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(fsdp_config={"forward_prefetch": True})
        with pytest.raises(
            ValueError, match="fsdp_config.forward_prefetch .*no FSDP2 equivalent"
        ):
            validate_config(cfg)

    @pytest.mark.parametrize(
        "key",
        [
            "sync_module_states",
            "fsdp_sync_module_states",
            "backward_prefetch",
            "backward_prefetch_policy",
            "limit_all_gathers",
            "use_orig_params",
            "fsdp_use_orig_params",
        ],
    )
    def test_fsdp1_only_ignored_key_warns_and_is_dropped(
        self, min_base_cfg, caplog, key
    ):
        cfg = min_base_cfg | DictDefault(
            fsdp_config={key: True, "reshard_after_forward": True}
        )
        ax_logger = logging.getLogger("axolotl")
        old_propagate = ax_logger.propagate
        ax_logger.propagate = True
        try:
            with caplog.at_level("WARNING", logger="axolotl"):
                validated_cfg = validate_config(cfg)
        finally:
            ax_logger.propagate = old_propagate
        name = key.removeprefix("fsdp_")
        assert (
            f"fsdp_config.{name} is an FSDP1-only option and is ignored under FSDP2"
            in caplog.text
        )
        assert key not in validated_cfg.fsdp_config
        assert name not in validated_cfg.fsdp_config
        assert validated_cfg.fsdp_config.reshard_after_forward is True

    def test_bare_fsdp_list_rejected(self, min_base_cfg):
        cfg = min_base_cfg | DictDefault(fsdp=["full_shard", "auto_wrap"])
        with pytest.raises(
            ValueError, match="`fsdp` list is no longer supported.*multi-gpu"
        ):
            validate_config(cfg)

    @pytest.mark.parametrize("value", [None, []])
    def test_blank_fsdp_key_accepted(self, min_base_cfg, value):
        cfg = min_base_cfg | DictDefault(
            fsdp=value, fsdp_config={"reshard_after_forward": True}
        )
        validated_cfg = validate_config(cfg)
        assert validated_cfg.fsdp_version == 2

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
