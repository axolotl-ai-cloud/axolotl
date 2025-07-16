"""Test for config validation for activation offloading."""

from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


class TestActivationOffloading:
    """
    Test cases for activation offloading schema validation
    """

    def test_gc_converts_offload_wo_lora(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing="offload",
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing is True
        assert cfg.activation_offloading is True

    def test_gc_converts_offload_w_lora(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing="offload",
                adapter="lora",
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing is True
        assert cfg.activation_offloading == "legacy"

    def test_gc_converts_offload_w_qlora(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing="offload",
                adapter="qlora",
                load_in_4bit=True,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing is True
        assert cfg.activation_offloading == "legacy"

    def test_ac_impl_changes_w_lora(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                activation_offloading=True,
                adapter="lora",
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing is True
        assert cfg.activation_offloading == "legacy"

    def test_ac_impl_changes_w_qlora(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                activation_offloading=True,
                adapter="qlora",
                load_in_4bit=True,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing is True
        assert cfg.activation_offloading == "legacy"

    def test_ac_offload_impl_noop_wo_adapter(self, min_base_cfg):
        cfg = (
            DictDefault(
                gradient_checkpointing=True,
                activation_offloading=True,
            )
            | min_base_cfg
        )

        cfg = validate_config(cfg)
        assert cfg.gradient_checkpointing is True
        assert cfg.activation_offloading is True
