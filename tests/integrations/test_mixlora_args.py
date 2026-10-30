"""Config validation for the MixLoRA plugin args."""

from collections import OrderedDict

import pytest

from axolotl.integrations.base import PluginManager
from axolotl.utils.config import prepare_plugins, validate_config
from axolotl.utils.dict import DictDefault

PLUGIN = "axolotl.integrations.mixlora.MixLoraPlugin"


@pytest.fixture(name="plugin_manager")
def plugin_manager_fixture():
    plugin_manager = PluginManager.get_instance()
    original_plugins = plugin_manager.plugins
    plugin_manager.plugins = OrderedDict()
    try:
        yield plugin_manager
    finally:
        plugin_manager.plugins = original_plugins


def _cfg(**overrides):
    return DictDefault(
        {
            "base_model": "HuggingFaceTB/SmolLM2-135M",
            "plugins": [PLUGIN],
            "adapter": "mixlora",
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.0,
            "lora_target_modules": ["q_proj", "v_proj"],
            "datasets": [{"path": "mhenrichsen/alpaca_2k_test", "type": "alpaca"}],
            "micro_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "learning_rate": 1e-4,
            **overrides,
        }
    )


def _validate(cfg):
    prepare_plugins(cfg)
    return validate_config(cfg)


class TestMixLoraArgs:
    def test_args_come_from_the_plugin(self, plugin_manager):
        cfg = _validate(_cfg(mixlora_num_experts=4, mixlora_top_k=2))
        assert cfg.mixlora_num_experts == 4
        assert cfg.mixlora_top_k == 2
        assert cfg.mixlora_router_aux_loss_coef == 0.01

    def test_core_schema_has_no_mixlora_fields(self):
        from axolotl.utils.schemas.config import AxolotlInputConfig

        assert not [
            field for field in AxolotlInputConfig.model_fields if "mixlora" in field
        ]

    def test_top_k_must_not_exceed_num_experts(self, plugin_manager):
        with pytest.raises(ValueError, match="mixlora_top_k"):
            _validate(_cfg(mixlora_num_experts=2, mixlora_top_k=4))

    def test_lora_r_required(self, plugin_manager):
        with pytest.raises(ValueError, match="lora_r is required"):
            _validate(_cfg(lora_r=None))

    def test_rejects_lora_target_linear(self, plugin_manager):
        with pytest.raises(ValueError, match="lora_target_linear"):
            _validate(_cfg(lora_target_linear=True, lora_target_modules=None))

    def test_rejects_ffn_lora_targets(self, plugin_manager):
        with pytest.raises(ValueError, match="gate_proj"):
            _validate(_cfg(lora_target_modules=["q_proj", "gate_proj"]))

    def test_rejects_fused_mlp(self, plugin_manager):
        with pytest.raises(ValueError, match="flash_attn_fuse_mlp"):
            _validate(_cfg(flash_attn_fuse_mlp=True))

    def test_rejects_rl(self, plugin_manager):
        with pytest.raises(ValueError, match="RL training"):
            _validate(_cfg(rl="dpo"))

    def test_forces_non_reentrant_gradient_checkpointing(self, plugin_manager):
        cfg = _validate(
            _cfg(
                gradient_checkpointing=True,
                gradient_checkpointing_kwargs={"use_reentrant": True},
            )
        )
        assert cfg.gradient_checkpointing_kwargs["use_reentrant"] is False


class TestMixLoraPluginAdapter:
    def test_load_adapter_ignores_other_adapters(self):
        from axolotl.integrations.mixlora import MixLoraPlugin

        assert (
            MixLoraPlugin().load_adapter(None, DictDefault({"adapter": "lora"})) is None
        )
