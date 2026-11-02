"""Core adapter plugin registry tests."""

from unittest.mock import Mock

import pytest
import torch

from axolotl.integrations.base import AdapterCapabilities, BasePlugin, PluginManager
from axolotl.loaders import adapter as adapter_module
from axolotl.loaders.adapter import load_adapter
from axolotl.utils.config import validate_config
from axolotl.utils.dict import DictDefault


class FakeAdapterPlugin(BasePlugin):
    def get_adapter_capabilities(self) -> list[AdapterCapabilities]:
        return [AdapterCapabilities(name="fake-adapter", lora_like=True, relora=True)]

    def get_lora_config_kwargs(self, cfg: DictDefault) -> dict:
        if cfg.adapter != "fake-adapter":
            return {}
        return {"fake_kwarg": "from-plugin"}


class TestAdapterPluginRegistry:
    def test_lora_like_plugin_adapter_contributes_peft_kwargs(self, monkeypatch):
        model = torch.nn.Linear(4, 4)
        cfg = DictDefault(
            {
                "adapter": "fake-adapter",
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
            }
        )
        PluginManager.get_instance().plugins["fake"] = FakeAdapterPlugin()
        captured = {}

        class FakeLoraConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.__dict__.update(kwargs)

        monkeypatch.setattr(adapter_module, "LoraConfig", FakeLoraConfig)
        monkeypatch.setattr(adapter_module, "get_peft_model", Mock())

        _, config = load_adapter(model, cfg, "fake-adapter", config_only=True)

        assert config is not None
        assert captured["fake_kwarg"] == "from-plugin"
        assert captured["task_type"].name == "CAUSAL_LM"

    def test_unknown_adapter_error_mentions_plugin_registry(self):
        model = torch.nn.Linear(4, 4)
        cfg = DictDefault({"adapter": "missing-adapter"})

        with pytest.raises(NotImplementedError, match="registered by a plugin"):
            load_adapter(model, cfg, "missing-adapter")

    def test_relora_accepts_plugin_adapter_capability(self, min_base_cfg):
        PluginManager.get_instance().plugins["fake"] = FakeAdapterPlugin()
        cfg = min_base_cfg | DictDefault(
            {
                "adapter": "fake-adapter",
                "relora": True,
                "jagged_restart_steps": 100,
            }
        )

        validated = validate_config(cfg)

        assert validated.adapter == "fake-adapter"
        assert validated.relora is True

    @pytest.mark.parametrize(
        "adapter,lora_like,supports_merge",
        [
            ("lora", True, True),
            ("qlora", True, True),
            ("mora", True, True),
            ("mixlora", True, False),
            ("llama-adapter", False, True),
            (None, False, True),
        ],
    )
    def test_adapter_capability_queries(
        self, monkeypatch, adapter, lora_like, supports_merge
    ):
        from axolotl.integrations.mixlora import MixLoraPlugin
        from axolotl.integrations.mora import MoraPlugin

        manager = PluginManager.get_instance()
        monkeypatch.setitem(manager.plugins, "mora", MoraPlugin())
        monkeypatch.setitem(manager.plugins, "mixlora", MixLoraPlugin())

        assert manager.is_lora_like(adapter) is lora_like
        assert manager.adapter_supports_merge(adapter) is supports_merge
