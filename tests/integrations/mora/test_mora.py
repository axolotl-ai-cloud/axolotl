"""Integration tests for the MoRA / ReMoRA adapter path."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from axolotl.integrations.base import PluginManager
from axolotl.integrations.mora import plugin as mora_plugin
from axolotl.loaders import adapter as adapter_module
from axolotl.loaders.adapter import load_adapter
from axolotl.utils.dict import DictDefault


class TestMoraAdapterLoading:
    """MoRA adapter selection and config wiring."""

    def test_load_adapter_uses_plugin_lora_like_registration(self, monkeypatch):
        model = torch.nn.Linear(4, 4)
        cfg = DictDefault(
            {
                "adapter": "mora",
                "mora": {"use_mora": True, "mora_type": "rope"},
            }
        )

        PluginManager.get_instance().plugins["axolotl.integrations.mora.MoraPlugin"] = (
            mora_plugin.MoraPlugin()
        )

        calls = []

        def fake_load_lora(*args, **kwargs):
            calls.append((args, kwargs))
            return args[0], "adapter-config"

        monkeypatch.setattr(adapter_module, "load_lora", fake_load_lora)

        _, config = load_adapter(model, cfg, "mora")

        assert config == "adapter-config"
        assert calls[0][1]["config_only"] is False

    def test_mora_plugin_raises_when_peft_missing_support(self):
        model = torch.nn.Linear(4, 4)
        cfg = DictDefault(
            {
                "adapter": "mora",
                "mora": {"use_mora": True, "mora_type": "rope"},
            }
        )
        PluginManager.get_instance().plugins["axolotl.integrations.mora.MoraPlugin"] = (
            mora_plugin.MoraPlugin()
        )

        with pytest.raises(ImportError, match="MoRA support"):
            load_adapter(model, cfg, "mora", config_only=True)

    def test_mora_plugin_rejects_quantized_base_model(self):
        model = torch.nn.Linear(4, 4)
        cfg = DictDefault(
            {
                "adapter": "mora",
                "load_in_4bit": True,
                "mora": {"use_mora": True, "mora_type": "rope"},
            }
        )
        PluginManager.get_instance().plugins["axolotl.integrations.mora.MoraPlugin"] = (
            mora_plugin.MoraPlugin()
        )

        with pytest.raises(ValueError, match="full-precision base model"):
            load_adapter(model, cfg, "mora", config_only=True)

    def test_mora_plugin_builds_mora_config_when_supported(self, monkeypatch):
        model = torch.nn.Linear(4, 4)
        cfg = DictDefault(
            {
                "adapter": "mora",
                "mora": {
                    "use_mora": True,
                    "mora_type": "rope",
                },
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
            }
        )

        captured = {}

        class FakeLoraConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.__dict__.update(kwargs)

        fake_model = SimpleNamespace(print_trainable_parameters=Mock())
        PluginManager.get_instance().plugins["axolotl.integrations.mora.MoraPlugin"] = (
            mora_plugin.MoraPlugin()
        )
        monkeypatch.setattr(mora_plugin, "_peft_supports_mora", lambda: True)
        monkeypatch.setattr(adapter_module, "LoraConfig", FakeLoraConfig)
        monkeypatch.setattr(
            adapter_module, "get_peft_model", Mock(return_value=fake_model)
        )

        _, config = load_adapter(model, cfg, "mora", config_only=True)

        assert captured["use_mora"] is True
        assert captured["mora_type"] == 6
        assert captured["task_type"].name == "CAUSAL_LM"
        assert config is not None
        assert config.use_mora is True
        assert config.mora_type == 6

    def test_mora_plugin_uses_lora_model_dir_resume_path(self, monkeypatch):
        model = torch.nn.Linear(4, 4)
        cfg = DictDefault(
            {
                "adapter": "mora",
                "mora": {"use_mora": True, "mora_type": "rope"},
                "lora_model_dir": "adapter-checkpoint",
                "lora_on_cpu": False,
                "lora_r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
            }
        )

        class FakeLoraConfig:
            def __init__(self, **kwargs):
                self.__dict__.update(kwargs)

        class FakePeftModel:
            def print_trainable_parameters(self):
                pass

            def named_parameters(self):
                return []

        from_pretrained = Mock(return_value=FakePeftModel())
        PluginManager.get_instance().plugins["axolotl.integrations.mora.MoraPlugin"] = (
            mora_plugin.MoraPlugin()
        )
        monkeypatch.setattr(mora_plugin, "_peft_supports_mora", lambda: True)
        monkeypatch.setattr(adapter_module, "LoraConfig", FakeLoraConfig)
        monkeypatch.setattr(
            adapter_module.PeftModel,
            "from_pretrained",
            from_pretrained,
        )

        peft_model, config = load_adapter(model, cfg, "mora")

        assert isinstance(peft_model, FakePeftModel)
        assert config.use_mora is True
        from_pretrained.assert_called_once()
        assert from_pretrained.call_args.args[:2] == (model, "adapter-checkpoint")
        assert from_pretrained.call_args.kwargs["is_trainable"] is True
