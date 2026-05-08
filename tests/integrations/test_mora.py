"""Integration tests for the MoRA / ReMoRA adapter path."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from axolotl.loaders import adapter as adapter_module
from axolotl.loaders.adapter import load_adapter, load_mora
from axolotl.utils.dict import DictDefault


class TestMoraAdapterLoading:
    """MoRA adapter selection and config wiring."""

    def test_load_adapter_dispatches_to_mora(self, monkeypatch):
        model = torch.nn.Linear(4, 4)
        cfg = DictDefault(
            {
                "adapter": "mora",
                "mora": {"use_mora": True, "mora_type": 6},
            }
        )

        calls = []

        def fake_load_lora(*args, **kwargs):
            calls.append("lora")
            return args[0], "lora-config"

        def fake_load_mora(*args, **kwargs):
            calls.append("mora")
            return args[0], "mora-config"

        monkeypatch.setattr(adapter_module, "load_lora", fake_load_lora)
        monkeypatch.setattr(adapter_module, "load_mora", fake_load_mora)

        _, config = load_adapter(model, cfg, "mora")
        assert config == "mora-config"
        assert calls == ["mora"]

        _, config = load_adapter(model, cfg, "lora")
        assert config == "lora-config"
        assert calls == ["mora", "lora"]

    def test_load_mora_raises_when_peft_missing_support(self):
        model = torch.nn.Linear(4, 4)
        cfg = DictDefault(
            {
                "adapter": "mora",
                "mora": {"use_mora": True, "mora_type": 6},
            }
        )

        with pytest.raises(ImportError, match="MoRA support"):
            load_mora(model, cfg, config_only=True)

    def test_load_mora_builds_mora_config_when_supported(self, monkeypatch):
        model = torch.nn.Linear(4, 4)
        cfg = DictDefault(
            {
                "adapter": "mora",
                "mora": {
                    "use_mora": True,
                    "mora_type": 6,
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
        monkeypatch.setattr(adapter_module, "_peft_supports_mora", lambda: True)
        monkeypatch.setattr(adapter_module, "LoraConfig", FakeLoraConfig)
        monkeypatch.setattr(adapter_module, "get_peft_model", Mock(return_value=fake_model))

        _, config = load_mora(model, cfg, config_only=True)

        assert captured["use_mora"] is True
        assert captured["mora_type"] == 6
        assert captured["task_type"].name == "CAUSAL_LM"
        assert config is not None
        assert config.use_mora is True
        assert config.mora_type == 6
