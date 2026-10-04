from types import SimpleNamespace
from unittest.mock import Mock

from axolotl.loaders.processor import load_processor
from axolotl.model_support.profile import ModelStrategies, ModelStrategyOverrides
from axolotl.utils.dict import DictDefault


def test_processor_provider_inheritance_override_and_removal():
    provider = lambda: object
    parent = ModelStrategies(auto_processor_cls=provider)
    assert (
        parent.with_overrides(ModelStrategyOverrides()).auto_processor_cls is provider
    )
    assert (
        parent.with_overrides(
            ModelStrategyOverrides(auto_processor_cls=None)
        ).auto_processor_cls
        is None
    )


def test_loader_uses_model_processor_and_preserves_explicit_override(monkeypatch):
    import axolotl.loaders.processor as module

    custom = Mock()
    generic = Mock()
    monkeypatch.setattr(module, "AutoProcessor", generic)
    monkeypatch.setattr(module.transformers, "AutoProcessor", generic)
    monkeypatch.setattr(module, "get_model_support_for_cfg", lambda cfg: object())
    monkeypatch.setattr(
        module,
        "resolve_model_support",
        lambda support: SimpleNamespace(
            strategies=ModelStrategies(auto_processor_cls=lambda: custom)
        ),
    )
    cfg = DictDefault(
        processor_config="model-source",
        revision_of_model="revision",
        trust_remote_code=True,
        image_size=256,
    )
    tokenizer = object()
    load_processor(cfg, tokenizer)
    custom.from_pretrained.assert_called_once_with(
        "model-source", revision="revision", trust_remote_code=True, tokenizer=tokenizer
    )
    assert custom.from_pretrained.return_value.tokenizer is tokenizer
    cfg.processor_type = "AutoProcessor"
    load_processor(cfg, tokenizer)
    generic.from_pretrained.assert_called_once_with(
        "model-source", revision="revision", trust_remote_code=True
    )
