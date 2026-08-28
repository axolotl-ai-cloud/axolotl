"""Tests that multimodal processing strategies dispatch on model_type.

Architectures that share a processor class with another family (qwen4_exp, qwen3_5,
qwen3_5_moe and qwen3_vl_moe all use ``Qwen3VLProcessor``) cannot be told apart by the
processor matcher, so the registry has to be consulted by model_type first.
"""

from types import SimpleNamespace

import axolotl.model_support
from axolotl.model_support import get_model_support, resolve_model_support
from axolotl.processing_strategies import get_processing_strategy


class _SentinelStrategy:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


def _stub_processor():
    tokenizer = SimpleNamespace(
        chat_template="{{ x }}",
        convert_tokens_to_ids=lambda _tok: 0,
        pad_token_id=0,
    )
    return SimpleNamespace(tokenizer=tokenizer)


def test_qwen4_exp_descriptor_declares_a_processing_strategy():
    """The descriptor's provider must be reachable, not dead configuration."""
    resolved = resolve_model_support(get_model_support("qwen4_exp"))
    provider = resolved.strategies.processing_strategy_cls
    assert provider is not None
    assert provider().__name__ == "Qwen3_5ProcessingStrategy"


def test_model_type_is_preferred_over_chat_template(monkeypatch):
    seen = []

    def fake_get_model_support(key):
        seen.append(key)
        if key == "fake_arch":
            return SimpleNamespace(
                strategies=SimpleNamespace(
                    processing_strategy_cls=lambda: _SentinelStrategy
                )
            )
        return None

    monkeypatch.setattr(
        axolotl.model_support, "get_model_support", fake_get_model_support
    )
    monkeypatch.setattr(
        axolotl.model_support, "get_model_support_for_processor", lambda _p: None
    )
    monkeypatch.setattr(axolotl.model_support, "resolve_model_support", lambda s: s)

    strategy = get_processing_strategy(
        _stub_processor(), None, "tokenizer_default", model_type="fake_arch"
    )
    assert isinstance(strategy, _SentinelStrategy)
    assert seen[0] == "fake_arch", "model_type must be consulted before chat_template"


def test_absent_model_type_falls_back_to_chat_template(monkeypatch):
    def fake_get_model_support(key):
        if key == "some_template":
            return SimpleNamespace(
                strategies=SimpleNamespace(
                    processing_strategy_cls=lambda: _SentinelStrategy
                )
            )
        return None

    monkeypatch.setattr(
        axolotl.model_support, "get_model_support", fake_get_model_support
    )
    monkeypatch.setattr(
        axolotl.model_support, "get_model_support_for_processor", lambda _p: None
    )
    monkeypatch.setattr(axolotl.model_support, "resolve_model_support", lambda s: s)

    strategy = get_processing_strategy(
        _stub_processor(), None, "some_template", model_type=None
    )
    assert isinstance(strategy, _SentinelStrategy)
