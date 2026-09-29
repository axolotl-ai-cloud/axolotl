"""One predicate decides LoRA kernel auto-enable and the validators that predict it."""

from types import SimpleNamespace

import pytest

from axolotl.model_support import Unsupported
from axolotl.utils.schemas.validation import (
    _lora_kernel_enabled,
    lora_kernels_auto_enable,
)

BASE = {"adapter": "lora", "env_capabilities": {"torch_version": "2.13.0"}}


def _get(**overrides):
    data = {**BASE, **overrides}
    return data.get


def _fake_support(monkeypatch, lora_kernels):
    import axolotl.model_support as model_support

    support = SimpleNamespace(capabilities={"lora_kernels": lora_kernels})
    monkeypatch.setattr(model_support, "get_model_support", lambda _t: support)
    monkeypatch.setattr(model_support, "resolve_model_support", lambda s: s)


def test_enables_plain_lora():
    assert lora_kernels_auto_enable(_get()) is True


@pytest.mark.parametrize(
    "overrides",
    [
        {"rl": "dpo"},
        {"nvfp4_merge_aware": True},
        {"adapter": None},
        {"lora_qkv_kernel": False},
        {"load_in_8bit": True},
        {"trust_remote_code": True},
    ],
)
def test_blocked_by_config(overrides):
    assert lora_kernels_auto_enable(_get(**overrides)) is False


def test_blocked_by_model_capability(monkeypatch):
    _fake_support(monkeypatch, Unsupported("no fused kernels"))
    assert lora_kernels_auto_enable(_get(model_config_type="x")) is False
    _fake_support(monkeypatch, None)
    assert lora_kernels_auto_enable(_get(model_config_type="x")) is True


def test_blocked_for_moe_without_grouped_mm():
    get = _get(
        model_config_type="qwen3_moe", env_capabilities={"torch_version": "2.8.0"}
    )
    assert lora_kernels_auto_enable(get) is False
    get = _get(
        model_config_type="qwen3_moe", env_capabilities={"torch_version": "2.9.0"}
    )
    assert lora_kernels_auto_enable(get) is True


def test_lora_kernel_enabled_follows_the_predicate(monkeypatch):
    cfg = SimpleNamespace(**BASE, lora_mlp_kernel=None, model_config_type="x")
    _fake_support(monkeypatch, Unsupported("no fused kernels"))
    assert _lora_kernel_enabled(cfg, "lora_mlp_kernel") is False
    _fake_support(monkeypatch, None)
    assert _lora_kernel_enabled(cfg, "lora_mlp_kernel") is True
    cfg.lora_mlp_kernel = True
    _fake_support(monkeypatch, Unsupported("no fused kernels"))
    assert _lora_kernel_enabled(cfg, "lora_mlp_kernel") is True
