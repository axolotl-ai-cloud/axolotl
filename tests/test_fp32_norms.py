"""Unit tests for fp32 norm sharding (FSDP2). Pure-CPU, no dist init."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from axolotl.loaders.model import (
    DEFAULT_FP32_NORM_SUFFIXES,
    _matches_norm_class,
    shard_norms_fp32,
)


class LlamaRMSNorm(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class AfmoeRMSNorm(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class CustomNorm(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(8, 8)


def test_suffix_matches_multiple_norm_families():
    patterns = list(DEFAULT_FP32_NORM_SUFFIXES)
    assert _matches_norm_class(LlamaRMSNorm(), patterns)
    assert _matches_norm_class(AfmoeRMSNorm(), patterns)
    assert _matches_norm_class(nn.LayerNorm(8), patterns)


def test_suffix_does_not_match_non_norm_modules():
    patterns = list(DEFAULT_FP32_NORM_SUFFIXES)
    assert not _matches_norm_class(MLP(), patterns)
    assert not _matches_norm_class(nn.Linear(8, 8), patterns)
    assert not _matches_norm_class(CustomNorm(), patterns)


def test_explicit_classname_matches_custom_norm():
    assert _matches_norm_class(CustomNorm(), ["CustomNorm"])


def test_fully_qualified_pattern_matches_exact_path():
    qualified = f"{LlamaRMSNorm.__module__}.LlamaRMSNorm"
    assert _matches_norm_class(LlamaRMSNorm(), [qualified])
    assert not _matches_norm_class(AfmoeRMSNorm(), [qualified])


def test_mixed_patterns_suffix_and_qualified():
    qualified = f"{LlamaRMSNorm.__module__}.LlamaRMSNorm"
    patterns = [qualified, "LayerNorm"]
    assert _matches_norm_class(LlamaRMSNorm(), patterns)
    assert _matches_norm_class(nn.LayerNorm(8), patterns)
    assert not _matches_norm_class(AfmoeRMSNorm(), patterns)


class _Cfg:
    def __init__(self, **kwargs):
        self.fp32_norms = kwargs.get("fp32_norms", False)
        self.fp32_norm_classes = kwargs.get("fp32_norm_classes", None)
        self.fsdp_version = kwargs.get("fsdp_version", None)


def test_disabled_is_noop():
    model = nn.Sequential(LlamaRMSNorm(), MLP())
    assert shard_norms_fp32(model, _Cfg(fp32_norms=False)) == 0


def test_enabled_requires_fsdp2():
    model = nn.Sequential(LlamaRMSNorm())
    cfg = _Cfg(fp32_norms=True, fsdp_version=1)
    with pytest.raises(ValueError, match="fsdp_version: 2"):
        shard_norms_fp32(model, cfg)


def test_meta_device_is_rejected():
    with torch.device("meta"):
        model = nn.Sequential(LlamaRMSNorm())
    cfg = _Cfg(fp32_norms=True, fsdp_version=2)
    with pytest.raises(RuntimeError, match="meta-device"):
        shard_norms_fp32(model, cfg)


def test_no_matches_warns_and_returns_zero(caplog):
    model = nn.Sequential(MLP(), nn.Linear(8, 8))
    cfg = _Cfg(fp32_norms=True, fsdp_version=2)
    with caplog.at_level("WARNING", logger="axolotl.loaders.model"):
        n = shard_norms_fp32(model, cfg)
    assert n == 0
    assert "no modules matched" in caplog.text


def test_explicit_classes_override_defaults(monkeypatch):
    model = nn.Sequential(CustomNorm(), MLP())
    cfg = _Cfg(fp32_norms=True, fsdp_version=2, fp32_norm_classes=["CustomNorm"])

    import torch.distributed.fsdp as fsdp_module

    calls = []

    def fake_fully_shard(module, mp_policy=None, **_):
        calls.append((type(module).__name__, mp_policy))
        return module

    monkeypatch.setattr(fsdp_module, "fully_shard", fake_fully_shard)

    n = shard_norms_fp32(model, cfg)
    assert n == 1
    assert calls[0][0] == "CustomNorm"
    assert calls[0][1].param_dtype == torch.float32
    assert calls[0][1].reduce_dtype == torch.float32
