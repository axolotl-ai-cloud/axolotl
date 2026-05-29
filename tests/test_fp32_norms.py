"""Unit tests for fp32 norm sharding (FSDP2). Pure-CPU, no dist init."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy

from axolotl.loaders.model import ModelLoader
from axolotl.utils.dict import DictDefault
from axolotl.utils.fp32_norms import (
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


class CustomNormWithBuffer(nn.Module):
    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.bfloat16))
        self.register_buffer("running_scale", torch.ones(dim, dtype=torch.float16))


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
        self.fsdp_config = kwargs.get("fsdp_config", None)
        self.tensor_parallel_size = kwargs.get("tensor_parallel_size", 1)
        self.lora_on_cpu = kwargs.get("lora_on_cpu", False)


def test_disabled_is_noop():
    model = nn.Sequential(LlamaRMSNorm(), MLP())
    assert shard_norms_fp32(model, _Cfg(fp32_norms=False)) == 0


def test_enabled_requires_fsdp2():
    model = nn.Sequential(LlamaRMSNorm())
    cfg = _Cfg(fp32_norms=True, fsdp_version=1)
    with pytest.raises(ValueError, match="fsdp_version: 2"):
        shard_norms_fp32(model, cfg)


def test_meta_device_is_supported(monkeypatch):
    with torch.device("meta"):
        model = nn.Sequential(LlamaRMSNorm())
    cfg = _Cfg(fp32_norms=True, fsdp_version=2)

    import torch.distributed.fsdp as fsdp_module

    calls = []

    def fake_fully_shard(module, mp_policy=None, **kwargs):
        calls.append((type(module).__name__, mp_policy, kwargs))
        return module

    monkeypatch.setattr(fsdp_module, "fully_shard", fake_fully_shard)

    n = shard_norms_fp32(model, cfg)
    assert n == 1
    assert calls[0][0] == "LlamaRMSNorm"
    assert calls[0][1].param_dtype == torch.float32


def test_passthrough_fully_shard_kwargs_are_used(monkeypatch):
    model = nn.Sequential(LlamaRMSNorm())
    cfg = _Cfg(fp32_norms=True, fsdp_version=2)

    import torch.distributed.fsdp as fsdp_module

    calls = []

    def fake_fully_shard(module, mp_policy=None, **kwargs):
        calls.append((module, mp_policy, kwargs))
        return module

    monkeypatch.setattr(fsdp_module, "fully_shard", fake_fully_shard)

    sentinel_mesh = object()
    outer_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    n = shard_norms_fp32(
        model,
        cfg,
        fully_shard_kwargs={
            "mesh": sentinel_mesh,
            "reshard_after_forward": True,
            "mp_policy": outer_policy,
        },
    )
    assert n == 1
    assert calls[0][2]["mesh"] is sentinel_mesh
    assert calls[0][2]["reshard_after_forward"] is True
    assert calls[0][1].output_dtype == torch.bfloat16


def test_no_matches_warns_and_returns_zero(caplog):
    model = nn.Sequential(MLP(), nn.Linear(8, 8))
    cfg = _Cfg(fp32_norms=True, fsdp_version=2)
    # axolotl.cli.configure_logging() sets propagate=False on the `axolotl`
    # logger, so pytest caplog can't see records by default. Temporarily
    # re-enable propagation for this assertion.
    ax_logger = logging.getLogger("axolotl")
    old_propagate = ax_logger.propagate
    ax_logger.propagate = True
    try:
        with caplog.at_level("WARNING", logger="axolotl"):
            n = shard_norms_fp32(model, cfg)
    finally:
        ax_logger.propagate = old_propagate
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


def test_matched_norm_storage_is_cast_to_fp32_before_sharding(monkeypatch):
    model = nn.Sequential(CustomNormWithBuffer(), MLP())
    cfg = _Cfg(
        fp32_norms=True,
        fsdp_version=2,
        fp32_norm_classes=["CustomNormWithBuffer"],
    )

    import torch.distributed.fsdp as fsdp_module

    seen = []

    def fake_fully_shard(module, mp_policy=None, **kwargs):
        seen.append(
            (
                module.weight.dtype,
                module.running_scale.dtype,
                mp_policy.output_dtype,
                kwargs,
            )
        )
        return module

    monkeypatch.setattr(fsdp_module, "fully_shard", fake_fully_shard)

    outer_policy = MixedPrecisionPolicy(param_dtype=torch.float16)
    n = shard_norms_fp32(
        model,
        cfg,
        fully_shard_kwargs={"mp_policy": outer_policy},
    )

    assert n == 1
    assert model[0].weight.dtype == torch.float32
    assert model[0].running_scale.dtype == torch.float32
    assert seen[0][0] == torch.float32
    assert seen[0][1] == torch.float32
    assert seen[0][2] == torch.float16


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(8, 8)
        self.input_norm = CustomNorm()
        self.fc = nn.Linear(8, 8)


def test_convert_embedding_modules_dtype_keeps_fp32_norm_matches():
    loader = ModelLoader.__new__(ModelLoader)
    loader.cfg = DictDefault(
        fp32_norms=True,
        fp32_norm_classes=["CustomNorm"],
        lora_on_cpu=False,
    )
    loader.model = TinyModel()
    loader.model_config = SimpleNamespace(model_type="llama")

    loader._convert_embedding_modules_dtype(
        embedding_modules=["embed_tokens"],
        dist_dtype=torch.bfloat16,
        before_kbit_train_or_finetune=False,
    )

    assert loader.model.input_norm.weight.dtype == torch.float32
    assert loader.model.embed_tokens.weight.dtype == torch.bfloat16
    assert loader.model.fc.weight.dtype == torch.float32
