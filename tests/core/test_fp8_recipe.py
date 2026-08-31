"""CPU tests for TorchAO FP8 recipe configuration."""

import sys
from dataclasses import dataclass
from enum import Enum
from types import ModuleType
from typing import ClassVar

import pytest

from axolotl.core.fp8 import build_fp8_linear_config


class FakeScalingGranularity(Enum):
    TENSORWISE = "tensorwise"
    AXISWISE = "axiswise"


@dataclass(frozen=True)
class FakeCastConfig:
    scaling_granularity: FakeScalingGranularity


@dataclass(frozen=True)
class FakeFloat8LinearConfig:
    cast_config_weight: FakeCastConfig = FakeCastConfig(
        FakeScalingGranularity.TENSORWISE
    )
    enable_fsdp_float8_all_gather: bool = False
    force_recompute_fp8_weight_in_bwd: bool = False
    grad_weight_in_high_precision: bool = False

    recipe_calls: ClassVar[list[str]] = []

    @classmethod
    def from_recipe_name(cls, recipe: str):
        cls.recipe_calls.append(recipe)
        if recipe == "tensorwise":
            return cls()
        if recipe == "rowwise":
            return cls(
                cast_config_weight=FakeCastConfig(FakeScalingGranularity.AXISWISE)
            )
        if recipe == "rowwise_with_gw_hp":
            return cls(
                cast_config_weight=FakeCastConfig(FakeScalingGranularity.AXISWISE),
                grad_weight_in_high_precision=True,
            )
        raise AssertionError(f"unexpected recipe {recipe!r}")


@pytest.fixture
def fake_fp8_modules(monkeypatch):
    FakeFloat8LinearConfig.recipe_calls.clear()

    torchao = ModuleType("torchao")
    torchao.__path__ = []
    torchao_float8 = ModuleType("torchao.float8")
    torchao_float8.Float8LinearConfig = FakeFloat8LinearConfig
    torchao.float8 = torchao_float8

    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setitem(sys.modules, "torchao.float8", torchao_float8)


def test_tensorwise_recipe_preserves_fsdp_all_gather_flags(fake_fp8_modules):
    config = build_fp8_linear_config(
        fp8_recipe="tensorwise", enable_fsdp_float8_all_gather=True
    )

    assert config.cast_config_weight.scaling_granularity is (
        FakeScalingGranularity.TENSORWISE
    )
    assert config.enable_fsdp_float8_all_gather is True
    assert config.force_recompute_fp8_weight_in_bwd is True
    assert FakeFloat8LinearConfig.recipe_calls == ["tensorwise"]


def test_rowwise_recipe_uses_axiswise_config(fake_fp8_modules):
    config = build_fp8_linear_config(fp8_recipe="rowwise")

    assert config.cast_config_weight.scaling_granularity is (
        FakeScalingGranularity.AXISWISE
    )
    assert config.grad_weight_in_high_precision is False
    assert config.enable_fsdp_float8_all_gather is False
    assert FakeFloat8LinearConfig.recipe_calls == ["rowwise"]


def test_rowwise_with_gw_hp_recipe_keeps_grad_weight_high_precision(
    fake_fp8_modules,
):
    config = build_fp8_linear_config(fp8_recipe="rowwise_with_gw_hp")

    assert config.cast_config_weight.scaling_granularity is (
        FakeScalingGranularity.AXISWISE
    )
    assert config.grad_weight_in_high_precision is True
    assert config.enable_fsdp_float8_all_gather is False
    assert FakeFloat8LinearConfig.recipe_calls == ["rowwise_with_gw_hp"]


@pytest.mark.parametrize("recipe", ["rowwise", "rowwise_with_gw_hp"])
def test_rowwise_recipe_rejects_fsdp_all_gather(fake_fp8_modules, recipe):
    with pytest.raises(ValueError, match="only supports the tensorwise"):
        build_fp8_linear_config(
            fp8_recipe=recipe,
            enable_fsdp_float8_all_gather=True,
        )

    assert FakeFloat8LinearConfig.recipe_calls == []
