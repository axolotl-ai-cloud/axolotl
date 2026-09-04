"""CPU tests for TorchAO FP8 recipe configuration."""

import pytest

pytest.importorskip("torchao")

from torchao.float8 import ScalingGranularity, ScalingType  # noqa: E402

from axolotl.core.fp8 import build_fp8_linear_config  # noqa: E402


def test_tensorwise_recipe_preserves_fsdp_all_gather_flags():
    config = build_fp8_linear_config(
        fp8_recipe="tensorwise", enable_fsdp_float8_all_gather=True
    )

    assert config.cast_config_weight.scaling_granularity is (
        ScalingGranularity.TENSORWISE
    )
    assert config.enable_fsdp_float8_all_gather is True
    assert config.force_recompute_fp8_weight_in_bwd is True


def test_rowwise_recipe_uses_axiswise_config():
    config = build_fp8_linear_config(fp8_recipe="rowwise")

    assert config.cast_config_weight.scaling_granularity is (
        ScalingGranularity.AXISWISE
    )
    assert config.cast_config_input.scaling_granularity is ScalingGranularity.AXISWISE
    assert config.cast_config_grad_output.scaling_granularity is (
        ScalingGranularity.AXISWISE
    )
    assert config.enable_fsdp_float8_all_gather is False


def test_rowwise_with_gw_hp_uses_disabled_grad_weight_casts():
    config = build_fp8_linear_config(fp8_recipe="rowwise_with_gw_hp")

    assert config.cast_config_weight.scaling_granularity is (
        ScalingGranularity.AXISWISE
    )
    assert config.cast_config_input_for_grad_weight.scaling_type is ScalingType.DISABLED
    assert (
        config.cast_config_grad_output_for_grad_weight.scaling_type
        is ScalingType.DISABLED
    )
    assert config.enable_fsdp_float8_all_gather is False


@pytest.mark.parametrize("recipe", ["rowwise", "rowwise_with_gw_hp"])
def test_rowwise_recipe_rejects_fsdp_all_gather(recipe):
    with pytest.raises(ValueError, match="only supports the tensorwise"):
        build_fp8_linear_config(
            fp8_recipe=recipe,
            enable_fsdp_float8_all_gather=True,
        )
