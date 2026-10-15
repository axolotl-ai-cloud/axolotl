"""Helpers for configuring TorchAO FP8 training recipes.

Recipe semantics: https://github.com/pytorch/ao/tree/main/torchao/float8
"""

from dataclasses import replace
from typing import TYPE_CHECKING

from axolotl.utils.schemas.fp8 import DEFAULT_FP8_RECIPE

if TYPE_CHECKING:
    from torchao.float8 import Float8LinearConfig


def build_fp8_linear_config(
    fp8_recipe: str = DEFAULT_FP8_RECIPE,
    enable_fsdp_float8_all_gather: bool = False,
) -> "Float8LinearConfig":
    """Build a TorchAO float8 config while preserving tensorwise FSDP behavior."""
    from torchao.float8 import Float8LinearConfig

    if enable_fsdp_float8_all_gather and fp8_recipe != DEFAULT_FP8_RECIPE:
        raise ValueError(
            "`fp8_enable_fsdp_float8_all_gather` only supports the "
            "tensorwise `fp8_config.recipe`; disable it when using rowwise scaling."
        )

    config = Float8LinearConfig.from_recipe_name(fp8_recipe)
    if fp8_recipe == DEFAULT_FP8_RECIPE:
        config = replace(
            config,
            enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
            force_recompute_fp8_weight_in_bwd=(enable_fsdp_float8_all_gather is True),
        )
    return config
