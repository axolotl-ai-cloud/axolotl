"""FP8 mixed-precision configuration schema."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

FP8Recipe = Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"]
DEFAULT_FP8_RECIPE: FP8Recipe = "tensorwise"


class FP8Config(BaseModel):
    """Configuration for TorchAO FP8 scaling."""

    # `fp8_enable_fsdp_float8_all_gather` is a sibling top-level key, so a nested
    # spelling of it must fail loudly rather than be dropped.
    model_config = ConfigDict(extra="forbid")

    recipe: FP8Recipe = Field(
        default=DEFAULT_FP8_RECIPE,
        description=(
            "TorchAO FP8 scaling recipe: 'tensorwise' (default), 'rowwise', "
            "or 'rowwise_with_gw_hp'."
        ),
    )


def resolve_fp8_recipe(fp8_config: Any) -> str:
    """Read the recipe from a raw config dict, an ``FP8Config``, or ``None``."""
    if isinstance(fp8_config, dict):
        return fp8_config.get("recipe", DEFAULT_FP8_RECIPE)
    return getattr(fp8_config, "recipe", DEFAULT_FP8_RECIPE)
