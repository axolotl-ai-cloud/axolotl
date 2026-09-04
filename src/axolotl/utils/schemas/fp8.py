"""FP8 mixed-precision configuration schema."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class FP8Config(BaseModel):
    """Configuration for TorchAO FP8 scaling."""

    # `fp8_enable_fsdp_float8_all_gather` is a sibling top-level key, so a nested
    # spelling of it must fail loudly rather than be dropped.
    model_config = ConfigDict(extra="forbid")

    recipe: Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"] = Field(
        default="tensorwise",
        description=(
            "TorchAO FP8 scaling recipe: 'tensorwise' (default), 'rowwise', "
            "or 'rowwise_with_gw_hp'."
        ),
    )
