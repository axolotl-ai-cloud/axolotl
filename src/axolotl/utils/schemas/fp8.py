"""FP8 mixed-precision configuration schema."""

from typing import Literal

from pydantic import BaseModel, Field


class FP8Config(BaseModel):
    """Configuration for TorchAO FP8 scaling."""

    recipe: Literal["tensorwise", "rowwise", "rowwise_with_gw_hp"] = Field(
        default="tensorwise",
        description=(
            "TorchAO FP8 scaling recipe: 'tensorwise' (default), 'rowwise', "
            "or 'rowwise_with_gw_hp'."
        ),
    )
