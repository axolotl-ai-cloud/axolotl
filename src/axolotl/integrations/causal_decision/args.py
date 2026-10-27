"""Configuration for decision-head training on causal LMs."""

from pydantic import BaseModel, ConfigDict, Field


class CausalDecisionConfig(BaseModel):
    """Decision-head and data settings."""

    model_config = ConfigDict(extra="forbid")

    num_options: int = Field(
        default=16,
        ge=2,
        le=26,
        json_schema_extra={
            "description": "Decision-head width: answer letters from A (16 covers A-P)."
        },
    )
    rank: int = Field(
        default=16,
        ge=1,
        json_schema_extra={
            "description": "Rank of the zero-initialized correction added to the lm_head letter rows."
        },
    )
    shuffle: bool = Field(
        default=True,
        json_schema_extra={
            "description": "Shuffle each record's choice order, remapping its targets."
        },
    )


class CausalDecisionArgs(BaseModel):
    """Plugin entry that exposes the ``causal_decision`` block."""

    causal_decision: CausalDecisionConfig = Field(
        default_factory=CausalDecisionConfig,
        json_schema_extra={
            "description": "Decision-head training settings for causal LMs."
        },
    )
