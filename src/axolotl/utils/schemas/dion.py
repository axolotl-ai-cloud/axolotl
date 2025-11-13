"""pydantic models for dion optimizer configuration"""

from pydantic import BaseModel, Field


class DionConfig(BaseModel):
    """dion optimizer configuration subset"""

    dion_lr: float | None = Field(
        default=None,
        json_schema_extra={"description": "learning rate for dion optimizer"},
    )
    dion_momentum: float | None = Field(
        default=None,
        json_schema_extra={"description": "momentum for dion optimizer"},
    )
    dion_rank_fraction: float | None = Field(
        default=1.0,
        json_schema_extra={
            "description": "r/d fraction for low-rank approximation. used to compute the low-rank dimension. 1.0 means full rank, 0.5 means 50% rank compression."
        },
    )
    dion_rank_multiple_of: int | None = Field(
        default=1,
        json_schema_extra={
            "description": "round up the low-rank dimension to a multiple of this number. useful to ensure even sharding in fsdp."
        },
    )
