"""Config args for MoRA / ReMoRA."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator


class MoraType(str, Enum):
    """MoRA variants supported by the reference implementation."""

    SHARING = "sharing"
    ROPE = "rope"

    @property
    def peft_value(self) -> int:
        return {
            MoraType.SHARING: 1,
            MoraType.ROPE: 6,
        }[self]


class MoraConfig(BaseModel):
    """Nested MoRA configuration available under the `mora` key."""

    use_mora: bool = Field(
        default=True,
        description=(
            "Enable MoRA adapter construction. Requires a PEFT build with MoRA "
            "support (for example, the MoRA fork)."
        ),
    )
    mora_type: MoraType = Field(
        default=MoraType.ROPE,
        description=(
            "MoRA variant selector. Supported values are `sharing` for type 1 "
            "and `rope` for type 6. Numeric values 1 and 6 are accepted for "
            "backwards compatibility."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def normalize_mora_type(cls, data):
        if not isinstance(data, dict) or "mora_type" not in data:
            return data
        data = data.copy()
        mora_type = data["mora_type"]
        if mora_type == 1:
            data["mora_type"] = MoraType.SHARING
        elif mora_type == 6:
            data["mora_type"] = MoraType.ROPE
        return data


class MoraArgs(BaseModel):
    """Plugin entry that exposes the nested `mora` block to the core config."""

    mora: MoraConfig = Field(
        default_factory=MoraConfig,
        description=(
            "MoRA / ReMoRA training configuration. Register the "
            "`axolotl.integrations.mora.MoraPlugin` plugin to enable this block."
        ),
    )
