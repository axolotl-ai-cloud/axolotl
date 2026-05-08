"""Config args for MoRA / ReMoRA."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class MoraConfig(BaseModel):
    """Nested MoRA configuration available under the `mora` key."""

    use_mora: bool = Field(
        default=True,
        description=(
            "Enable MoRA adapter construction. Requires a PEFT build with MoRA "
            "support (for example, the MoRA fork)."
        ),
    )
    mora_type: int = Field(
        default=6,
        ge=1,
        description=(
            "MoRA variant selector. The MoRA repo uses type 1 for sharing and "
            "type 6 for RoPE-based updates."
        ),
    )
    use_relora: bool = Field(
        default=False,
        description=(
            "Enable ReMoRA restart scheduling. Axolotl maps this to the existing "
            "ReLoRA restart path."
        ),
    )
    use_relora_step: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Restart interval in steps when ReMoRA is enabled. This maps to "
            "Axolotl's jagged_restart_steps."
        ),
    )

    @model_validator(mode="after")
    def validate_relora(self):
        if self.use_relora_step is not None:
            self.use_relora = True
        if self.use_relora and self.use_relora_step is None:
            raise ValueError("mora.use_relora requires mora.use_relora_step")
        return self


class MoraArgs(BaseModel):
    """Plugin entry that exposes the nested `mora` block to the core config."""

    mora: MoraConfig = Field(
        default_factory=MoraConfig,
        description=(
            "MoRA / ReMoRA training configuration. Register the "
            "`axolotl.integrations.mora.MoraPlugin` plugin to enable this block."
        ),
    )
