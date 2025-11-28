"""Schema for dynamic checkpoint configuration."""

from pydantic import BaseModel, Field


class DynamicCheckpointConfig(BaseModel):
    """Configuration for dynamic checkpoint triggering during training."""

    enabled: bool = Field(
        default=False,
        json_schema_extra={
            "description": "Enable dynamic checkpoint triggering during training. "
            "Create a file 'axolotl_checkpoint.save' in the configured `output_dir` to trigger. "
        },
    )
    check_interval: int = Field(
        default=10,
        ge=1,
        json_schema_extra={
            "description": "Check for trigger file every N steps (reduces I/O overhead). "
            "Default: 100"
        },
    )
    trigger_file_path: str = Field(
        default="",
        json_schema_extra={
            "description": "Custom trigger filename (optional). "
            "If not specified, defaults to 'axolotl_checkpoint.save'. "
            "Specify a filename (not a full path) to override the default."
        },
    )
