"""SwanLab configuration arguments"""

from pydantic import BaseModel, Field, field_validator, model_validator


class SwanLabConfig(BaseModel):
    """SwanLab configuration subset"""

    use_swanlab: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "Enable SwanLab experiment tracking and visualization"
        },
    )
    swanlab_project: str | None = Field(
        default=None,
        json_schema_extra={"description": "Your SwanLab project name"},
    )
    swanlab_experiment_name: str | None = Field(
        default=None,
        json_schema_extra={"description": "Set the name of your SwanLab experiment"},
    )
    swanlab_description: str | None = Field(
        default=None,
        json_schema_extra={"description": "Description for your SwanLab experiment"},
    )
    swanlab_mode: str | None = Field(
        default=None,
        json_schema_extra={
            "description": '"cloud" to sync to SwanLab cloud, "local" for local only, "offline" to save metadata locally, "disabled" to turn off SwanLab'
        },
    )
    swanlab_workspace: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "SwanLab workspace name (organization or username)"
        },
    )
    swanlab_api_key: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "SwanLab API key for authentication. Can also be set via SWANLAB_API_KEY environment variable"
        },
    )
    swanlab_log_model: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "Whether to log model checkpoints to SwanLab (feature coming soon)"
        },
    )
    swanlab_web_host: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Web address for SwanLab cloud environment (for private deployment)"
        },
    )
    swanlab_api_host: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "API address for SwanLab cloud environment (for private deployment)"
        },
    )
    swanlab_lark_webhook_url: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Lark (Feishu) webhook URL for sending training notifications to team chat"
        },
    )
    swanlab_lark_secret: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Secret for Lark webhook HMAC signature authentication (optional)"
        },
    )
    swanlab_log_completions: bool | None = Field(
        default=True,
        json_schema_extra={
            "description": "Enable logging RLHF completions to SwanLab for qualitative analysis (DPO/KTO/ORPO/GRPO)"
        },
    )
    swanlab_completion_log_interval: int | None = Field(
        default=100,
        json_schema_extra={
            "description": "Number of training steps between completion table logging to SwanLab"
        },
    )
    swanlab_completion_max_buffer: int | None = Field(
        default=128,
        json_schema_extra={
            "description": "Maximum number of completions to buffer before logging (prevents memory leaks)"
        },
    )

    @field_validator("swanlab_mode")
    @classmethod
    def validate_swanlab_mode(cls, v):
        """Validate swanlab_mode is one of the allowed values."""
        if v is None:
            return v

        valid_modes = ["cloud", "local", "offline", "disabled"]
        if v not in valid_modes:
            raise ValueError(
                f"Invalid swanlab_mode: '{v}'.\n\n"
                f"Valid options: {', '.join(valid_modes)}\n\n"
                f"Examples:\n"
                f"  swanlab_mode: cloud     # Sync to SwanLab cloud\n"
                f"  swanlab_mode: local     # Local only, no cloud sync\n"
                f"  swanlab_mode: offline   # Save metadata locally\n"
                f"  swanlab_mode: disabled  # Turn off SwanLab\n"
            )
        return v

    @field_validator("swanlab_project")
    @classmethod
    def validate_swanlab_project(cls, v):
        """Validate swanlab_project is non-empty when provided."""
        if v is not None and isinstance(v, str) and len(v.strip()) == 0:
            raise ValueError(
                "swanlab_project cannot be an empty string.\n\n"
                "Either:\n"
                "  1. Provide a valid project name: swanlab_project: my-project\n"
                "  2. Remove the swanlab_project field entirely\n"
            )
        return v

    @model_validator(mode="after")
    def validate_swanlab_enabled_requires_project(self):
        """Validate that if use_swanlab is True, swanlab_project must be set."""
        if self.use_swanlab is True and not self.swanlab_project:
            raise ValueError(
                "SwanLab enabled (use_swanlab: true) but 'swanlab_project' is not set.\n\n"
                "Solutions:\n"
                "  1. Add 'swanlab_project: your-project-name' to your config\n"
                "  2. Set 'use_swanlab: false' to disable SwanLab\n\n"
                "Example:\n"
                "  use_swanlab: true\n"
                "  swanlab_project: my-llm-training\n"
            )
        return self
