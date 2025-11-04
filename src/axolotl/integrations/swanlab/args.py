"""SwanLab configuration arguments"""

from pydantic import BaseModel, Field


class SwanLabConfig(BaseModel):
    """SwanLab configuration subset"""

    use_swanlab: bool | None = Field(
        default=None,
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






