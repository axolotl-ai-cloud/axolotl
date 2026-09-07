"""Pydantic models for Axolotl integrations"""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class MLFlowConfig(BaseModel):
    """MLFlow configuration subset"""

    use_mlflow: bool | None = None
    mlflow_tracking_uri: str | None = Field(
        default=None, json_schema_extra={"description": "URI to mlflow"}
    )
    mlflow_experiment_name: str | None = Field(
        default=None, json_schema_extra={"description": "Your experiment name"}
    )
    mlflow_run_name: str | None = Field(
        default=None, json_schema_extra={"description": "Your run name"}
    )
    hf_mlflow_log_artifacts: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "set to true to copy each saved checkpoint on each save to mlflow artifact registry"
        },
    )


class LISAConfig(BaseModel):
    """LISA configuration subset"""

    lisa_n_layers: int | None = Field(
        default=None,
        json_schema_extra={"description": "the number of activate layers in LISA"},
    )
    lisa_step_interval: int | None = Field(
        default=None,
        json_schema_extra={"description": "how often to switch layers in LISA"},
    )
    lisa_layers_attribute: str | None = Field(
        default="model.layers",
        json_schema_extra={"description": "path under the model to access the layers"},
    )


class WandbConfig(BaseModel):
    """Wandb configuration subset"""

    use_wandb: bool | None = None
    wandb_name: str | None = Field(
        default=None,
        json_schema_extra={"description": "Set the name of your wandb run"},
    )
    wandb_run_id: str | None = Field(
        default=None, json_schema_extra={"description": "Set the ID of your wandb run"}
    )
    wandb_mode: str | None = Field(
        default=None,
        json_schema_extra={
            "description": '"offline" to save run metadata locally and not sync to the server, "disabled" to turn off wandb'
        },
    )
    wandb_project: str | None = Field(
        default=None, json_schema_extra={"description": "Your wandb project name"}
    )
    wandb_entity: str | None = Field(
        default=None,
        json_schema_extra={"description": "A wandb Team name if using a Team"},
    )
    wandb_watch: str | None = None
    wandb_log_model: str | None = Field(
        default=None,
        json_schema_extra={
            "description": '"checkpoint" to log model to wandb Artifacts every `save_steps` or "end" to log only at the end of training'
        },
    )

    @model_validator(mode="before")
    @classmethod
    def check_wandb_run(cls, data):
        if data.get("wandb_run_id") and not data.get("wandb_name"):
            data["wandb_name"] = data.get("wandb_run_id")

            LOG.warning(
                "wandb_run_id sets the ID of the run. If you would like to set the name, please use wandb_name instead."
            )

        return data


class CometConfig(BaseModel):
    """Comet configuration subset"""

    use_comet: bool | None = Field(
        default=None,
        json_schema_extra={"description": "Enable or disable Comet integration."},
    )
    comet_api_key: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "API key for Comet. Recommended to set via `comet login`."
        },
    )
    comet_workspace: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Workspace name in Comet. Defaults to the user's default workspace."
        },
    )
    comet_project_name: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Project name in Comet. Defaults to Uncategorized."
        },
    )
    comet_experiment_key: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Identifier for the experiment. Used to append data to an existing experiment or control the key of new experiments. Default to a random key."
        },
    )
    comet_mode: str | None = Field(
        default=None,
        json_schema_extra={
            "description": 'Create a new experiment ("create") or log to an existing one ("get"). Default ("get_or_create") auto-selects based on configuration.'
        },
    )
    comet_online: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Set to True to log data to Comet server, or False for offline storage. Default is True."
        },
    )
    comet_experiment_config: dict[str, Any] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Dictionary for additional configuration settings, see the doc for more details."
        },
    )


class GradioConfig(BaseModel):
    """Gradio configuration subset"""

    gradio_title: str | None = None
    gradio_share: bool | None = None
    gradio_server_name: str | None = None
    gradio_server_port: int | None = None
    gradio_max_new_tokens: int | None = None
    gradio_temperature: float | None = None


class RayConfig(BaseModel):
    """Ray launcher configuration subset"""

    use_ray: bool = Field(default=False)
    ray_run_name: str | None = Field(
        default=None,
        json_schema_extra={
            "help": "The training results will be saved at `saves/ray_run_name`."
        },
    )
    ray_num_workers: int = Field(
        default=1,
        json_schema_extra={
            "help": "The number of workers for Ray training. Default is 1 worker."
        },
    )
    resources_per_worker: dict = Field(
        default_factory=lambda: {"GPU": 1},
        json_schema_extra={
            "help": "The resources per worker for Ray training. Default is to use 1 GPU per worker."
        },
    )


class OpenTelemetryConfig(BaseModel):
    """OpenTelemetry configuration subset"""

    use_otel_metrics: bool | None = Field(
        default=False,
        json_schema_extra={
            "description": "Enable OpenTelemetry metrics collection and Prometheus export"
        },
    )
    otel_metrics_host: str | None = Field(
        default="localhost",
        json_schema_extra={
            "title": "OpenTelemetry Metrics Host",
            "description": "Host to bind the OpenTelemetry metrics server to",
        },
    )
    otel_metrics_port: int | None = Field(
        default=8000,
        json_schema_extra={
            "description": "Port for the Prometheus metrics HTTP server"
        },
    )


class TrackioConfig(BaseModel):
    """Trackio configuration subset"""

    use_trackio: bool | None = None
    trackio_project_name: str | None = Field(
        default=None,
        json_schema_extra={"description": "Your trackio project name"},
    )
    trackio_run_name: str | None = Field(
        default=None,
        json_schema_extra={"description": "Set the name of your trackio run"},
    )
    trackio_space_id: str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Hugging Face Space ID to sync dashboard to (optional, runs locally if not provided)"
        },
    )
