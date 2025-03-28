"""Pydantic models for Axolotl integrations"""

import logging
from typing import Any

from pydantic import BaseModel, Field, model_validator

LOG = logging.getLogger(__name__)


class MLFlowConfig(BaseModel):
    """MLFlow configuration subset"""

    use_mlflow: bool | None = None
    mlflow_tracking_uri: str | None = None
    mlflow_experiment_name: str | None = None
    mlflow_run_name: str | None = None
    hf_mlflow_log_artifacts: bool | None = None


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
    wandb_name: str | None = None
    wandb_run_id: str | None = None
    wandb_mode: str | None = None
    wandb_project: str | None = None
    wandb_entity: str | None = None
    wandb_watch: str | None = None
    wandb_log_model: str | None = None

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

    use_comet: bool | None = None
    comet_api_key: str | None = None
    comet_workspace: str | None = None
    comet_project_name: str | None = None
    comet_experiment_key: str | None = None
    comet_mode: str | None = None
    comet_online: bool | None = None
    comet_experiment_config: dict[str, Any] | None = None


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
