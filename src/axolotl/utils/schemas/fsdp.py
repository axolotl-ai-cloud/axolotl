from typing import Literal

from packaging import version
from pydantic import BaseModel, Field, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class FSDPConfig(BaseModel):
    fsdp: list[str] | None = Field(
        default=None,
        json_schema_extra={"description": "FSDP configuration"},
        deprecated="Configuring FSDP using `fsdp` is deprecated. Please use `fsdp_config` instead. ",
    )
    fsdp_version: int | None = Field(
        default=None,
        json_schema_extra={"description": "FSDP version"},
    )
    fp8_enable_fsdp_float8_all_gather: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": "Enable FSDP float8 all-gather optimization for FP8 training. Can "
            "improve training speed by 10-15% when FSDP is enabled."
        },
    )
    final_state_dict_type: (
        Literal["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"] | None
    ) = Field(
        default=None,
        deprecated="Configuring FSDP final state dict type using `fsdp_final_state_dict_type` is deprecated. Please use `fsdp_config.final_state_dict_type` instead.",
    )
    activation_checkpointing: bool | None = Field(
        default=None, description="Enable activation checkpointing for FSDP."
    )
    offload_params: bool | None = Field(
        default=None, description="Enable parameter offloading to CPU for FSDP."
    )
    sync_module_states: bool | None = Field(
        default=None, description="Synchronize module states across FSDP processes."
    )
    use_orig_params: bool | None = Field(
        default=None, description="Use original parameters for FSDP."
    )
    state_dict_type: (
        Literal["FULL_STATE_DICT", "LOCAL_STATE_DICT", "SHARDED_STATE_DICT"] | None
    ) = Field(default=None, description="Type of state dict to use for FSDP.")
    auto_wrap_policy: Literal["TRANSFORMER_BASED_WRAP"] | None = Field(
        default=None, description="Auto wrap policy for FSDP."
    )
    transformer_layer_cls_to_wrap: list[str] | None = Field(
        default=None, description="List of transformer layer classes to wrap with FSDP."
    )

    @model_validator(mode="before")
    @classmethod
    def check_fsdp_torch_version(cls, data):
        env_capabilities = data.get("env_capabilities", {})
        torch_version = env_capabilities.get("torch_version")

        if torch_version is None:
            import torch

            torch_version = str(torch.__version__).split("+", maxsplit=1)[0]

        if data.get("fsdp_config") and str(data.get("fsdp_version")) == "2":
            if version.parse(torch_version) < version.parse("2.7.0"):
                raise ValueError("FSDP2 is not supported on torch version < 2.7.0")

        return data
