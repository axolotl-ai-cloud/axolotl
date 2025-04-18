""" "
Takes care of quantization configuration
"""

from typing import Annotated

from annotated_types import MinLen
from pydantic import BaseModel, Field, model_validator


class HQQConfig(BaseModel):
    """HQQ configuration subset"""

    nbits: int | None = Field(default=None)
    group_size: int | None = Field(default=None)
    target_modules: list[str] | str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Target modules for HQQ quantization. If not specified, the whole model will be quantized."
        },
    )


class QuantizationConfig(BaseModel):
    """Over all Quantization configuration subset"""

    # We will use this class as base future refactoring of all quantization configs
    use_hqq: bool = False
    hqq_config: Annotated[list[HQQConfig], MinLen(1)] | None = None

    @model_validator(mode="before")
    @classmethod
    def check_hqq_config(cls, data):
        if data.get("use_hqq") and not data.get("hqq_config"):
            raise ValueError(
                "If using HQQ, must set `hqq_config` to a list of HQQConfig objects"
            )

        if data.get("hqq_config") and len(data.get("hqq_config")) > 1:
            for hqq_config in data.get("hqq_config"):
                if hqq_config.get("target_modules") is None:
                    raise ValueError(
                        "If using HQQ, `target_modules` must be specified for each HQQConfig object"
                    )

        return data


def get_hqq_quant_config_kwargs(cfg):

    # If no target module is specified, then target the whole model
    if len(cfg.hqq_config) == 1 and cfg.hqq_config[0].target_modules is None:
        return {
            "nbits": cfg.hqq_config[0].nbits,
            "group_size": cfg.hqq_config[0].group_size,
        }

    hqq_quant_config_kwargs = {"dynamic_config": {}}
    for hqq_config in cfg.hqq_config:
        target_modules = hqq_config.target_modules
        if not isinstance(target_modules, list):
            target_modules = [target_modules]

        for module in target_modules:
            hqq_quant_config_kwargs["dynamic_config"][module] = {
                "nbits": hqq_config.nbits,
                "group_size": hqq_config.group_size,
            }

    return hqq_quant_config_kwargs
