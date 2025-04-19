""" "
Takes care of quantization configuration
"""

from typing import Annotated, Any, Literal

from annotated_types import MinLen
from pydantic import BaseModel, Field, model_validator


class HQQConfig(BaseModel):
    """HQQ configuration subset"""

    nbits: Literal[8, 4, 3, 2, 1] | None = Field(
        default=None,
        json_schema_extra={
            "description": "Number of bits for HQQ quantization. 8, 4, 3, 2, or 1."
        },
    )

    group_size: int = Field(default=64)
    target_modules: list[str] | str | None = Field(
        default=None,
        json_schema_extra={
            "description": "Target modules for HQQ quantization. If not specified, the whole model will be quantized."
        },
    )


class QuantizationConfig(BaseModel):
    """Over all Quantization configuration subset"""

    # We will use this class as base future refactoring of all quantization configs
    backend: Literal["bnb", "hqq", "gptq"] | None = None
    bits: Literal[8, 4, 3, 2, 1] | None = None
    bnb_config_kwargs: dict[str, Any] | None = None
    hqq_config: HQQConfig | Annotated[list[HQQConfig], MinLen(1)] | None = None

    @model_validator(mode="before")
    @classmethod
    def check_hqq_config(cls, data):
        if data.get("backend") == "hqq" and not data.get("hqq_config"):
            raise ValueError("If using HQQ, must set `group_size` under `hqq_config`")

        if data.get("hqq_config") and len(data.get("hqq_config")) > 1:
            for hqq_config in data.get("hqq_config"):
                if hqq_config.get("target_modules") is None:
                    raise ValueError(
                        "For list of hqq configs, `target_modules` must be specified for each"
                    )

        return data


def get_hqq_quant_config_kwargs(cfg):

    # If no target module is specified, then target the whole model
    if not isinstance(cfg.quantization.hqq_config, list):
        cfg.quantization.hqq_config = [cfg.quantization.hqq_config]

    if (
        len(cfg.quantization.hqq_config) == 1
        and cfg.quantization.hqq_config[0].target_modules is None
    ):

        nbits = (
            cfg.quantization.hqq_config[0].nbits
            if cfg.quantization.hqq_config[0].nbits is not None
            else cfg.quantization.bits
        )

        return {
            "nbits": nbits,
            "group_size": cfg.quantization.hqq_config[0].group_size,
        }

    hqq_quant_config_kwargs = {"dynamic_config": {}}
    for hqq_config in cfg.quantization.hqq_config:
        nbits = (
            hqq_config.nbits if hqq_config.nbits is not None else cfg.quantization.bits
        )

        target_modules = hqq_config.target_modules
        if not isinstance(target_modules, list):
            target_modules = [target_modules]

        for module in target_modules:
            hqq_quant_config_kwargs["dynamic_config"][module] = {
                "nbits": nbits,
                "group_size": hqq_config.group_size,
            }

    return hqq_quant_config_kwargs
