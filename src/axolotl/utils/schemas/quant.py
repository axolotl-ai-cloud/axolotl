""" "
Takes care of quantization configuration
"""

from typing import Literal

from pydantic import BaseModel, model_validator


class HQQConfig(BaseModel):
    """HQQ configuration subset"""

    hqq_nbits: Literal[8, 4, 3, 2, 1] | None = None
    hqq_group_size: int | None = None
    hqq_target_module: list[str] | None = None

    @model_validator(mode="before")
    @classmethod
    def check_hqq_config_fields(cls, data):
        fields = ("hqq_nbits", "hqq_group_size")
        non_empty_count = sum(1 for field in fields if data.get(field))
        if non_empty_count == 1 or (
            data.get("'hqq_target_module") and non_empty_count < 2
        ):
            raise ValueError(
                "If using HQQ, must set both `hqq_nbits` and `hqq_group_size`"
            )


def get_hqq_quant_config_kwargs(cfg):

    # If no target module is specified, then target the whole model
    if cfg.hqq_module_name is None:
        return {
            "nbits": cfg.hqq_nbits,
            "group_size": cfg.hqq_group_size,
        }

    hqq_target_module = cfg.hqq_target_module
    if not isinstance(cfg.hqq_target_module, list):
        hqq_target_module = [hqq_target_module]

    hqq_quant_config_kwargs = {"dynamic_config": {}}
    for module in hqq_target_module:
        hqq_quant_config_kwargs["dynamic_config"][module] = {
            "nbits": cfg.hqq_nbits,
            "group_size": cfg.hqq_group_size,
        }
    return hqq_quant_config_kwargs
