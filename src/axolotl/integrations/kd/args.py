"""
Plugin args for KD support.
"""
from typing import Optional

from pydantic import BaseModel


class KDArgs(BaseModel):
    """
    Input args for knowledge distillation.
    """

    kd_trainer: Optional[bool] = None  # whether to use KD trainer
    kd_ce_alpha: Optional[
        float
    ] = None  # loss coefficient for cross-entropy loss during KD
    kd_alpha: Optional[float] = None  # loss coefficient for KD loss
    kd_temperature: Optional[float] = None  # temperature for sampling during KD
