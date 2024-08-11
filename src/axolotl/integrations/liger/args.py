"""
Module for handling LIGER input arguments.
"""
from typing import Optional

from pydantic import BaseModel


class LigerArgs(BaseModel):
    """
    Input args for LIGER.
    """

    liger_rope: Optional[bool] = None
    liger_rms_norm: Optional[bool] = None
    liger_swiglu: Optional[bool] = None
    liger_cross_entropy: Optional[bool] = None
    liger_fused_linear_cross_entropy: Optional[bool] = None
