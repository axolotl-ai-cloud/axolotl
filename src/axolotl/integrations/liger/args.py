"""
Module for handling LIGER input arguments.
"""
from pydantic import BaseModel


class LigerArgs(BaseModel):
    """
    Input args for LIGER.
    """

    liger_rope: bool
    liger_rms_norm: bool
    liger_swiglu: bool
    liger_cross_entropy: bool
    liger_fused_linear_cross_entropy: bool
