"""
Module for handling linear attention input arguments.
"""

from pydantic import BaseModel


class LinearAttentionArgs(BaseModel):
    """
    Input args for linear attention
    """

    attention_config: dict
