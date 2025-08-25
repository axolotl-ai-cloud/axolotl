"""Pydantic models for DenseMixer plugin"""

from pydantic import BaseModel


class DenseMixerArgs(BaseModel):
    """
    Args for DenseMixer
    """

    dense_mixer: bool = True
