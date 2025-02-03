"""
Module for handling linear attention input arguments.
"""

from typing import Optional

from pydantic import BaseModel


class FeatureMapKwargs(BaseModel):
    """Args for feature map"""

    eps: float
    mlp: Optional[None] = None
    fullspace: bool


class LearnedKernelKwargs(BaseModel):
    """Args for learned kernel"""

    feature_dim: int
    skip_connection: bool
    bias: bool
    zero_init: bool


class AttentionConfig(BaseModel):
    """Args for attention config"""

    attention_type: str
    feature_map: str
    feature_map_kwargs: FeatureMapKwargs
    layer_idx: Optional[None] = None
    learned_kernel: str
    learned_kernel_kwargs: LearnedKernelKwargs
    tie_qk_kernels: bool
    train_qk: bool


class LinearAttentionArgs(BaseModel):
    """
    Input args for linear attention
    """

    attention_config: AttentionConfig

    linearize: bool
