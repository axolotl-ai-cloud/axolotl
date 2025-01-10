"""Module for handling differential transfomer input arguments."""

import logging
from typing import Optional

from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class DifferentialTransformerArgs(BaseModel):
    """
    Input args for differential transformer.

    Attributes:
        diff_attention: Whether to use differential attention layers.
        diff_attn_log_every: How often to log differential attention statistics.
        diff_attn_num_monitor_layers: Number of layers to monitor for attention stats.
        diff_attn_warmup_steps: Number of steps to linearly increase negative attention
            mixing weight from 0 to 1. If specified, will reach full mixing at this
            step. If `None`, negative attention has full weight from the start.
    """

    diff_attention: Optional[bool] = None
    diff_attn_log_every: Optional[int] = 100
    diff_attn_num_monitor_layers: Optional[int] = 3
    diff_attn_warmup_steps: Optional[int] = None
