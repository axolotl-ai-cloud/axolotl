"""Module for handling differential transfomer input arguments."""

import logging
from typing import Optional

from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class DifferentialTransformerArgs(BaseModel):
    """Input args for differential transformer."""

    diff_attention: Optional[bool] = None
    diff_attn_log_every: Optional[int] = 100
    diff_attn_num_monitor_layers: Optional[int] = 3
