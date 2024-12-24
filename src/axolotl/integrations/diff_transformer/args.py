"""Module for handling differential transfomer input arguments."""

import logging
from typing import Optional

from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class DifferentialTransformerArgs(BaseModel):
    """Input args for differential transformer."""

    diff_attention: Optional[bool] = None
    diff_attn_zero_init: Optional[bool] = None
    diff_attn_sublayer_norm: Optional[bool] = None
    diff_attn_split_heads: Optional[bool] = None
