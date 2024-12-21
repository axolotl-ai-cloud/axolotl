"""Module for handling RALA input arguments."""

import logging
from typing import Optional

from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class RalaArgs(BaseModel):
    """Input args for RALA."""

    rala_attention: Optional[bool] = None
