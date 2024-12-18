"""Module for handling differential transfomer input arguments."""

import logging
from typing import Optional

from pydantic import BaseModel

LOG = logging.getLogger(__name__)


class DifferentialTransformerArgs(BaseModel):
    """Input args for differential transformer."""

    differential_attention: Optional[bool] = None
