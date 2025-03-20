"""
config args for grokfast plugin
"""

from typing import Optional

from pydantic import BaseModel


class GrokfastArgs(BaseModel):
    """
    Input args for Grokfast optimizer.
    """

    grokfast_alpha: Optional[float] = 0.98
    grokfast_lamb: Optional[float] = 2.0
