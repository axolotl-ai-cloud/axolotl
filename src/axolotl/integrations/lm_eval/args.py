"""
Module for handling lm eval harness input arguments.
"""

from typing import List, Optional

from pydantic import BaseModel


class LMEvalArgs(BaseModel):
    """
    Input args for lm eval harness
    """

    lm_eval_tasks: List[str] = []
    lm_eval_batch_size: Optional[int] = 8
    lm_eval_post_train: Optional[bool] = True
    lm_eval_model: Optional[str] = None
