"""
Pydantic model for accepting `llmcompressor` specific arguments.
"""
from typing import Optional, Any
from pydantic import BaseModel


class LLMCompressorArgs(BaseModel):
    """
    Input arguments for Sparse Finetuning.
    """
    
    recipe: Optional[Any] = None