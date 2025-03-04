"""
Module for handling llm-compressor input arguments.
"""
from typing import Optional, Any
from pydantic import BaseModel


class LLMCompressorArgs(BaseModel):
    """
    Input args for Sparse Finetuning with llmcompressor.
    """ 
    recipe: Optional[Any] = None # TODO: Make this a Recipe object