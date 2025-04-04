"""
LLMCompressor and Sparse Finetuning config models.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import Annotated


class SFTArgs(BaseModel):
    """Sparse Finetuning config for LLMCompressor."""

    # Typing for recipe is set to Any due to:
    # https://github.com/vllm-project/llm-compressor/issues/1319
    recipe: Annotated[Any, Field(description="The recipe containing the compression algorithms and hyperparameters to apply.")]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


class LLMCompressorArgs(BaseModel):
    """LLMCompressor configuration BaseModel."""

    llmcompressor: Annotated[SFTArgs, Field(description="Arguments enabling compression pathways through the LLM Compressor plugins")]

    model_config = ConfigDict(
        validate_assignment=True,
    )
