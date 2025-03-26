"""
LLMCompressor and Sparse Finetuning config models.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Any
from typing_extensions import Annotated

class SFTArgs(BaseModel):
    """Sparse Finetuning config for LLMCompressor."""

    # Typing for recipe is set to Any due to:
    # https://github.com/vllm-project/llm-compressor/issues/1319
    recipe: Annotated[
        Any,
        Field(description="Recipe config.")
    ]

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=True,
    )


class LLMCompressorArgs(BaseModel):
    """LLMCompressor configuration BaseModel."""

    llmcompressor: Annotated[
        SFTArgs,
        Field(description="SFT llmcompressor args")
    ]

    model_config = ConfigDict(
        validate_assignment=True,
    )
