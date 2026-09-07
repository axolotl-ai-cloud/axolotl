"""
LLMCompressor and Sparse Finetuning config models.
"""

from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import Annotated


class CompressionArgs(BaseModel):
    """Sparse Finetuning config for LLMCompressor."""

    # Typing for recipe is set to Any due to:
    # https://github.com/vllm-project/llm-compressor/issues/1319
    recipe: Annotated[
        Any,
        Field(
            description="The recipe containing the compression algorithms and hyperparameters to apply."
        ),
    ]

    save_compressed: Annotated[
        bool,
        Field(
            default=False,
            description="Whether to save the compressed model after training.",
        ),
    ]


class LLMCompressorArgs(BaseModel):
    """LLMCompressor configuration BaseModel."""

    llmcompressor: Annotated[
        CompressionArgs,
        Field(
            description="Arguments enabling compression pathways through the LLM Compressor plugins"
        ),
    ]
