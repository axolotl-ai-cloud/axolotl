"""DFT (Dynamic Fine-Tuning) plugin arguments.

This module defines configuration options for Dynamic Fine-Tuning (DFT) loss,
including support for:
- Basic DFT: L_dft = L_ce * exp(-L_ce.detach())
- Chunked cross-entropy for memory optimization (dft_chunk_size)
- Channel loss integration (enable_dft_channel_loss)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from pydantic import BaseModel, Field


class DFTArgs(BaseModel):
    """Input args for Dynamic Fine-Tuning (DFT) loss.

    DFT dynamically adjusts per-token loss weights based on the model's
    current prediction confidence. Tokens with medium difficulty (loss ≈ 1)
    receive the highest training signal.

    Reference: https://arxiv.org/abs/2508.05629
    """

    enable_dft_loss: bool = Field(
        default=False,
        description=(
            "Enable DFT loss: L_dft = L_ce * exp(-L_ce.detach()) for SFT training."
        ),
    )

    dft_chunk_size: Optional[int] = Field(
        default=None,
        description=(
            "Chunk size for memory-efficient cross-entropy computation. "
            "When set (e.g., 2048), logits are processed in chunks to reduce peak memory "
            "by 50-75% for large vocabulary models (e.g., Qwen 152K vocab). "
            "Trade-off: ~5% more compute time for significant memory savings. "
            "\n\nRecommended values:"
            "\n- vocab_size < 50K: None (chunking not needed)"
            "\n- vocab_size 50K-100K: 2048-4096"
            "\n- vocab_size > 100K: 1024-2048"
            "\n\nEquivalent to ms-swift's CELOSS_PARALLEL_SIZE."
        ),
    )

    enable_dft_channel_loss: bool = Field(
        default=False,
        description=(
            "Enable per-channel loss tracking alongside DFT. "
            "This allows monitoring loss per data source/channel while using DFT. "
            "Requires channel_loss plugin to be enabled."
        ),
    )


@dataclass
class DFTTrainingArgsMixin:
    """TrainingArguments mixin for DFT."""

    enable_dft_loss: bool = field(
        default=False,
        metadata={
            "help": "Enable DFT loss: L_dft = L_ce * exp(-L_ce.detach())",
        },
    )

    dft_chunk_size: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Chunk size for memory-efficient cross-entropy computation. "
                "Reduces peak memory by 50-75% for large vocab models (e.g., Qwen 152K). "
                "Recommended: 2048 for vocab > 100K, None for vocab < 50K. "
                "Trade-off: ~5% more compute for significant memory savings."
            ),
        },
    )

    enable_dft_channel_loss: bool = field(
        default=False,
        metadata={
            "help": "Enable per-channel loss tracking with DFT.",
        },
    )

