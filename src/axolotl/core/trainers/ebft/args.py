"""
EBFT-specific training arguments.

Two config classes:
- AxolotlEBFTConfig: extends GRPOConfig for structured QA data (uses vLLM generation)
- AxolotlStridedEBFTConfig: extends TrainingArguments for unstructured text (strided generation)
"""

from dataclasses import dataclass, field

from transformers import TrainingArguments
from trl import GRPOConfig

from axolotl.core.training_args import AxolotlTrainingMixins


# -- Shared EBFT fields as a mixin --
@dataclass
class EBFTFieldsMixin:
    """Common fields shared between structured and strided EBFT configs."""

    ebft_feature_layers: list[float] = field(
        default_factory=lambda: [0.25, 0.5, 0.75],
        metadata={"help": "Fractional layer depths for feature extraction"},
    )
    ebft_embed_method: str = field(
        default="last_token",
        metadata={"help": "Pooling method: 'last_token', 'mean_pooling', or 'concat'"},
    )
    ebft_use_whitening: bool = field(
        default=False,
        metadata={"help": "Apply SVD whitening to feature embeddings"},
    )
    ebft_alignment_coef: float = field(
        default=1.0,
        metadata={"help": "Coefficient for alignment reward (cosine similarity)"},
    )
    ebft_diversity_coef: float = field(
        default=1.0,
        metadata={"help": "Coefficient for diversity penalty"},
    )
    ebft_ce_coef: float = field(
        default=0.0,
        metadata={"help": "Cross-entropy loss coefficient on ground-truth tokens"},
    )


# -- Structured mode: extends GRPOTrainer for QA data with vLLM --
@dataclass
class AxolotlEBFTConfig(EBFTFieldsMixin, AxolotlTrainingMixins, GRPOConfig):
    """EBFT config for structured QA data — extends GRPOConfig."""


# -- Strided mode: extends TrainingArguments for unstructured text --
@dataclass
class AxolotlStridedEBFTConfig(EBFTFieldsMixin, AxolotlTrainingMixins, TrainingArguments):
    """EBFT config for unstructured text with strided block-parallel generation."""

    ebft_stride: int = field(
        default=8,
        metadata={"help": "Stride between anchor points (in tokens)"},
    )
    ebft_context_length: int = field(
        default=8,
        metadata={"help": "Context window size for each block"},
    )
    ebft_generate_max_len: int = field(
        default=8,
        metadata={"help": "Number of tokens to generate per block"},
    )
    ebft_n_samples_per_prompt: int = field(
        default=4,
        metadata={"help": "Number of independent rollouts per document"},
    )
    ebft_temperature: float = field(
        default=0.6,
        metadata={"help": "Sampling temperature for strided generation"},
    )
    ebft_top_p: float = field(
        default=1.0,
        metadata={"help": "Top-p nucleus sampling threshold"},
    )
    ebft_rl_coef: float = field(
        default=1.0,
        metadata={"help": "RL policy gradient loss coefficient"},
    )
    ebft_advantage_estimator: str = field(
        default="rloo",
        metadata={"help": "Advantage estimator: 'rloo', 'group_norm', or 'reinforce'"},
    )
