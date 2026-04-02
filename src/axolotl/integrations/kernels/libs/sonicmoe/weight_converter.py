"""
Custom WeightConverter operations for SonicMoE weight format conversion.

SonicMoE requires gate_up_proj weights in interleaved format:
- Standard (concatenated): [E, 2*I, H] where first I rows are gate, last I rows are up
- SonicMoE (interleaved): [E, 2*I, H] where rows alternate [g0, u0, g1, u1, ...]

These ConversionOps integrate with transformers' WeightConverter system so that
weights are transparently converted during loading and reverted during saving.
"""

from typing import Any

import torch
from einops import rearrange
from transformers.core_model_loading import ConversionOps

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def interleave_gate_up(tensor: torch.Tensor) -> torch.Tensor:
    """[gate..., up...] -> [g0, u0, g1, u1, ...] along the 2*I dimension."""
    return rearrange(tensor, "... (two out) h -> ... (out two) h", two=2)


def deinterleave_gate_up(tensor: torch.Tensor) -> torch.Tensor:
    """[g0, u0, g1, u1, ...] -> [gate..., up...] along the 2*I dimension."""
    return rearrange(tensor, "... (out two) h -> ... (two out) h", two=2)


class ConcatenatedToInterleaved(ConversionOps):
    """Convert concatenated gate/up projections to interleaved format.

    Input:  [E, 2*I, H] with gate=[E, :I, H] and up=[E, I:, H]
    Output: [E, 2*I, H] with rows alternating [g0, u0, g1, u1, ...]

    This operation is applied along ``dim`` (default 1, the 2*I dimension).
    """

    def __init__(self, dim: int = 1):
        self.dim = dim

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_pattern = self._get_target_pattern(
            input_dict, source_patterns, target_patterns
        )
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors

        interleaved = interleave_gate_up(tensor)

        return {target_pattern: interleaved}

    def _get_target_pattern(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str],
        target_patterns: list[str],
    ) -> str:
        # Follow the same logic as Transpose.get_target_pattern
        if len(input_dict) != 1:
            raise ValueError("Undefined Operation encountered!")
        if len(target_patterns) > 1:
            if len(source_patterns) == 1:
                return source_patterns[0]
            raise ValueError("Undefined Operation encountered!")
        return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        return InterleavedToConcatenated(self.dim)


class InterleavedToConcatenated(ConversionOps):
    """Convert interleaved gate/up projections back to concatenated format.

    Input:  [E, 2*I, H] with rows alternating [g0, u0, g1, u1, ...]
    Output: [E, 2*I, H] with gate=[E, :I, H] and up=[E, I:, H]

    This is the reverse of ``ConcatenatedToInterleaved``.
    """

    def __init__(self, dim: int = 1):
        self.dim = dim

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        target_pattern = self._get_target_pattern(
            input_dict, source_patterns, target_patterns
        )
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors

        concatenated = deinterleave_gate_up(tensor)

        return {target_pattern: concatenated}

    def _get_target_pattern(
        self,
        input_dict: dict[str, Any],
        source_patterns: list[str],
        target_patterns: list[str],
    ) -> str:
        if len(input_dict) != 1:
            raise ValueError("Undefined Operation encountered!")
        if len(target_patterns) > 1:
            if len(source_patterns) == 1:
                return source_patterns[0]
            raise ValueError("Undefined Operation encountered!")
        return target_patterns[0]

    @property
    def reverse_op(self) -> ConversionOps:
        return ConcatenatedToInterleaved(self.dim)


def _make_same_key_interleave_converter():
    """Create a WeightConverter that interleaves an already-fused gate_up_proj."""
    from transformers.core_model_loading import WeightConverter

    return WeightConverter(
        source_patterns="mlp.experts.gate_up_proj",
        target_patterns="mlp.experts.gate_up_proj",
        operations=[ConcatenatedToInterleaved(dim=1)],
    )


def _has_same_key_interleave(mapping) -> bool:
    """Check whether the mapping already has a same-key gate_up_proj interleave converter."""
    for conv in mapping:
        if (
            hasattr(conv, "source_patterns")
            and conv.source_patterns == ["mlp.experts.gate_up_proj"]
            and conv.target_patterns == ["mlp.experts.gate_up_proj"]
            and hasattr(conv, "operations")
            and any(isinstance(op, ConcatenatedToInterleaved) for op in conv.operations)
        ):
            return True
    return False


def register_sonicmoe_weight_converter(model_type: str):
    """Register weight converters to interleave gate_up_proj for SonicMoE.

    Handles two checkpoint formats:
    1. Separate per-expert weights (e.g. qwen3_moe): appends interleave to the
       existing merge chain (MergeModulelist -> Concatenate -> Interleave).
    2. Already-fused gate_up_proj (e.g. qwen3_5_moe_text): adds a same-key
       converter (gate_up_proj -> gate_up_proj with Interleave).

    The loader matches whichever source pattern exists in the checkpoint.
    """
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )

    existing = get_checkpoint_conversion_mapping(model_type)

    if existing is None:
        # No mapping at all — create one with just the same-key converter
        mapping = [_make_same_key_interleave_converter()]
        register_checkpoint_conversion_mapping(model_type, mapping)
        LOG.info(f"Registered SonicMoE weight converter for model type '{model_type}'")
        return

    # Append interleave to any existing many-to-one merge chain
    for converter in existing:
        if hasattr(converter, "operations") and any(
            "gate_up_proj" in pat for pat in converter.target_patterns
        ):
            has_separate_sources = any(
                "gate_proj" in pat or "up_proj" in pat
                for pat in converter.source_patterns
            )
            if has_separate_sources and not any(
                isinstance(op, ConcatenatedToInterleaved) for op in converter.operations
            ):
                converter.operations.append(ConcatenatedToInterleaved(dim=1))
            break

    # Also add a same-key converter for already-fused checkpoints
    if not _has_same_key_interleave(existing):
        existing.append(_make_same_key_interleave_converter())

    register_checkpoint_conversion_mapping(model_type, existing, overwrite=True)
    LOG.info(f"Registered SonicMoE weight converter for model type '{model_type}'")
