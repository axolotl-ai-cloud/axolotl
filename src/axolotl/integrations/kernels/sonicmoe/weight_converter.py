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
    ) -> dict[str, list[torch.Tensor]]:
        target_pattern = self._get_target_pattern(
            input_dict, source_patterns, target_patterns
        )
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors

        # [gate..., up...] -> [g0, u0, g1, u1, ...] along the 2*I dimension
        interleaved = rearrange(tensor, "... (two out) h -> ... (out two) h", two=2)

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
    ) -> dict[str, list[torch.Tensor]]:
        target_pattern = self._get_target_pattern(
            input_dict, source_patterns, target_patterns
        )
        tensors = next(iter(input_dict.values()))
        tensor = tensors[0] if isinstance(tensors, list) else tensors

        # [g0, u0, g1, u1, ...] -> [gate..., up...] along the 2*I dimension
        concatenated = rearrange(tensor, "... (out two) h -> ... (two out) h", two=2)

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


def register_sonicmoe_weight_converter(model_type: str):
    """Override the conversion mapping to add interleave step for gate_up_proj.

    Appends a ConcatenatedToInterleaved operation to the existing gate_up_proj
    converter chain. For example, qwen3_moe's chain becomes:
        MergeModulelist(dim=0) -> Concatenate(dim=1) -> ConcatenatedToInterleaved(dim=1)

    The reverse is auto-generated for saving:
        InterleavedToConcatenated(dim=1) -> Chunk(dim=1) -> SplitModulelist(dim=0)
    """
    from transformers.conversion_mapping import (
        get_checkpoint_conversion_mapping,
        register_checkpoint_conversion_mapping,
    )

    existing = get_checkpoint_conversion_mapping(model_type)
    if existing is None:
        LOG.warning(
            f"No conversion mapping found for model type '{model_type}'. "
            "SonicMoE weight interleaving will not be applied during checkpoint loading."
        )
        return

    # Find the gate_up_proj converter and append ConcatenatedToInterleaved
    patched = False
    for converter in existing:
        if hasattr(converter, "operations") and any(
            "gate_up_proj" in pat for pat in converter.target_patterns
        ):
            # Guard against double registration (e.g. plugin reloaded)
            if any(
                isinstance(op, ConcatenatedToInterleaved) for op in converter.operations
            ):
                LOG.info(
                    f"SonicMoE weight converter already registered for '{model_type}'"
                )
                return
            converter.operations.append(ConcatenatedToInterleaved(dim=1))
            patched = True
            break

    if not patched:
        LOG.warning(
            f"Could not find gate_up_proj converter for model type '{model_type}'. "
            "SonicMoE weight interleaving will not be applied during checkpoint loading."
        )
        return

    register_checkpoint_conversion_mapping(model_type, existing, overwrite=True)
    LOG.info(f"Registered SonicMoE weight converter for model type '{model_type}'")
