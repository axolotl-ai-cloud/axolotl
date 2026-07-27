"""Triton fused ops for ternary QAT.

Signatures mirror the eager oracle in `axolotl.integrations.ternary.quant`
one-for-one; results must match it bit-exactly on the integer paths and within the
documented tolerance on the float ones.
"""

from __future__ import annotations

import torch

from ._common import HAVE_TRITON
from .act_quant import act_quant
from .fake_quant import fake_quant_weight
from .stats import code_stats, flip_count, pack_codes

__all__ = [
    "act_quant",
    "code_stats",
    "fake_quant_weight",
    "flip_count",
    "is_available",
    "pack_codes",
]


def is_available() -> bool:
    """Whether Triton and a supported device are present for these kernels."""
    return HAVE_TRITON and torch.cuda.is_available()
