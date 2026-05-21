"""Per-block mode dispatcher; idempotent wrap/unwrap via _protrain_wrapped_mode marker."""

from __future__ import annotations

from torch import nn

from axolotl.integrations.protrain.block.checkpoint import CheckpointedBlock
from axolotl.integrations.protrain.block.offload import OffloadedBlock
from axolotl.integrations.protrain.block.strategy import BlockMode, StrategyError
from axolotl.integrations.protrain.block.swap import SwappedBlock
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


_MARKER_ATTR = "_protrain_wrapped_mode"


def _is_wrapped(block: nn.Module) -> bool:
    """True iff ``block`` was produced by a previous ``wrap_block`` call."""
    return hasattr(block, _MARKER_ATTR)


def unwrap_block(block: nn.Module) -> nn.Module:
    """Return the original module under any ProTrain wrapper; no-op if unwrapped."""
    if not _is_wrapped(block):
        return block
    inner = getattr(block, "block", None)
    if inner is None:
        raise StrategyError(
            "module has _protrain_wrapped_mode marker but no 'block' attribute; "
            "cannot unwrap"
        )
    return inner


def wrap_block(block: nn.Module, mode: BlockMode) -> nn.Module:
    """Dispatch block to NONE / CKPT / SWAP / OFFLOAD wrapper; idempotent (unwrap-then-rewrap)."""
    if _is_wrapped(block):
        block = unwrap_block(block)

    if mode is BlockMode.NONE:
        return block
    if mode is BlockMode.CKPT:
        wrapped: nn.Module = CheckpointedBlock(block)
    elif mode is BlockMode.SWAP:
        wrapped = SwappedBlock(block)
    elif mode is BlockMode.OFFLOAD:
        wrapped = OffloadedBlock(block)
    else:
        raise StrategyError(f"unknown BlockMode: {mode!r}")
    setattr(wrapped, _MARKER_ATTR, mode)
    return wrapped


__all__ = ["unwrap_block", "wrap_block"]
