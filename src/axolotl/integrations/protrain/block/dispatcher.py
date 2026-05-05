"""Per-block mode dispatcher.

Takes an ``nn.Module`` plus a ``BlockMode`` and returns the wrapped
module that implements that mode. The inverse ``unwrap_block`` returns
the original block, letting callers re-wrap idempotently (rewrapping
an already-wrapped block unwraps first, then re-wraps under the new
mode).

Wrapped modules carry a ``_protrain_wrapped_mode`` attribute so that
inspection, unwrap, and re-wrap all work without needing a registry.
"""

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
    """Return the original module underneath any ProTrain wrapper.

    If ``block`` is not wrapped this is a no-op that returns ``block``
    unchanged. Raises ``StrategyError`` if the marker is present but the
    inner ``block`` attribute is missing (corrupt state).
    """
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
    """Dispatch ``block`` to the wrapper implementing ``mode``.

    - ``BlockMode.NONE`` — returns ``block`` unchanged (identity).
    - ``BlockMode.CKPT`` — wraps with ``CheckpointedBlock``.
    - ``BlockMode.SWAP`` — wraps with ``SwappedBlock``. The wrapper
      pool + swap stream are injected post-construction by the model
      wrapper via ``SwappedBlock.attach_runtime``; see ``swap.py``.

    Idempotent: if ``block`` is already wrapped, it is unwrapped first
    and then re-wrapped under ``mode``. This lets the searcher re-apply
    a new layout without needing external state.
    """
    # Unwrap first to keep the operation idempotent.
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
