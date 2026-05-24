"""Gradient-checkpointing wrapper (CKPT path); use_reentrant=False natively forwards kwargs."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.utils.checkpoint as torch_checkpoint
from torch import nn

from axolotl.integrations.protrain.block.strategy import BlockMode
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class CheckpointedBlock(nn.Module):
    """Wrap an nn.Module so its forward activations are recomputed in backward."""

    def __init__(self, block: nn.Module) -> None:
        """Wrap ``block`` for activation checkpointing."""
        super().__init__()
        self.block = block
        self._protrain_wrapped_mode: BlockMode = BlockMode.CKPT
        # Recompute callback re-gathers chunks before backward replay.
        self._protrain_recompute_pre_hook: Callable[[], None] | None = None

    def set_recompute_pre_hook(self, hook: Callable[[], None] | None) -> None:
        """Install a callback that fires before recompute only (not initial forward)."""
        self._protrain_recompute_pre_hook = hook

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped block under torch.utils.checkpoint."""
        block = self.block
        # _run fires twice per top-level forward when activations drop: initial + recompute.
        # An outer HF gradient_checkpointing wrap inverts that order and breaks this count
        # semantically (not just inefficiently); ProTrain's validator hard-rejects the combo.
        fwd_call_count = 0

        def _run(*inner_args: Any, **inner_kwargs: Any) -> Any:
            nonlocal fwd_call_count
            fwd_call_count += 1
            # Skip hook on initial forward; fire on recompute (count >= 2).
            if fwd_call_count >= 2:
                hook = self._protrain_recompute_pre_hook
                if hook is not None:
                    hook()
            return block(*inner_args, **inner_kwargs)

        # use_reentrant=False natively threads kwargs to the wrapped callable.
        return torch_checkpoint.checkpoint(
            _run,
            *args,
            use_reentrant=False,
            **kwargs,
        )

    def extra_repr(self) -> str:
        """Return the wrapper's mode tag for ``print(model)``."""
        return f"mode={self._protrain_wrapped_mode.value}"


__all__ = ["CheckpointedBlock"]
