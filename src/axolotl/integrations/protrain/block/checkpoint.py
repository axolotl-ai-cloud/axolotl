"""Gradient-checkpointing wrapper for a single transformer block.

CKPT mode in the ProTrain three-way block strategy (§3.1.2). The wrapper
defers to ``torch.utils.checkpoint.checkpoint`` with ``use_reentrant=False``
so activations for the wrapped block are dropped after forward and
recomputed during backward.

Kwargs handling
---------------
HuggingFace transformer blocks take positional tensors plus keyword
arguments such as ``attention_mask``, ``position_ids``, ``past_key_value``,
``output_attentions``, ``use_cache``. With ``use_reentrant=False``,
``torch.utils.checkpoint.checkpoint`` natively forwards keyword arguments
to the wrapped callable, so we pass ``*args, **kwargs`` straight through
without a wrapping closure. This preserves the block's native call
signature.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch.utils.checkpoint as torch_checkpoint
from torch import nn

from axolotl.integrations.protrain.block.strategy import BlockMode
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class CheckpointedBlock(nn.Module):
    """Wrap an ``nn.Module`` so its forward activations are recomputed in backward.

    Marks the wrapper with ``_protrain_wrapped_mode = BlockMode.CKPT`` so the
    dispatcher can recognise and unwrap it idempotently.
    """

    def __init__(self, block: nn.Module) -> None:
        """Wrap ``block`` for activation checkpointing under ``torch.utils.checkpoint``."""
        super().__init__()
        self.block = block
        # Public marker consumed by dispatcher.unwrap_block and inspection code.
        self._protrain_wrapped_mode: BlockMode = BlockMode.CKPT
        # Optional callback installed by runtime.hooks. It re-gathers
        # this block's parameter chunks before checkpoint recompute,
        # because the recompute calls ``self.block`` directly and does
        # not pass through hooks attached to this wrapper module.
        self._protrain_recompute_pre_hook: Callable[[], None] | None = None

    def set_recompute_pre_hook(self, hook: Callable[[], None] | None) -> None:
        """Install a callback run before recompute (backward) forwards only.

        The callback is suppressed on the initial forward — the wrapper's
        forward-pre hooks already ensure block residency for that pass.
        It fires only on the recompute that ``torch.utils.checkpoint``
        triggers during backward, when the dropped activations are
        reconstructed by re-running ``self.block`` directly (bypassing
        any hooks attached to this wrapper module).
        """
        self._protrain_recompute_pre_hook = hook

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the wrapped block under ``torch.utils.checkpoint`` (activations recomputed)."""
        block = self.block
        # Per-invocation call counter captured by the ``_run`` closure.
        # ``torch.utils.checkpoint`` invokes ``_run`` twice per top-level
        # forward when activations are dropped: once during the initial
        # forward (count == 1) and once during the backward replay /
        # recompute pass (count >= 2). Keeping the counter local to this
        # ``forward()`` invocation avoids cross-talk when the same wrapped
        # block is called multiple times before backward.
        fwd_call_count = 0

        def _run(*inner_args: Any, **inner_kwargs: Any) -> Any:
            nonlocal fwd_call_count
            fwd_call_count += 1
            # Skip the hook on the initial forward (count == 1): the
            # wrapper's forward-pre hooks have already gathered this
            # block's params. Fire only on recompute (count >= 2).
            if fwd_call_count >= 2:
                hook = self._protrain_recompute_pre_hook
                if hook is not None:
                    hook()
            return block(*inner_args, **inner_kwargs)

        # ``use_reentrant=False`` natively threads kwargs to the wrapped
        # callable, so HF block kwargs (attention_mask=, position_ids=, ...)
        # flow through without manual capture.
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
