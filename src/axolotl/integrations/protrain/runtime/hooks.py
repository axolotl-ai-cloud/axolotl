"""Block-granularity forward/backward hooks for the ProTrain runtime.

``install_hooks`` attaches four hooks per transformer block:

* forward-pre hook -> :meth:`Scheduler.pre_block_forward`
* forward-post hook -> :meth:`Scheduler.post_block_forward`
* backward-pre hook -> :meth:`Scheduler.pre_block_backward`
* backward-post hook -> :meth:`Scheduler.post_block_backward`

The hooks operate at **block** granularity only — op-level hooks are
the profiler's job (M1). This module's contract is to wire the already-
wrapped blocks (see :mod:`axolotl.integrations.protrain.block.dispatcher`)
into the scheduler's prefetch / release / reduce-offload machine.

Ordering note: ``protrain_model_wrapper`` wraps every block *before*
installing these hooks, so the hooks attach to the post-wrap modules
(``CheckpointedBlock`` / ``SwappedBlock`` / identity). The wrapper
idempotency guarantee means a re-search at epoch boundaries can
uninstall + re-wrap + re-install without any hook-level bookkeeping.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import nn

from axolotl.integrations.protrain.block.layout_rules import (
    discover_blocks,
    flatten_block_trees,
)
from axolotl.integrations.protrain.block.offload import OffloadedBlock
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockStrategyMap,
)
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle

    from axolotl.integrations.protrain.chunk import ChunkManager
    from axolotl.integrations.protrain.runtime.scheduler import Scheduler

LOG = get_logger(__name__)


class _RecomputePreHookHandle:
    """Small removable handle for CheckpointedBlock recompute callbacks."""

    def __init__(self, module: nn.Module) -> None:
        self._module: nn.Module | None = module

    def remove(self) -> None:
        module = self._module
        if module is not None and hasattr(module, "set_recompute_pre_hook"):
            module.set_recompute_pre_hook(None)
        self._module = None


def _make_forward_pre_hook(scheduler: "Scheduler", block_id: BlockId):
    """Build a forward-pre hook bound to ``scheduler`` and ``block_id``."""

    def _hook(module: nn.Module, inputs):  # noqa: ARG001 — signature required
        scheduler.pre_block_forward(block_id)
        return None  # allow default arg flow

    return _hook


def _make_forward_post_hook(scheduler: "Scheduler", block_id: BlockId):
    """Build a forward-post hook bound to ``scheduler`` and ``block_id``."""

    def _hook(module: nn.Module, inputs, output):  # noqa: ARG001
        scheduler.post_block_forward(block_id)
        return None

    return _hook


def _make_backward_pre_hook(scheduler: "Scheduler", block_id: BlockId):
    """Build a backward-pre hook bound to ``scheduler`` and ``block_id``."""

    def _hook(module: nn.Module, grad_output):  # noqa: ARG001
        scheduler.pre_block_backward(block_id)
        return None

    return _hook


def _make_backward_post_hook(scheduler: "Scheduler", block_id: BlockId):
    """Build a backward-post hook bound to ``scheduler`` and ``block_id``."""

    def _hook(module: nn.Module, grad_input, grad_output):  # noqa: ARG001
        scheduler.post_block_backward(block_id)
        return None

    return _hook


def install_hooks(
    model: nn.Module,
    chunk_manager: "ChunkManager",
    block_map: BlockStrategyMap,
    scheduler: "Scheduler",
) -> list["RemovableHandle"]:
    """Attach the four-per-block scheduler hooks.

    The ``block_map`` parameter is accepted for API symmetry with the
    design doc but is not consulted directly — the scheduler already
    holds a reference. Keeping it in the signature lets the plugin
    (M5) compose ``install_hooks`` without reaching into the
    ``Scheduler``'s private state. The ``chunk_manager`` IS consumed
    here: ``OffloadedBlock`` wrappers need it injected via
    :meth:`OffloadedBlock.attach_runtime` so their saved-tensor pack
    hook can resolve storage pointers to chunk ids and the unpack
    hook can call ``gather_for_backward``.

    Parameters
    ----------
    model:
        The user model, post-block-wrapping. ``discover_blocks`` runs
        against this to locate the transformer-block ModuleList.
    chunk_manager:
        Runtime chunk driver. Reserved.
    block_map:
        Per-block activation mode. Reserved.
    scheduler:
        The :class:`Scheduler` instance that owns the prefetch stream
        and the per-block entry points.

    Returns
    -------
    list[RemovableHandle]
        One ``RemovableHandle`` per installed hook — pass to
        :func:`uninstall_hooks` to restore the model to its pre-install
        state.
    """
    blocks = flatten_block_trees(discover_blocks(model))

    # Fail fast if the discovered block layout disagrees with the
    # ``block_map`` the scheduler was configured with. Without this
    # guard a drift between wrapping and scheduler setup would still
    # install hooks and silently call ``Scheduler.pre/post_*`` with
    # the wrong ``BlockId``s — i.e. prefetch/release the wrong chunks
    # — instead of failing at install time.
    expected_ids = set(block_map.keys())
    actual_ids = {cast(BlockId, idx) for idx in range(len(blocks))}
    if actual_ids != expected_ids:
        missing = sorted(expected_ids - actual_ids)
        extra = sorted(actual_ids - expected_ids)
        raise ValueError(
            "install_hooks block layout mismatch: discovered "
            f"{len(blocks)} block(s) with ids {sorted(actual_ids)} but "
            f"block_map has {len(expected_ids)} id(s) {sorted(expected_ids)}; "
            f"missing from discovery: {missing}; "
            f"extra in discovery: {extra}"
        )

    handles: list["RemovableHandle"] = []
    for idx, block in enumerate(blocks):
        block_id = cast(BlockId, idx)

        handles.append(
            block.register_forward_pre_hook(_make_forward_pre_hook(scheduler, block_id))
        )
        handles.append(
            block.register_forward_hook(_make_forward_post_hook(scheduler, block_id))
        )
        # ``register_full_backward_pre_hook`` exists on nn.Module from
        # PyTorch >= 2.0. We use the "full" variant so the hook observes
        # grads to the entire block, not just the last parameter.
        handles.append(
            block.register_full_backward_pre_hook(
                _make_backward_pre_hook(scheduler, block_id)
            )
        )
        handles.append(
            block.register_full_backward_hook(
                _make_backward_post_hook(scheduler, block_id)
            )
        )
        if hasattr(block, "set_recompute_pre_hook"):
            block.set_recompute_pre_hook(
                lambda block_id=block_id: scheduler.ensure_block_resident(block_id)
            )
            handles.append(_RecomputePreHookHandle(block))  # type: ignore[arg-type]

        # Wire OFFLOAD-mode wrappers to the runtime. Mirrors the SWAP
        # wrapper path in ``api/model_wrapper.py``, but lives here so
        # plugin authors composing ``install_hooks`` directly (without
        # going through the full model wrapper) still get correctly-
        # attached OFFLOAD blocks. ``attach_runtime`` is idempotent —
        # re-calling with the same manager/scheduler is a no-op.
        if isinstance(block, OffloadedBlock):
            block.attach_runtime(chunk_manager, scheduler)

    LOG.debug(
        "install_hooks: attached %d handles across %d transformer blocks",
        len(handles),
        len(blocks),
    )
    return handles


def uninstall_hooks(handles: list["RemovableHandle"]) -> None:
    """Remove every handle produced by :func:`install_hooks`.

    Safe to call multiple times — ``RemovableHandle.remove`` is
    idempotent in modern PyTorch.
    """
    failed: list["RemovableHandle"] = []
    for h in handles:
        try:
            h.remove()
        except Exception as exc:  # noqa: BLE001 — best-effort removal
            LOG.warning("uninstall_hooks: handle.remove() failed: %s", exc)
            failed.append(h)
    # Retain handles whose .remove() raised so a future cleanup /
    # re-install pass can try again; clearing them unconditionally
    # would leak the only reference to a still-installed hook.
    handles[:] = failed


__all__ = ["install_hooks", "uninstall_hooks"]
