"""Block-granularity forward/backward hooks plus per-PEFT-LoRA-container quartet hooks that re-bind chunk data across every autograd window where ``param.data`` could otherwise be observed as the empty placeholder."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from torch import nn

from axolotl.integrations.protrain.block.layout_rules import (
    discover_blocks,
    flatten_block_trees,
)
from axolotl.integrations.protrain.block.offload import OffloadedBlock
from axolotl.integrations.protrain.profiler.on_demand import (
    _find_peft_lora_containers,
)
from axolotl.integrations.protrain.types import (
    BlockId,
    BlockStrategyMap,
    ChunkId,
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


def _container_chunk_ids(
    container: nn.Module,
    chunk_manager: "ChunkManager",
) -> tuple[ChunkId, ...]:
    """Return the sorted+deduped chunk-id set covering ``container``'s subtree; lookups go via ``id(param)`` because post-wrap names differ from chunk-manager construction-time names."""
    # Reverse index: id(Parameter) -> ParamId (dotted name string).
    cm_id_to_name = {id(p): name for name, p in chunk_manager._params_by_id.items()}  # noqa: SLF001
    chunk_ids: set[ChunkId] = set()
    for param in container.parameters(recurse=True):
        cm_name = cm_id_to_name.get(id(param))
        if cm_name is None:
            # Param post-dates chunk-manager construction (e.g. an
            # adapter PEFT installed AFTER protrain_model_wrapper —
            # not the supported flow but cheap to skip defensively).
            continue
        cid = chunk_manager.layout.param_to_chunk.get(cm_name)
        if cid is None:
            continue
        chunk_ids.add(cid)
    # Sort for determinism — gather order doesn't matter (the chunk
    # manager's gather is per-chunk independent), but a stable order
    # keeps test-time enumeration reproducible.
    return tuple(sorted(chunk_ids))


def _make_lora_container_pre_forward_hook(
    scheduler: "Scheduler", chunk_ids: tuple[ChunkId, ...]
):
    """Build a forward-pre hook that gathers ``chunk_ids`` via idempotent ``ensure_chunks_resident``; chunk_ids is precomputed once per container to avoid walking parameters every forward."""

    def _hook(module: nn.Module, inputs):  # noqa: ARG001
        scheduler.ensure_chunks_resident(chunk_ids)
        return None

    return _hook


def _make_lora_container_pre_backward_hook(
    scheduler: "Scheduler", chunk_ids: tuple[ChunkId, ...]
):
    """Backward-pre mirror of the forward variant; the cold-path re-gather prevents the autograd ``shape compatible with [0]`` error when a chunk was evicted before the LoRA backward kernel runs."""

    def _hook(module: nn.Module, grad_output):  # noqa: ARG001
        scheduler.ensure_chunks_resident(chunk_ids)
        return None

    return _hook


def _make_lora_container_post_forward_hook(
    scheduler: "Scheduler", chunk_ids: tuple[ChunkId, ...]
):
    """Forward-post defensive re-bind; guarantees ``param.data`` is gathered before the block-level post-forward fires its release, even if an intermediate scheduler reentrancy nulled it mid-forward."""

    def _hook(module: nn.Module, inputs, output):  # noqa: ARG001
        scheduler.ensure_chunks_resident(chunk_ids)
        return None

    return _hook


def _make_lora_container_post_backward_hook(
    scheduler: "Scheduler", chunk_ids: tuple[ChunkId, ...]
):
    """Backward-post defensive re-bind; covers the gap between the outer container's pre-backward and the inner Linear's ``TBackward0`` apply where the block-level scheduler may have released the chunk."""

    def _hook(module: nn.Module, grad_input, grad_output):  # noqa: ARG001
        scheduler.ensure_chunks_resident(chunk_ids)
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

    # per-PEFT-LoRA-container hooks gather LoRA-factor chunks before autograd shape-derivation runs, closing the cold-path ``shape compatible with [0]`` failure that block-level hooks miss
    peft_lora_containers = _find_peft_lora_containers(model)
    if peft_lora_containers:
        # INFO so the load-bearing per-container hook install surfaces in production logs
        LOG.info(
            "install_hooks: %d PEFT-LoRA container(s) detected; "
            "installing per-container fwd/bwd pre+post-gather hook quartet",
            len(peft_lora_containers),
        )
    for container in peft_lora_containers:
        cids = _container_chunk_ids(container, chunk_manager)
        if not cids:
            # container's params post-date chunk-manager construction; nothing to gather
            continue
        # prepend=True so the gather precedes any trace-driver snapshot pre-hook that would otherwise read pre-gather state
        handles.append(
            container.register_forward_pre_hook(
                _make_lora_container_pre_forward_hook(scheduler, cids),
                prepend=True,
            )
        )
        # post-forward re-assert: closes the mid-forward param.data null window before block-level offload(cid) release
        handles.append(
            container.register_forward_hook(
                _make_lora_container_post_forward_hook(scheduler, cids)
            )
        )
        handles.append(
            container.register_full_backward_pre_hook(
                _make_lora_container_pre_backward_hook(scheduler, cids)
            )
        )
        # post-backward re-assert: pins the chunk across the gap between outer container's post-forward and inner Linear's TBackward0 apply
        handles.append(
            container.register_full_backward_hook(
                _make_lora_container_post_backward_hook(scheduler, cids)
            )
        )

    LOG.debug(
        "install_hooks: attached %d handles across %d transformer blocks "
        "(plus %d PEFT-LoRA container pre+post fwd/bwd hook quartet(s))",
        len(handles),
        len(blocks),
        len(peft_lora_containers),
    )
    return handles


def uninstall_hooks(
    handles: list["RemovableHandle"],
    model: "nn.Module | None" = None,
) -> None:
    """Remove every handle produced by :func:`install_hooks`.

    Safe to call multiple times — ``RemovableHandle.remove`` is
    idempotent in modern PyTorch.

    When ``model`` is provided, also detaches OFFLOAD-mode runtime
    references (chunk_manager / scheduler) from every
    ``OffloadedBlock`` in the discovered block forest. This mirrors
    the ``attach_runtime`` call ``install_hooks`` makes, leaving the
    model in its pre-install state with no lingering ProTrain
    runtime refs. Pre-existing callers that omit ``model`` retain
    the old hook-handle-only teardown semantics.
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

    if model is not None:
        for block in flatten_block_trees(discover_blocks(model)):
            if isinstance(block, OffloadedBlock):
                try:
                    block.detach_runtime()
                except Exception as exc:  # noqa: BLE001 — best-effort
                    LOG.warning(
                        "uninstall_hooks: OffloadedBlock.detach_runtime() failed: %s",
                        exc,
                    )


__all__ = ["install_hooks", "uninstall_hooks"]
