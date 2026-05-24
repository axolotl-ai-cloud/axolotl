"""Block-granularity hooks + PEFT-LoRA container hooks that keep chunk data live across autograd."""

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
    BlockMode,
    BlockStrategyMap,
    ChunkId,
)
from axolotl.utils.logging import get_logger

try:
    from torch.compiler import disable as _compile_disable
except Exception:  # noqa: BLE001 — older torches lack torch.compiler.disable

    def _compile_disable(fn=None, *, recursive=True):  # noqa: ARG001
        return fn if fn is not None else (lambda f: f)


# Sentinel for external grep-based detection that the torch.compile compat layer is present.
_PROTRAIN_TORCH_COMPILE_COMPAT = 1

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

    @_compile_disable(recursive=True)
    def _hook(module: nn.Module, inputs):  # noqa: ARG001 — signature required
        scheduler.pre_block_forward(block_id)
        return None  # allow default arg flow

    return _hook


def _make_forward_post_hook(scheduler: "Scheduler", block_id: BlockId):
    """Build a forward-post hook bound to ``scheduler`` and ``block_id``."""

    @_compile_disable(recursive=True)
    def _hook(module: nn.Module, inputs, output):  # noqa: ARG001
        scheduler.post_block_forward(block_id)
        return None

    return _hook


def _make_backward_pre_hook(scheduler: "Scheduler", block_id: BlockId):
    """Build a backward-pre hook bound to ``scheduler`` and ``block_id``."""

    @_compile_disable(recursive=True)
    def _hook(module: nn.Module, grad_output):  # noqa: ARG001
        scheduler.pre_block_backward(block_id)
        return None

    return _hook


def _make_backward_post_hook(scheduler: "Scheduler", block_id: BlockId):
    """Build a backward-post hook bound to ``scheduler`` and ``block_id``."""

    @_compile_disable(recursive=True)
    def _hook(module: nn.Module, grad_input, grad_output):  # noqa: ARG001
        scheduler.post_block_backward(block_id)
        return None

    return _hook


def _container_chunk_ids(
    container: nn.Module,
    chunk_manager: "ChunkManager",
) -> tuple[ChunkId, ...]:
    """Return sorted chunk-ids covering container's subtree; id(param) lookup tolerates post-wrap rename."""
    # Reverse index: id(Parameter) -> ParamId (dotted name).
    cm_id_to_name = {id(p): name for name, p in chunk_manager._params_by_id.items()}  # noqa: SLF001
    chunk_ids: set[ChunkId] = set()
    for param in container.parameters(recurse=True):
        cm_name = cm_id_to_name.get(id(param))
        if cm_name is None:
            continue
        cid = chunk_manager.layout.param_to_chunk.get(cm_name)
        if cid is None:
            continue
        chunk_ids.add(cid)
    return tuple(sorted(chunk_ids))


def _make_lora_container_pre_forward_hook(
    scheduler: "Scheduler", chunk_ids: tuple[ChunkId, ...]
):
    """Forward-pre hook gathering precomputed chunk_ids; idempotent via ensure_chunks_resident."""

    @_compile_disable(recursive=True)
    def _hook(module: nn.Module, inputs):  # noqa: ARG001
        scheduler.ensure_chunks_resident(chunk_ids)
        return None

    return _hook


def _make_lora_container_pre_backward_hook(
    scheduler: "Scheduler", chunk_ids: tuple[ChunkId, ...]
):
    """Backward-pre re-gather; prevents autograd 'shape compatible with [0]' on evicted chunks."""

    @_compile_disable(recursive=True)
    def _hook(module: nn.Module, grad_output):  # noqa: ARG001
        scheduler.ensure_chunks_resident(chunk_ids)
        return None

    return _hook


def _make_lora_container_post_forward_hook(
    scheduler: "Scheduler", chunk_ids: tuple[ChunkId, ...]
):
    """Forward-post defensive re-bind before block-level release."""

    @_compile_disable(recursive=True)
    def _hook(module: nn.Module, inputs, output):  # noqa: ARG001
        scheduler.ensure_chunks_resident(chunk_ids)
        return None

    return _hook


def _make_lora_container_post_backward_hook(
    scheduler: "Scheduler", chunk_ids: tuple[ChunkId, ...]
):
    """Backward-post defensive re-bind across outer pre-backward → inner TBackward0 gap."""

    @_compile_disable(recursive=True)
    def _hook(module: nn.Module, grad_input, grad_output):  # noqa: ARG001
        scheduler.ensure_chunks_resident(chunk_ids)
        return None

    return _hook


def _is_runtime_inert(
    blocks: list[nn.Module],
    block_map: BlockStrategyMap,
    n_persist: int,
    N_chunk: int,
) -> bool:
    """All chunks persistent + no OFFLOAD blocks + every block mode in {NONE, CKPT} → hooks no-op."""
    if n_persist != N_chunk:
        return False
    if any(isinstance(b, OffloadedBlock) for b in blocks):
        return False
    return all(mode in (BlockMode.NONE, BlockMode.CKPT) for mode in block_map.values())


def install_hooks(
    model: nn.Module,
    chunk_manager: "ChunkManager",
    block_map: BlockStrategyMap,
    scheduler: "Scheduler",
) -> list["RemovableHandle"]:
    """Attach four-per-block scheduler hooks; wire OffloadedBlock + PEFT-LoRA containers."""
    blocks = flatten_block_trees(discover_blocks(model))

    # Fail fast on block-id drift between discovery and configured block_map.
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

    # Test stubs may omit _persistent_ids; default to None so inert check fails closed.
    persistent_ids = getattr(chunk_manager, "_persistent_ids", None)
    n_persist = len(persistent_ids) if persistent_ids is not None else -1
    N_chunk = chunk_manager.layout.N_chunk
    if _is_runtime_inert(blocks, block_map, n_persist, N_chunk):
        LOG.info(
            "ProTrain runtime is inert (n_persist == N_chunk, no offloaded blocks, "
            "all blocks NONE/CKPT). Skipping hook installation — gather/offload would "
            "early-return anyway. Expected ~20-40%% step-time reduction at bs=1."
        )
        scheduler._is_inert = True  # noqa: SLF001
        return []

    handles: list["RemovableHandle"] = []
    for idx, block in enumerate(blocks):
        block_id = cast(BlockId, idx)

        handles.append(
            block.register_forward_pre_hook(_make_forward_pre_hook(scheduler, block_id))
        )
        handles.append(
            block.register_forward_hook(_make_forward_post_hook(scheduler, block_id))
        )
        # "full" variant so hook observes grads to entire block.
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

            @_compile_disable(recursive=True)
            def _recompute_pre_hook(block_id=block_id):
                scheduler.ensure_block_resident(block_id)

            block.set_recompute_pre_hook(_recompute_pre_hook)
            handles.append(_RecomputePreHookHandle(block))  # type: ignore[arg-type]

        # OFFLOAD wrappers attach idempotently here for direct-install callers.
        if isinstance(block, OffloadedBlock):
            block.attach_runtime(chunk_manager, scheduler)

    # PEFT-LoRA container quartet hooks close cold-path autograd shape-derivation race.
    peft_lora_containers = _find_peft_lora_containers(model)
    if peft_lora_containers:
        LOG.info(
            "install_hooks: %d PEFT-LoRA container(s) detected; "
            "installing per-container fwd/bwd pre+post-gather hook quartet",
            len(peft_lora_containers),
        )
    for container in peft_lora_containers:
        cids = _container_chunk_ids(container, chunk_manager)
        if not cids:
            continue
        # prepend so gather precedes trace-driver snapshot pre-hook.
        handles.append(
            container.register_forward_pre_hook(
                _make_lora_container_pre_forward_hook(scheduler, cids),
                prepend=True,
            )
        )
        # post-forward re-assert before block-level offload release.
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
        # post-backward re-assert across outer post-forward → inner TBackward0 gap.
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
    """Remove every handle; detach OFFLOAD wrappers when model passed. Idempotent."""
    failed: list["RemovableHandle"] = []
    for h in handles:
        try:
            h.remove()
        except Exception as exc:  # noqa: BLE001 — best-effort removal
            LOG.warning("uninstall_hooks: handle.remove() failed: %s", exc)
            failed.append(h)
    # Retain failed-remove handles for future cleanup attempts.
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
