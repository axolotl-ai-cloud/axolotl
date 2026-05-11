"""Block-granularity forward/backward hooks for the ProTrain runtime.

``install_hooks`` attaches four hooks per transformer block:

* forward-pre hook -> :meth:`Scheduler.pre_block_forward`
* forward-post hook -> :meth:`Scheduler.post_block_forward`
* backward-pre hook -> :meth:`Scheduler.pre_block_backward`
* backward-post hook -> :meth:`Scheduler.post_block_backward`

In addition (M6C-fix-3) it attaches per-PEFT-LoRA-container forward-
and backward-pre hooks for every module returned by
:func:`_find_peft_lora_containers`. Block-level gathers are a
*superset* of the chunks any enclosed LoRA factor needs, but PEFT's
``LoraLayer.forward`` records autograd graph nodes (notably the bf16
cast in ``_cast_input_dtype``) whose shape-derivation step reads
``param.size()`` at the moment the op is constructed. If those reads
race the block-level gather (e.g. the cold path where the LoRA
factor's chunk hasn't yet been gathered before its first attribute
read in the wrapped layer's forward), autograd records the
empty-placeholder shape ``[0]`` and the matching backward fails with
``ToCopyBackward0 returned an invalid gradient at index 0 - got
[14336, 16] but expected shape compatible with [0]``. The
container-level pre-hooks defensively re-gather the LoRA factor's
chunks immediately before the PEFT layer's forward (and again before
its backward) so the param's recorded size reflects its real shape.
The fix mirrors M6C-fix-2 in ``profiler/on_demand.py``, which
installed the analogous per-LoRA-container hooks for the *profiler-
trace* path; this module closes the same gap on the runtime training
path.

M6C-fix-6 extends the per-container coverage from the pre-edge pair
to a full pre/post fwd+bwd quartet. The pre-* hooks remain the
load-bearing first re-gather; the new post-* hooks defensively
re-assert the gather BEFORE the block-level post-* hook fires its
release / reduce-and-offload. This closes the residual failure mode
from M6C-fix-5's b787acb5 diagnosis: ``RuntimeError: TBackward0
returned an invalid gradient at index 0 - got [14336, 16] but
expected shape compatible with [0]``. The ``[0]`` placeholder shape
can only be observed if ``param.data`` was rebound to
``_empty_placeholder`` between the autograd Function's construction
(forward time) and its apply (backward time). The post-forward
re-assert covers the window between the OUTER container's forward
returning and the block-level post-forward release; the post-
backward re-assert covers the window between the OUTER container's
pre-backward fire and the inner ``nn.Linear``'s ``TBackward0``
apply (which executes deep inside the OUTER's backward graph
unrolling). Together with M6C-fix-3's pre-edge hooks and M6C-fix-4's
synchronous routing through the chunk manager, every transition
window the chunk could pass through during the LoRA container's
autograd lifecycle is covered by an idempotent re-bind.

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
    """Return the chunk-id set covering ``container``'s direct + descendant params.

    The container is a PEFT-LoRA module returned by
    :func:`_find_peft_lora_containers` — typically a wrapped
    ``nn.Linear`` (``q_proj`` / ``v_proj`` / etc.) carrying
    ``lora_A`` / ``lora_B`` ``nn.ModuleDict`` children plus a
    ``base_layer`` Linear. Walks every parameter reachable from
    ``container`` and looks each up by ``id(param)`` in the chunk
    manager's ``_params_by_id`` index — the canonical reverse
    lookup the chunk manager populates at construction time.

    Notes on the lookup direction: ``ChunkManager._params_by_id`` keys
    on the *dotted parameter name as captured at chunk-manager
    construction* (i.e. before block-wrapping inserted the ``.block.``
    infix). At install_hooks time the post-wrap names look different,
    so we cannot match by name. Going via ``id(param)`` is robust
    because the wrapping does not allocate new ``Parameter`` objects
    — it merely relocates them under the wrapper module.

    Returned tuple is sorted+deduped for deterministic enumeration in
    test assertions, and constant per container (computed once at
    install_hooks time, captured by the closures returned below).
    """
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
    """Build a forward-pre hook that ensures ``chunk_ids`` are GPU-resident.

    Closure over the precomputed ``chunk_ids`` (computed once per
    container at install time) avoids walking
    ``container.parameters()`` on every forward. The scheduler's
    ``ensure_chunks_resident`` is idempotent — chunks already
    gathered by the enclosing block's pre-forward hit the
    ``_active_chunks`` fast path with a no-copy tag re-bind.
    """

    def _hook(module: nn.Module, inputs):  # noqa: ARG001
        scheduler.ensure_chunks_resident(chunk_ids)
        return None

    return _hook


def _make_lora_container_pre_backward_hook(
    scheduler: "Scheduler", chunk_ids: tuple[ChunkId, ...]
):
    """Build a backward-pre hook mirror of the forward variant.

    Backward time is symmetric: PEFT's autograd graph through the
    LoRA forward references the live ``param.size()`` at
    ``ToCopyBackward0`` apply time. The block-level
    ``pre_block_backward`` hook gathers a superset, so this is
    typically a fast-path tag re-bind — but on the cold path (e.g.
    the chunk was evicted between block-pre-bwd and the LoRA
    layer's actual backward kernel running) it is the load-bearing
    re-gather that prevents the same ``invalid gradient ... shape
    compatible with [0]`` error class fired at forward time.
    """

    def _hook(module: nn.Module, grad_output):  # noqa: ARG001
        scheduler.ensure_chunks_resident(chunk_ids)
        return None

    return _hook


def _make_lora_container_post_forward_hook(
    scheduler: "Scheduler", chunk_ids: tuple[ChunkId, ...]
):
    """Build a forward-post hook that re-asserts the gather (defensive).

    M6C-fix-6: the OUTER ``lora.Linear`` container's pre-forward
    hook calls ``ensure_chunks_resident`` synchronously. In steady
    state the inner ``lora_A`` / ``lora_B`` ``nn.Linear`` forwards
    that follow read ``self.weight`` (a Parameter whose ``.data``
    was just rebound to a real-shape view) and ``at::linear``
    records ``TBackward0`` against the real ``weight.size()``.

    This post-forward hook is a defense-in-depth idempotent re-bind:
    if some intermediate scheduler reentrancy (e.g. a cross-block
    prefetch lookahead that races the OUTER forward) NULLED the
    rebound ``param.data`` mid-forward, the post-forward re-bind
    keeps the param consistent BEFORE the block-level
    post-forward fires the actual ``offload(cid)`` release. The
    cost is one ``ensure_chunks_resident`` per container per
    forward, which on the hot path is a tag-lookup-only re-bind
    (chunks already in ``_active_chunks``).
    """

    def _hook(module: nn.Module, inputs, output):  # noqa: ARG001
        scheduler.ensure_chunks_resident(chunk_ids)
        return None

    return _hook


def _make_lora_container_post_backward_hook(
    scheduler: "Scheduler", chunk_ids: tuple[ChunkId, ...]
):
    """Build a backward-post hook that re-asserts the gather (defensive).

    M6C-fix-6: pin chunks across the OUTER ``lora.Linear``
    container's *entire* backward window (pre-backward through
    post-backward) by re-asserting ``ensure_chunks_resident`` at
    the post-backward edge. The pre-backward variant already
    rebinds ``param.data`` to the gathered buffer; this post-
    backward call defensively re-asserts the binding in case the
    block-level scheduler released the chunk via
    :meth:`Scheduler.post_block_backward` BETWEEN the OUTER
    container's pre-backward fire and the inner ``lora_A`` /
    ``lora_B`` ``nn.Linear`` ``TBackward0`` apply.

    The fix targets the M6C-fix-5 residual failure mode:
    ``RuntimeError: TBackward0 returned an invalid gradient at
    index 0 - got [14336, 16] but expected shape compatible with
    [0]``. The ``[0]`` placeholder shape can only be observed if
    ``param.data`` was rebound to ``_empty_placeholder`` between
    the autograd Function's construction (forward time) and its
    apply (backward time). With the post-forward and pre/post-
    backward defensive re-binds, every transition window the
    chunk could pass through during the OUTER container's autograd
    lifecycle is covered.

    The hook is a no-op release in itself — chunk lifetime stays
    owned by the block-level scheduler. The redundant
    ``ensure_chunks_resident`` is idempotent on the
    ``_active_chunks`` fast path.
    """

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

    # M6C-fix-3: per-PEFT-LoRA-container forward/backward pre-hooks.
    # Same root cause as M6C-fix-2 in ``profiler/on_demand.py``: PEFT's
    # ``LoraLayer.forward`` constructs autograd graph nodes (notably
    # the bf16 cast in ``_cast_input_dtype``) whose shape derivation
    # reads ``param.size()`` at op-construction time. When the LoRA
    # factor's chunk hasn't yet been gathered (cold path before the
    # block-level pre-forward hook fires, or a non-block op that
    # dereferences a LoRA factor outside its block's gather window),
    # the recorded shape is the empty placeholder ``[0]`` and backward
    # fails with ``ToCopyBackward0 returned an invalid gradient at
    # index 0 - got [...] but expected shape compatible with [0]``.
    #
    # The container detector (re-used from ``profiler/on_demand.py``)
    # returns the OUTERMOST modules that own a trainable PEFT LoRA
    # factor as a direct attribute or one-level child — typically each
    # PEFT-wrapped ``q_proj`` / ``v_proj`` etc. inside every transformer
    # block. We compute each container's chunk-id set at install time
    # via ``_container_chunk_ids`` (an ``id(param) -> ChunkId`` walk
    # through the chunk manager's reverse index — robust against the
    # ``.block.`` infix the post-wrap named_parameters paths carry)
    # and capture it in the hook closure. ``ensure_chunks_resident``
    # is idempotent: in steady state the block-level pre-forward has
    # already gathered every chunk in this set; the container hook
    # then takes the no-copy ``_active_chunks`` fast path. The cold
    # path (e.g. the very first iteration where autograd graph
    # construction races the prefetch stream) is exactly the case the
    # M6C bug report identifies, and is what this hook closes.
    #
    # Detection runs against the post-wrap model — the container
    # detector walks ``model.modules()`` and inspects each module's
    # direct + one-level-child attribute names for the PEFT name
    # tags, so the wrap-introduced ``.block.`` infix on dotted paths
    # is invisible to the detection logic.
    peft_lora_containers = _find_peft_lora_containers(model)
    if peft_lora_containers:
        # INFO (not DEBUG) so the install line surfaces in production
        # logs — this is the load-bearing wiring confirmation for
        # M6C-fix-3's per-PEFT-LoRA-container gather hooks; without it,
        # diagnosing a regression that silently disables the hook
        # registration would mean re-instrumenting the call site under
        # debug log. Mirrors the materialize_offload INFO line that
        # likewise surfaces a load-bearing one-time setup decision.
        # Updated for M6C-fix-6: now installs the full pre/post fwd+bwd
        # quartet per container (4 hooks each), not just the pre-edge
        # pair (2 hooks each).
        LOG.info(
            "install_hooks (M6C-fix-6): %d PEFT-LoRA container(s) detected; "
            "installing per-container fwd/bwd pre+post-gather hook quartet",
            len(peft_lora_containers),
        )
    for container in peft_lora_containers:
        cids = _container_chunk_ids(container, chunk_manager)
        if not cids:
            # Container's params didn't land in any chunk (e.g. the
            # LoRA factor was added after the chunk manager was
            # built). Skip — the container hook would gather nothing
            # and the bug surface doesn't exist for these params.
            continue
        # ``prepend=True`` on the pre-forward hook to mirror
        # ``profiler/on_demand.py``'s rationale: the gather must
        # precede any other registered pre-hook (notably the trace
        # driver's snapshot hook in profiler runs that re-use this
        # codepath, but kept symmetric in production for predictable
        # ordering). Backward pre-hooks default to FIFO since the
        # block-level backward-pre is the only other registrant and
        # already gathers the same chunks first.
        handles.append(
            container.register_forward_pre_hook(
                _make_lora_container_pre_forward_hook(scheduler, cids),
                prepend=True,
            )
        )
        # M6C-fix-6: per-container POST-forward hook to re-assert the
        # gather BEFORE the block-level post-forward fires its
        # ``offload(cid)`` release. Idempotent in steady state; the
        # cold-path coverage closes the failure mode where some
        # intermediate scheduler reentrancy nulled ``param.data``
        # mid-forward (between the OUTER container's pre-forward and
        # the OUTER's forward returning). See the docstring on
        # :func:`_make_lora_container_post_forward_hook` for the
        # detailed rationale.
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
        # M6C-fix-6: per-container POST-backward hook to re-assert the
        # gather across the OUTER container's full backward window —
        # the precise failure surface the M6C-fix-5 commit
        # ``b787acb5`` diagnosed (chunk gets released between the
        # OUTER ``lora.Linear`` container's post-forward and the
        # inner ``nn.Linear``'s ``TBackward0`` apply). The
        # ``register_full_backward_hook`` variant fires AFTER the
        # container's grad_input has been computed but BEFORE
        # downstream consumers may release / overwrite the chunk
        # buffer. Idempotent; same fast-path/cold-path semantics as
        # the pre-backward variant.
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
