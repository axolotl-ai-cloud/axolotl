"""Allocate-before-use / free-after tensor context for profiling models > device memory.

The profiler must be able to trace models whose full state (params + grads +
optimizer state + activations) doesn't fit on a single GPU. ProTrain solves
this with two coordinated mechanisms (paper §3.2):

1. **Parameter offload** — every nn.Module's directly-owned parameters live
   on pinned CPU memory between modules. A pre-forward hook gathers a
   module's own params onto GPU just before its forward; a post-forward
   hook releases them. The GPU therefore only holds *one* module's params
   at a time during the traced forward, plus whatever the running op's
   inputs/outputs require.

2. **Saved-activation spill** — ``torch.autograd.graph.saved_tensors_hooks``
   intercepts every tensor that autograd would retain for backward, copies
   it to CPU at save time, and copies it back to ``self.device`` at unpack
   time. Backward under on-demand IS supported (CPU->GPU copy in unpack
   adds ~saved_activation_bytes / pcie_bw latency to the backward pass);
   the trace driver currently passes ``include_backward=False`` when on-
   demand engages because the bwd peak still exceeds device memory for the
   target models, but the hook path is correct for callers that want to
   run backward themselves.

Together these bound peak GPU at roughly ``max_leaf_param_bytes +
activation_workspace_per_op``, which is small enough that 13B / 70B-class
models can be profiled on a 24 GB card without OOM.

The disabled fast path (``disabled=True``) is a no-op context manager —
used by the tiny-GPT2 unit tests and by the model_wrapper when the model
fits on-device with headroom (no offload needed).
"""

from __future__ import annotations

import types
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from axolotl.integrations.protrain.types import OpRecord
from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    import torch
    from torch import nn

LOG = get_logger(__name__)


def _fused_kernel_func_names() -> frozenset[str]:
    """Names of ``axolotl.kernels.lora`` apply_* functions that bypass per-Linear hooks.

    Axolotl's fused LoRA kernels are installed by
    ``axolotl/monkeypatch/lora_kernels.py`` as ``types.MethodType`` bindings
    on transformer-block submodules. Each fused entry-point reads weight
    tensors via direct attribute access (e.g. ``self.gate_proj.weight``),
    NOT by calling the wrapped ``nn.Linear``'s ``__call__`` — so the
    standard per-leaf forward-pre hook the on-demand manager registers
    never fires for those projections, and the fused matmul reads the
    empty post-spill placeholder. Detecting these names lets us install
    a container-level pre-gather hook that gathers every sub-parameter
    before the fused forward runs.

    Listed by name (not import) so a missing kernel module does not break
    on-demand for non-fused users.
    """
    return frozenset(
        {
            "apply_lora_mlp_swiglu",
            "apply_lora_mlp_geglu",
            "apply_lora_qkv",
            "apply_lora_qk",
            "apply_lora_o",
            "apply_lora_embedding",
        }
    )


def _is_fused_method(attr: Any) -> bool:
    """True iff ``attr`` is a ``types.MethodType`` bound to a fused-kernel function.

    Handles both ``mlp.forward`` (instance-level forward swap) and
    ``self_attn.apply_qkv`` / ``self_attn.apply_o`` (instance-level
    method bindings). The bound-method's ``__func__.__name__`` is the
    apply_lora_* function we registered on the module.
    """
    if not isinstance(attr, types.MethodType):
        return False
    fn = getattr(attr, "__func__", None)
    name = getattr(fn, "__name__", None)
    return name in _fused_kernel_func_names()


def _find_fused_kernel_containers(model: "nn.Module") -> "list[nn.Module]":
    """Return modules whose forward-path bypasses per-Linear gather hooks.

    A container is any ``nn.Module`` carrying at least one fused-kernel
    method binding installed by ``apply_lora_kernel_patches``:

    * ``mlp.forward`` swapped to ``apply_lora_mlp_swiglu`` / ``..._geglu``
      (the swiglu/geglu kernel reads ``gate_proj``/``up_proj``/``down_proj``
      weight refs directly).
    * ``self_attn.apply_qkv`` swapped to ``apply_lora_qkv`` / ``apply_lora_qk``
      (the QKV kernel reads ``q_proj``/``k_proj``/``v_proj`` weight refs
      directly when ``self_attn.forward`` later calls ``self.apply_qkv``).
    * ``self_attn.apply_o`` swapped to ``apply_lora_o`` (analogous, for
      the output projection invoked from the patched attention forward).
    * ``embed_tokens.forward`` swapped to ``apply_lora_embedding`` (reads
      the embed weight + lora_embedding_A/B sub-Parameter refs directly).

    Returned in deterministic ``model.modules()`` order so test assertions
    can rely on a stable enumeration. Empty when no fused-kernel
    monkey-patch has been applied — the on-demand manager then falls back
    to its per-Linear-only hook path with no behavior change.
    """
    out: list["nn.Module"] = []
    for sub in model.modules():
        for attr_name in ("forward", "apply_qkv", "apply_o"):
            attr = getattr(sub, attr_name, None)
            if _is_fused_method(attr):
                out.append(sub)
                break
    return out


@dataclass
class _ParamSpill:
    """Bookkeeping for one parameter that's been spilled to CPU.

    Two original-device cases:

    * GPU-resident param (typical Axolotl path): we copy GPU→CPU at
      ``__enter__`` and DROP the reference to the original GPU tensor so
      the caching allocator can reclaim its storage (``original_data`` is
      ``None``). At ``__exit__`` we re-allocate a fresh tensor on
      ``original_device`` and copy ``cpu_storage`` back. Parameter
      identity (``id(param)``) is preserved; optimizer state keyed on
      ``id(param)`` (the PyTorch convention) survives the round trip.

    * CPU-resident param (paper's intent — model too big for GPU): no
      copy needed; ``cpu_storage`` IS the original tensor (pinned in
      place if possible). ``original_data`` is also ``None`` here. The
      pre-gather hook copies to the target device on demand.

    The ``original_data`` field stays in the dataclass for forward-compat
    with any caller that still populates it (the restore paths still
    handle the legacy retain-storage case), but the GPU spill path no
    longer sets it.
    """

    param: Any  # torch.nn.Parameter — Any keeps import light
    cpu_storage: Any  # torch.Tensor on CPU (pinned if possible)
    original_device: Any  # torch.device the param was on at __enter__
    original_data: Any  # legacy: GPU tensor at __enter__; None on the new path


class OnDemandTensorMgr:
    """Context manager that materializes each leaf's params just-in-time.

    Disabled fast path
    ------------------
    When ``disabled=True``, the context manager is a no-op and the profiler
    runs a normal forward/backward pass. This is the right choice when the
    model fits on-device with headroom — pure profiling cost, zero spill
    overhead. The model_wrapper uses this path for ~7B-class models on a
    24 GB card.

    Enabled mode (replay-equivalent)
    --------------------------------
    On ``__enter__``:

    * Every parameter is detached and moved to pinned CPU memory (best-effort
      pinning; falls back to pageable if pinning fails). The Parameter's
      ``.data`` slot is replaced with an empty GPU tensor of matching dtype.
    * A pre-forward hook is registered on every nn.Module to copy that
      module's *direct* parameters (``parameters(recurse=False)``) from CPU
      to GPU, replacing the empty placeholder.
    * A post-forward hook on every module replaces those parameters' ``.data``
      with empty placeholders again, releasing the GPU storage. The freshly-
      gathered GPU tensor remains alive only as long as the autograd graph
      (or downstream ops) hold a reference to it.
    * ``torch.autograd.graph.saved_tensors_hooks`` is entered for the duration
      of the traced forward. Every tensor autograd would retain for backward
      is copied to CPU at save time. This is the activation-spill half of
      the paper's allocate-before-use / free-after-use scheme; it makes
      ``post_forward``'s ``p.data = empty()`` actually reclaim GPU memory
      (otherwise the saved-for-backward slot would pin the gathered tensor).

    On ``__exit__``: hooks are removed; every parameter is restored to its
    original device (using the original GPU storage that the optimizer's
    state already references via ``id(param)``).

    Notes
    -----
    * Buffers (BatchNorm running stats, position-embedding buffers, etc.)
      are NOT offloaded — they're typically small (<<1% of param state) and
      offloading them complicates the BatchNorm fastpath. If a future model
      shows non-trivial buffer footprint the same hook structure can be
      extended.
    * The ``allocate_inputs`` / ``free_after`` methods on this class are
      kept for API compatibility with the original M1 scaffold (the
      profiler driver does not call them — hook-based gathering replaces
      that path) and to keep ``test_on_demand_disabled_fast_path`` green.
    """

    def __init__(
        self,
        device: "torch.device | str | int | None" = None,
        *,
        disabled: bool = False,
        model: "nn.Module | None" = None,
    ) -> None:
        """Configure target device and disabled-mode flag; defer spill until ``__enter__``."""
        self.device = device
        self.disabled = disabled
        self.model = model
        self._spills: dict[int, _ParamSpill] = {}
        # Active per-param gather ref-count, keyed on ``id(param)``.
        # Tied ``Parameter`` objects are registered on multiple owning
        # modules; pre/post hooks therefore fire once per owner. Without
        # this counter, an inner module's post-release would clear the
        # tied param's ``.data`` while the outer module still needs it.
        # ``_pre_gather`` increments before re-gathering; ``_post_release``
        # only resets the placeholder when the count drops to 0.
        self._active_param_users: dict[int, int] = {}
        self._handles: list[Any] = []
        self._sthook_ctx: Any = None
        self._entered = False
        self._n_pin_failures = 0
        # Populated by ``__enter__`` after fused-kernel detection. Tests
        # may inspect this to verify per-container hook installation.
        self._fused_containers: list["nn.Module"] = []

    # ---- context-manager protocol --------------------------------------

    def __enter__(self) -> "OnDemandTensorMgr":
        """Spill parameters to pinned CPU and install the gather/spill hooks."""
        if self.disabled:
            self._entered = True
            return self
        if self.model is None:
            raise ValueError(
                "OnDemandTensorMgr enabled mode requires a model. Pass "
                "model=... to __init__, or set disabled=True for the no-op "
                "fast path."
            )

        import torch

        # If no explicit device was provided, infer from the model's own
        # parameter placement first (so multi-GPU / non-default-CUDA-device
        # callers don't silently get cuda:current_device when their model
        # lives on a different card), then fall back to the active CUDA
        # device. Without this the unpack hook hits its
        # ``self.device is None`` early-return on the first saved
        # activation and backward fails the moment it touches a CPU
        # tensor on a CUDA grad path.
        if self.device is None:
            model_device = self._infer_model_device()
            if model_device is not None and model_device.type == "cuda":
                self.device = model_device
            elif torch.cuda.is_available():
                self.device = torch.device("cuda", torch.cuda.current_device())

        # Normalize self.device once: ``torch.device(0)`` is invalid in
        # PyTorch 2.6 — bare ints must go through ``torch.device("cuda", n)``.
        # Also fold ``str`` and existing ``torch.device`` into the same form
        # so all downstream consumers (_gather_target_device, _unpack_hook)
        # can rely on ``self.device`` being a ``torch.device`` or ``None``.
        if self.device is not None:
            self.device = self._normalize_device(self.device)
        # Annotate the local explicitly so mypy can narrow on
        # ``target_device.type == "cuda"`` below — ``self.device`` retains
        # a wider union type from the dataclass field.
        target_device: torch.device | None = self.device

        # 1. Spill every parameter to pinned CPU; replace .data with empty.
        # 2. Install module-level pre/post-forward hooks.
        # 3. Enter saved_tensors_hooks for activation spill.
        # If ANY of these raises (e.g. OOM during GPU->CPU copy of param N),
        # Python does NOT call ``__exit__`` because we never finished entering.
        # Wrap the entire setup in try/except: on failure, undo everything
        # we've already done (restore spilled params, remove hooks, exit
        # saved_tensors_hooks if entered) so the model is left in its
        # original state, then re-raise.
        try:
            for _name, param in self.model.named_parameters():
                self._spill_param_to_cpu(param, target_device)

            # Enabled mode only spills/gathers parameters. Direct buffers
            # that stay on CPU while inputs/params are on CUDA become a
            # device-mismatch footgun in ``forward``. Fail fast with an
            # actionable message rather than letting backward crash with
            # a confusing secondary device-mismatch downstream. Extending
            # the spill hooks to cover buffers is tracked separately;
            # until then, the contract is "no CPU-resident buffers when
            # target is CUDA".
            if target_device is not None:
                for buffer_name, buffer in self.model.named_buffers():
                    # Strict device equality (not just CPU vs CUDA) so a
                    # buffer pinned to a DIFFERENT CUDA index than the
                    # target (e.g. caller dispatches to ``cuda:0`` but
                    # ``register_buffer`` left this one on ``cuda:1``)
                    # also fails fast — otherwise the device mismatch
                    # would surface deep inside a forward kernel as an
                    # opaque "expected all tensors to be on the same
                    # device" runtime error.
                    if getattr(buffer, "device", None) != target_device:
                        raise RuntimeError(
                            f"OnDemandTensorMgr: buffer {buffer_name!r} on "
                            f"{getattr(buffer, 'device', None)!r} is not on "
                            f"target_device {target_device!r}. Move the buffer "
                            "to the target device or extend the spill hooks "
                            "to cover buffers."
                        )

            for sub in self.model.modules():
                # ``prepend=True`` on pre-hooks: the trace driver registers its
                # own pre_forward (and pre_backward) hooks BEFORE we enter this
                # context. PyTorch fires forward_pre hooks in registration
                # order, so without ``prepend`` the trace's snapshot of
                # allocated_before would be taken BEFORE our gather, and
                # ``intra_op_delta = peak - allocated_before`` would absorb
                # the per-leaf gather bytes for every op. By prepending, our
                # gather fires FIRST; the trace's allocated_before then
                # already includes the gathered param, and intra_op_delta
                # captures only workspace + output (the cost model's
                # peak-reconstruction expects exactly that).
                self._handles.append(
                    sub.register_forward_pre_hook(self._pre_gather, prepend=True)
                )
                # Post-release stays FIFO: it must fire AFTER the trace's
                # post_forward measures peak/end, otherwise we'd release
                # mid-measurement.
                self._handles.append(sub.register_forward_hook(self._post_release))
                # Backward path: re-gather params before each module's bwd
                # and release them after. Forward-only callers pay nothing
                # (the hooks never fire). Backward callers pay one extra
                # H2D copy of the param + one D2H release per module per
                # backward pass — the same per-module cost the forward
                # path already pays. Same ordering rationale: prepend the
                # pre-gather, FIFO the post-release.
                self._handles.append(
                    sub.register_full_backward_pre_hook(
                        self._pre_gather_bwd, prepend=True
                    )
                )
                self._handles.append(
                    sub.register_full_backward_hook(self._post_release_bwd)
                )

            # M1: container-level gather/release for fused-kernel modules.
            # When Axolotl's fused LoRA kernels are active, the host
            # module's forward (mlp / self_attn / embed_tokens) reads
            # child Linear weights via direct attribute access and never
            # invokes the children's ``__call__`` — the per-Linear
            # pre-hooks above therefore don't fire and the matmul reads
            # the empty placeholder. Detect those containers and install
            # a pre-/post-forward hook pair that gathers every sub-param
            # before the patched forward runs and releases after. The
            # ref-counter in ``_pre_gather`` makes this safe even if any
            # nested per-Linear hook does fire (it just bumps the count).
            #
            # ``prepend=True`` on pre: same rationale as the per-Linear
            # path — container gather must precede the trace driver's
            # snapshot so ``intra_op_delta`` doesn't absorb the gather
            # bytes. Post-release stays FIFO so the trace's
            # ``post_forward`` peak read happens before we release.
            self._fused_containers = _find_fused_kernel_containers(self.model)
            if self._fused_containers:
                LOG.debug(
                    "OnDemandTensorMgr: %d fused-kernel container(s) "
                    "detected; installing per-container gather hooks",
                    len(self._fused_containers),
                )
            for container in self._fused_containers:
                self._handles.append(
                    container.register_forward_pre_hook(
                        self._pre_gather_subtree, prepend=True
                    )
                )
                self._handles.append(
                    container.register_forward_hook(self._post_release_subtree)
                )
                # Backward hooks: the fused autograd Function (LoRA_MLP /
                # LoRA_QKV / LoRA_O) stores raw weight Tensor refs as a
                # plain Python attribute on ``ctx`` (e.g. ``ctx.weights``,
                # not ``ctx.save_for_backward``), so the saved-tensors
                # pack/unpack path does NOT spill them. By backward time
                # the forward post-release has reset every base
                # ``param.data`` to a length-0 placeholder, and the
                # autograd backward's matmul against ``ctx.weights[i]``
                # raises the same ``size mismatch ... vec (0)`` the M0
                # spike captured — but firing in ``LoRA_MLP.backward``
                # instead of forward (the fix's forward-only first cut
                # got the trace forward past the failure but tripped on
                # the backward equivalent during the trace's
                # ``loss.backward()`` call). Re-gathering the container's
                # subtree before its backward enters, then releasing
                # after, makes the fused autograd Function's backward
                # see real weights again. Symmetric with the forward pair.
                self._handles.append(
                    container.register_full_backward_pre_hook(
                        self._pre_gather_subtree_bwd, prepend=True
                    )
                )
                self._handles.append(
                    container.register_full_backward_hook(
                        self._post_release_subtree_bwd
                    )
                )

            # Saved-for-backward tensors spill to CPU. Without this, autograd
            # would keep the gathered GPU param alive via the saved-for-
            # backward slot of the linear's grad_fn, defeating post_release.
            self._sthook_ctx = torch.autograd.graph.saved_tensors_hooks(
                self._pack_hook, self._unpack_hook
            )
            self._sthook_ctx.__enter__()
        except BaseException:
            # Mirror __exit__'s teardown path so partial setup leaves no
            # wedged params with empty .data slots.
            self._restore_after_partial_setup()
            raise

        if self._n_pin_failures:
            LOG.debug(
                "OnDemandTensorMgr: %d params couldn't be pinned (using "
                "pageable CPU); H2D copies will be synchronous. Trace will "
                "still complete; runtime per copy ~2x slower.",
                self._n_pin_failures,
            )

        self._entered = True
        return self

    def _restore_after_partial_setup(self) -> None:
        """Undo whatever portion of __enter__ succeeded.

        Mirrors __exit__'s teardown but is callable from a partially-
        constructed enabled-mode state (some params spilled, some hooks
        registered, saved_tensors_hooks possibly entered). Best-effort:
        every step is independently try/except'd because we're already
        on an exception path and must not mask the original failure.
        """
        # Remove any hooks that were registered.
        for h in self._handles:
            try:
                h.remove()
            except Exception as exc:  # noqa: BLE001 - defensive
                LOG.debug(
                    "OnDemandTensorMgr: hook removal failed during partial-setup unwind (%s)",
                    exc,
                )
        self._handles.clear()

        # Exit saved_tensors_hooks if it was entered.
        if self._sthook_ctx is not None:
            try:
                self._sthook_ctx.__exit__(None, None, None)
            except Exception as exc:  # noqa: BLE001 - defensive
                LOG.debug(
                    "OnDemandTensorMgr: saved_tensors_hooks unwind failed during partial-setup (%s)",
                    exc,
                )
            self._sthook_ctx = None

        # Restore every already-spilled param using __exit__'s logic.
        try:
            import torch
        except Exception:  # noqa: BLE001 - defensive (torch import never fails in practice)
            torch = None  # type: ignore[assignment]

        for spill in self._spills.values():
            try:
                if spill.original_data is not None:
                    # Legacy retain-storage path (kept for forward-compat;
                    # the GPU spill no longer populates original_data).
                    spill.original_data.copy_(
                        spill.cpu_storage.to(
                            spill.original_data.device, non_blocking=True
                        )
                    )
                    spill.param.data = spill.original_data
                elif getattr(spill.original_device, "type", None) == "cuda":
                    # GPU-origin without retained storage — allocate a
                    # fresh tensor on the original device and copy from
                    # the CPU spill. ``id(param)`` is preserved.
                    spill.param.data = spill.cpu_storage.to(
                        spill.original_device, non_blocking=True
                    )
                else:
                    # CPU-original: cpu_storage IS the original tensor.
                    spill.param.data = spill.cpu_storage
                # Mirror __exit__'s grad restore: if a grad was moved to the
                # gather device during partial setup, move it back so the
                # caller doesn't see a param/grad device mismatch on the
                # exception path. Unlikely to fire (no backward has run by
                # the time setup unwinds), but symmetric with __exit__.
                if (
                    spill.param.grad is not None
                    and spill.param.grad.device != spill.original_device
                ):
                    spill.param.grad = spill.param.grad.to(
                        spill.original_device, non_blocking=True
                    )
            except Exception as _e:  # noqa: BLE001 - defensive
                LOG.warning(
                    "OnDemandTensorMgr: failed to restore param to %s during "
                    "partial-setup unwind (%s); param may be left wedged",
                    spill.original_device,
                    _e,
                )
        if torch is not None and torch.cuda.is_available():
            # Synchronize each unique CUDA target the restore loop wrote
            # to. Bare ``torch.cuda.synchronize()`` only waits on the
            # current device — non_blocking copies queued to other
            # devices (cuda:1+ on multi-GPU hosts) would still be in
            # flight when this method returns (CR 3191XXXXXX).
            cuda_targets = {
                spill.original_device
                for spill in self._spills.values()
                if getattr(spill.original_device, "type", None) == "cuda"
            }
            for dev in cuda_targets:
                try:
                    torch.cuda.synchronize(device=dev)
                except Exception as _e:  # noqa: BLE001 - defensive
                    LOG.debug(
                        "OnDemandTensorMgr: synchronize(device=%s) failed during "
                        "partial-setup unwind (%s)",
                        dev,
                        _e,
                    )
        self._spills.clear()
        self._active_param_users.clear()
        self._fused_containers = []

    def __exit__(self, exc_type, exc, tb) -> None:
        """Remove hooks and restore parameters from their pinned-CPU spill copies."""
        self._entered = False
        if self.disabled:
            return

        # Remove hooks first so partial forward calls during exit unwinding
        # don't try to gather params that are mid-restore. NOTE: rebind
        # ``except`` to ``_e`` (not ``exc``) because Python 3 deletes the
        # except-binding after the block exits — a name collision with the
        # ``exc`` parameter would silently delete the original and the
        # later ``_sthook_ctx.__exit__(exc_type, exc, tb)`` call would
        # raise ``NameError`` if any hook removal failed (CR 3191882429).
        for h in self._handles:
            try:
                h.remove()
            except Exception as _e:  # noqa: BLE001 - defensive
                LOG.debug("OnDemandTensorMgr: hook removal failed during exit (%s)", _e)
        self._handles.clear()

        # Exit saved_tensors_hooks BEFORE restoring params — any in-flight
        # backward has already completed by this point (run_trace synchs).
        if self._sthook_ctx is not None:
            try:
                self._sthook_ctx.__exit__(exc_type, exc, tb)
            except Exception as _e:  # noqa: BLE001 - defensive
                LOG.debug("saved_tensors_hooks exit raised: %s", _e)
            self._sthook_ctx = None

        # Restore every parameter back to its original location.
        # GPU-original: copy CPU contents back into the *original* GPU
        # tensor (preserving identity for the optimizer's state slots),
        # then point param.data at it. CPU-original: just restore the
        # original CPU tensor.
        import torch

        for spill in self._spills.values():
            try:
                if spill.original_data is not None:
                    # Legacy retain-storage path (kept for forward-compat;
                    # the GPU spill no longer populates original_data).
                    spill.original_data.copy_(
                        spill.cpu_storage.to(
                            spill.original_data.device, non_blocking=True
                        )
                    )
                    spill.param.data = spill.original_data
                elif getattr(spill.original_device, "type", None) == "cuda":
                    # GPU-origin without retained storage — allocate a
                    # fresh tensor on the original device and copy from
                    # the CPU spill. ``id(param)`` is preserved across
                    # the round trip; ``param.data_ptr()`` may differ.
                    # Optimizer state keys on ``id(param)`` (PyTorch
                    # convention), so this is safe.
                    spill.param.data = spill.cpu_storage.to(
                        spill.original_device, non_blocking=True
                    )
                else:
                    # CPU-original — cpu_storage is the original tensor.
                    spill.param.data = spill.cpu_storage
                # Grad may have been computed (or moved) on the gather
                # device while the param was spilled. If it's not on the
                # param's original device, the next optimizer step / CPU-
                # side use hits a device mismatch. Move it back.
                if (
                    spill.param.grad is not None
                    and spill.param.grad.device != spill.original_device
                ):
                    spill.param.grad = spill.param.grad.to(
                        spill.original_device, non_blocking=True
                    )
            except Exception as _e:  # noqa: BLE001 - defensive
                LOG.warning(
                    "OnDemandTensorMgr: failed to restore param to %s (%s); "
                    "leaving on CPU storage",
                    spill.original_device,
                    _e,
                )
                # Make the "leaving on CPU storage" claim real: point
                # ``param.data`` at the always-valid CPU spill copy and
                # move any grad to CPU so the caller doesn't see a
                # placeholder/transient tensor or a device-mismatched
                # grad on the failure path (CR 3191961003).
                spill.param.data = spill.cpu_storage
                if (
                    spill.param.grad is not None
                    and getattr(spill.param.grad.device, "type", None) != "cpu"
                ):
                    spill.param.grad = spill.param.grad.to("cpu", non_blocking=True)
        # Synchronize each unique CUDA target the restore loop wrote
        # to. Bare ``torch.cuda.synchronize()`` only waits on the
        # current device — non_blocking copies queued to other devices
        # (cuda:1+ on multi-GPU hosts) would still be in flight when
        # ``__exit__`` returns (CR 3191XXXXXX).
        if torch.cuda.is_available():
            cuda_targets = {
                spill.original_device
                for spill in self._spills.values()
                if getattr(spill.original_device, "type", None) == "cuda"
            }
            for dev in cuda_targets:
                try:
                    torch.cuda.synchronize(device=dev)
                except Exception as _e:  # noqa: BLE001 - defensive
                    LOG.debug(
                        "OnDemandTensorMgr: synchronize(device=%s) failed during exit (%s)",
                        dev,
                        _e,
                    )
        self._spills.clear()
        self._active_param_users.clear()
        self._fused_containers = []

    # ---- spill / restore helpers ---------------------------------------

    def _spill_param_to_cpu(
        self, param: Any, target_device: "torch.device | None"
    ) -> None:
        """Move ``param`` to pinned CPU storage; leave a placeholder in .data.

        Handles both GPU-resident (copy GPU→CPU, replace .data with empty)
        and CPU-resident (use param's existing tensor, pin if possible) cases.
        """
        import torch

        # Tied/shared ``Parameter`` objects can be reached via multiple
        # module paths; if we already spilled this exact object, return
        # early. A second pass would see ``.data`` already replaced with
        # the empty placeholder (numel==0), and would clobber the valid
        # ``cpu_storage`` recorded in ``self._spills`` with that placeholder.
        if id(param) in self._spills:
            return

        original_device = param.device

        if original_device.type == "cpu":
            # CPU-resident: capture the original tensor first so restore can
            # always recover it, then attempt to pin a (possibly new) copy
            # for async H2D in pre-gather. pin_memory() returns a NEW pinned
            # tensor on success (only returns self if already pinned), so we
            # must preserve the original reference separately — otherwise
            # tied-weight / shared-storage relationships break on restore.
            original_data = param.data
            try:
                pinned = original_data.pin_memory()
                cpu_storage = pinned
            except Exception:  # noqa: BLE001 - pinning is best-effort
                cpu_storage = original_data
                self._n_pin_failures += 1
            # If pin_memory returned self (already-pinned input), the two
            # references alias the same tensor; restore via cpu_storage path
            # is sufficient. Only set original_data when pinning produced a
            # distinct tensor that would otherwise replace the original.
            spill_original = original_data if cpu_storage is not original_data else None
            self._spills[id(param)] = _ParamSpill(
                param=param,
                cpu_storage=cpu_storage,
                original_device=original_device,
                original_data=spill_original,
            )
            return

        # GPU-resident: copy GPU→CPU, then drop our reference to the
        # original GPU tensor so the caching allocator can actually
        # reclaim its storage. ``original_data=None`` flags the GPU-origin
        # branch in the restore paths, which allocate a fresh tensor on
        # ``original_device`` and copy ``cpu_storage`` back. Parameter
        # identity (``id(param)``) is preserved across the round trip;
        # ``param.data_ptr()`` may differ post-restore. Optimizer state
        # keys on ``id(param)`` (PyTorch convention), so this is safe.
        #
        # Fix for CR 3192478323 / 3192535995 — the previous code retained
        # ``original_data`` for the whole context, which kept the
        # original GPU storage live and defeated the spill: peak memory
        # stayed inflated for GPU-resident models. Releasing the storage
        # here is what actually buys the paper's "model > device memory"
        # guarantee.
        try:
            cpu_storage = param.data.detach().to("cpu", copy=True)
            try:
                cpu_storage = cpu_storage.pin_memory()
            except Exception:  # noqa: BLE001 - pinning is best-effort
                self._n_pin_failures += 1
        except Exception as exc:  # noqa: BLE001 - defensive
            # Spill failed: the param is still on ``original_device``.
            # If the manager's gather destination differs from the
            # param's current device, leaving the param here is a
            # silent correctness bug — subsequent gather calls assume
            # the spill happened and will fetch a stale tensor on the
            # wrong device. Fail fast so the caller can recover.
            #
            # When ``target_device == original_device`` (in-place
            # pinning case where the spill is purely a pinning
            # optimization), best-effort warn-and-return preserves the
            # legacy behavior — the param ends up where it would have
            # been gathered anyway, just unpinned.
            if target_device is not None and target_device != original_device:
                LOG.error(
                    "OnDemandTensorMgr: failed to spill param to CPU (%s); "
                    "param is on %s but gather target is %s — propagating "
                    "to avoid a silent device-mismatch downstream.",
                    exc,
                    original_device,
                    target_device,
                )
                raise
            LOG.warning(
                "OnDemandTensorMgr: failed to spill param to CPU (%s); "
                "leaving on GPU. Profile peak will be inflated for this param.",
                exc,
            )
            return

        # Capture dtype before reassigning ``param.data`` — once the
        # placeholder is in place the original tensor is unreachable.
        orig_dtype = param.data.dtype
        placeholder = torch.empty(0, dtype=orig_dtype, device=original_device)
        param.data = placeholder
        self._spills[id(param)] = _ParamSpill(
            param=param,
            cpu_storage=cpu_storage,
            original_device=original_device,
            original_data=None,  # GPU storage released; restore re-allocates
        )

    # ---- module-level gather/release hooks -----------------------------

    @staticmethod
    def _normalize_device(device: "torch.device | str | int") -> "torch.device":
        """Normalize a device-like value to a ``torch.device``.

        ``torch.device(0)`` raises in PyTorch 2.6 (a bare int is not a
        valid single-arg constructor). Funnel ints through
        ``torch.device("cuda", index)`` and pass strings / existing
        ``torch.device`` through unchanged.
        """
        import torch

        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            return torch.device("cuda", device)
        return torch.device(device)

    def _infer_model_device(self) -> "torch.device | None":
        """Best-effort model-device inference for default target alignment.

        Returns the device of the first parameter we can find, falling
        back to the first buffer if the model has no parameters but does
        have CUDA buffers (so callers like ``_unpack_hook`` don't end up
        restoring activations to ``cuda:current_device`` on a non-default
        rank). Returns ``None`` if both iterations are empty or attribute
        access fails. Used only to pick a sensible default when the
        caller did not supply ``device=``; explicit user input always wins.
        """
        if self.model is None:
            return None
        try:
            for param in self.model.parameters():
                return param.device
            for buffer in self.model.buffers():
                return buffer.device
        except Exception:  # noqa: BLE001 - defensive
            return None
        return None

    def _gather_target_device(self) -> "torch.device | None":
        """Resolve the target device for gathered params.

        Falls back to the param's original device if the manager wasn't
        constructed with an explicit ``device``. ``self.device`` is
        already normalized to a ``torch.device`` (or ``None``) by
        ``__enter__`` — but if the manager is invoked outside the
        ``with`` block (e.g. by callers that drive hooks manually), or
        was never entered, ``self.device`` may still be a raw
        ``str``/``int``. Normalize defensively.
        """
        if self.device is None:
            return None
        import torch

        if isinstance(self.device, torch.device):
            return self.device
        return self._normalize_device(self.device)

    def _pre_gather(self, module: "nn.Module", inputs: Any) -> None:
        """Copy the module's *direct* params from CPU to target_device before forward.

        Tied params: hooks fire once per owning module. The first
        owner's pre-hook actually gathers; nested owners just bump the
        ref-count so the matching post-release defers placeholder reset
        until every owner has finished. See ``_active_param_users``.
        """
        target = self._gather_target_device()
        for param in module.parameters(recurse=False):
            spill = self._spills.get(id(param))
            if spill is None:
                continue
            pid = id(param)
            users = self._active_param_users.get(pid, 0)
            if users > 0:
                # Already gathered by an outer owner; just increment.
                self._active_param_users[pid] = users + 1
                continue
            dest = target if target is not None else spill.original_device
            try:
                gathered = spill.cpu_storage.to(dest, non_blocking=True)
                param.data = gathered
            except Exception as exc:  # noqa: BLE001 - defensive
                LOG.warning(
                    "OnDemandTensorMgr pre-gather failed (%s); falling back "
                    "to original data — peak may inflate for this op.",
                    exc,
                )
                if (
                    spill.original_data is not None
                    and spill.original_data.device == dest
                ):
                    param.data = spill.original_data
                else:
                    # Either CPU-original, OR a cross-device fallback where
                    # original_data lives on a different device than the
                    # current gather target. Both cases would leave a
                    # device-mismatched weight in place that would fail with
                    # a confusing secondary device-mismatch on the next op,
                    # hiding the real gather error/OOM. Surface the real
                    # cause (CR 3191961010).
                    raise
            self._active_param_users[pid] = 1

    def _post_release(self, module: "nn.Module", inputs: Any, output: Any) -> None:
        """Replace the module's *direct* params with empty placeholders.

        Tied params: only the OUTERMOST owner's post-release actually
        clears ``.data``. Inner owners decrement the ref-count and
        return — clearing while an outer owner still needs the param
        would leave an empty placeholder for the remaining ops.
        """
        import torch

        target = self._gather_target_device()
        for param in module.parameters(recurse=False):
            spill = self._spills.get(id(param))
            if spill is None:
                continue
            pid = id(param)
            users = self._active_param_users.get(pid, 0)
            if users > 1:
                # Outer owner(s) still need the gathered weight live.
                self._active_param_users[pid] = users - 1
                continue
            # Last (or only) owner: drop the entry and reset the placeholder.
            self._active_param_users.pop(pid, None)
            dest = target if target is not None else spill.original_device
            try:
                placeholder = torch.empty(0, dtype=param.dtype, device=dest)
                param.data = placeholder
            except Exception as exc:  # noqa: BLE001 - defensive
                LOG.debug("OnDemandTensorMgr post-release no-op (%s)", exc)

    def _pre_gather_subtree(self, module: "nn.Module", inputs: Any) -> None:
        """Container-level pre-gather for fused-kernel modules (M1).

        Walks every submodule under ``module`` and runs the standard
        ``_pre_gather`` over each so that *all* parameters owned by the
        fused container (its own + every descendant's) are GPU-resident
        for the duration of the patched forward.

        Why this is needed: Axolotl's fused LoRA kernels swap the host
        module's ``forward`` (or ``apply_qkv``/``apply_o`` method) with
        an entrypoint that reads child ``nn.Linear`` weight tensors via
        direct attribute access (``self.gate_proj.weight``). The per-
        Linear pre-gather hook therefore never fires for those leaves
        during the fused matmul, and the kernel reads the empty post-
        spill placeholder — the failure mode the M0 spike reproduced
        as ``RuntimeError: size mismatch ... vec (0)``. Container-level
        gathering covers every leaf the fused kernel might touch in one
        pre-forward pass; the per-Linear ref-counter (``_active_param_users``)
        keeps re-entrant per-Linear hooks safe even when both fire.

        Memory trade-off: a Llama transformer block's MLP container is
        ~135 MB fp16 (3 * gate/up/down at hidden=4096 -> 4096*14336*2 B);
        the self_attn container is ~67 MB; the embedding is ~525 MB on
        Llama-3-8B (vocab=128256 * hidden=4096 * 2 B). Forward peak
        rises by at most one container's worth of params relative to
        the per-leaf-only path. Documented in phase2.md §M1.
        """
        for sub in module.modules():
            self._pre_gather(sub, inputs)

    def _post_release_subtree(
        self, module: "nn.Module", inputs: Any, output: Any
    ) -> None:
        """Container-level post-release: mirror of ``_pre_gather_subtree``.

        Walks the same submodule set in reverse order so the active-user
        ref-counts that ``_pre_gather_subtree`` incremented unwind in
        the opposite order they were taken — matches the LIFO ownership
        pattern the per-Linear path already relies on for tied params.
        """
        for sub in reversed(list(module.modules())):
            self._post_release(sub, inputs, output)

    def _pre_gather_subtree_bwd(self, module: "nn.Module", grad_output: Any) -> None:
        """Backward-pre hook: gather every sub-param before container bwd.

        Mirrors ``_pre_gather_subtree`` for the backward direction. The
        fused autograd Function (LoRA_MLP / LoRA_QKV / LoRA_O) keeps
        Tensor refs to the base weights as plain Python attributes on
        ``ctx`` (e.g. ``ctx.weights``), bypassing
        ``ctx.save_for_backward`` and therefore bypassing the saved-
        tensors pack/unpack spill path. By the time the autograd
        backward runs, the forward post-release has already reset every
        base ``param.data`` to an empty placeholder; without this
        re-gather the bwd matmul against ``ctx.weights[i]`` raises the
        same ``size mismatch ... vec (0)`` error the M0 spike captured.
        """
        for sub in module.modules():
            self._pre_gather(sub, grad_output)

    def _post_release_subtree_bwd(
        self, module: "nn.Module", grad_input: Any, grad_output: Any
    ) -> None:
        """Backward-post hook: release after container bwd, mirror of subtree-fwd.

        Defers to ``_post_release_bwd`` per submodule so the
        premature-fire guard (the ``inputs_have_grad`` check around
        ``register_full_backward_hook``) still applies — leaf
        embeddings reached via the fused embedding container would
        otherwise see their post-bwd fire before the embedding's own
        backward kernel runs and clear the gathered weight to a length-0
        placeholder mid-AccumulateGrad. Walking in reverse keeps the
        active-user ref-count unwind LIFO, matching the pre-gather
        order.
        """
        for sub in reversed(list(module.modules())):
            self._post_release_bwd(sub, grad_input, grad_output)

    def _pre_gather_bwd(self, module: "nn.Module", grad_output: Any) -> None:
        """Backward-pre hook: gather direct params before this module's bwd.

        Linear's autograd computes ``grad_input = grad_output @ weight`` —
        the weight tensor's full data must be live, but ``_post_release``
        already cleared it to an empty placeholder. Re-running the gather
        here makes backward see the real param. Mirrors ``_pre_gather``
        but takes the backward-hook signature.
        """
        # Reuse the forward-gather logic; ``inputs`` is unused there.
        self._pre_gather(module, grad_output)

    def _post_release_bwd(
        self, module: "nn.Module", grad_input: Any, grad_output: Any
    ) -> None:
        """Backward-post hook: release direct params after this module's bwd.

        Caveat: ``register_full_backward_hook`` fires *prematurely* — before
        the module's actual backward kernel runs — for modules whose inputs
        do NOT require grad (e.g. ``nn.Embedding`` taking a ``LongTensor`` of
        ``input_ids``). In that case PyTorch's ``BackwardHook`` calls the
        user post-hook from within the *output* grad-fn callback (see
        ``torch/utils/hooks.py:_BackwardHook.setup_output_hook`` — the
        ``input_tensors_index is None`` branch warns and dispatches the
        user post-hook immediately). If we release the param here, the
        subsequent ``EmbeddingBackward`` runs against a length-0 placeholder
        and ``AccumulateGrad`` fails with
        ``"size of tensor a (0) must match the size of tensor b (...) at
        non-singleton dimension 1"`` — the param-grad shape derived from the
        live ``param.size()`` (now ``(0,)``) clashes with the real grad
        produced by the embedding's saved-shape backward.

        Detect the early-fire case by checking ``grad_input``: when no input
        required grad, ``grad_input`` is a tuple of ``None`` entries (see
        ``_pack_with_none([], [], n_inputs)`` in the hooks helper). Skip
        the release in that case — the param will be released by
        ``__exit__`` instead. Slightly inflates the post-trace peak (the
        gathered weight stays live for the rest of backward) but preserves
        correctness; the same modules are typically embeddings near the
        leaves of the autograd graph so the residency overlap is bounded.
        """
        if grad_input is not None and isinstance(grad_input, tuple):
            inputs_have_grad = any(g is not None for g in grad_input)
            if not inputs_have_grad:
                # Premature-fire path: PyTorch dispatched this post-hook
                # from the output-grad callback because no input required
                # grad. The module's own backward (which produces the
                # param grads) hasn't run yet. Decrement the active-user
                # ref-counts that ``_pre_gather_bwd`` incremented so a
                # later ``__exit__`` doesn't double-release, but leave
                # the gathered ``param.data`` in place so
                # ``AccumulateGrad`` sees the real shape.
                for param in module.parameters(recurse=False):
                    pid = id(param)
                    users = self._active_param_users.get(pid, 0)
                    if users > 0:
                        new_count = users - 1
                        if new_count <= 0:
                            self._active_param_users.pop(pid, None)
                        else:
                            self._active_param_users[pid] = new_count
                return
        # Normal path: inputs received gradients → module backward already
        # ran → param grads are accumulated → safe to release.
        self._post_release(module, grad_input, grad_output)

    # ---- saved-tensors spill / restore ---------------------------------
    #
    # Backward IS supported under on-demand: the unpack hook copies CPU-
    # spilled tensors back to ``self.device`` before returning, so autograd
    # receives a CUDA tensor on a CUDA backward. The H2D copy adds latency
    # proportional to the saved-tensor footprint (a 7B forward saves on the
    # order of a few GB of activations -> a few hundred ms of PCIe time
    # per backward pass on a 26 GB/s link); the trace driver currently
    # passes ``include_backward=False`` when on-demand engages, so this
    # path is dormant in production but no longer a footgun for callers
    # that want to run backward under on-demand themselves.

    def _pack_hook(self, tensor: Any) -> Any:
        """Spill autograd-retained GPU tensors to CPU at save time."""
        try:
            if not getattr(tensor, "is_cuda", False):
                return tensor
            return tensor.detach().to("cpu", non_blocking=False)
        except Exception as _e:  # noqa: BLE001 - surface spill failures
            # Returning the original CUDA tensor would silently keep the
            # saved-for-backward buffer alive on GPU, invalidating the
            # trace peak or causing a downstream OOM without exposing
            # why spill broke. Mirror ``_unpack_hook``'s log+raise
            # contract so failures are visible (CR 3191961017).
            LOG.warning("OnDemandTensorMgr pack spill failed (%s)", _e)
            raise

    def _unpack_hook(self, packed: Any) -> Any:
        """Restore a spilled tensor on the configured GPU device.

        If ``packed`` is a CPU tensor and we know the target device
        (``self.device`` set), copy it back to GPU before returning.
        Backward under on-demand otherwise gets a CPU tensor on a CUDA
        backward and fails deep in autograd C++.
        """
        try:
            # Non-tensor or already on GPU: nothing to do. ``torch.Tensor``
            # exposes ``is_cuda`` but not ``is_cpu``; check device.type instead.
            device = getattr(packed, "device", None)
            if device is None:
                return packed
            if getattr(device, "type", None) != "cpu":
                return packed
            if self.device is None:
                # No target device known — autograd will surface the CPU/CUDA
                # mismatch itself if it matters.
                return packed
            try:
                target = self._normalize_device(self.device)
            except Exception:  # noqa: BLE001 - defensive (torch import inside)
                return packed
            if target.type == "cpu":
                return packed
            return packed.to(target, non_blocking=True)
        except Exception as exc:  # noqa: BLE001 - defensive
            # Surface H2D failures: previously the unpack would silently
            # degrade and autograd later exploded with "expected CUDA,
            # got CPU" — actionable error hidden. Backward IS supported
            # publicly, so propagate the real cause.
            LOG.warning("OnDemandTensorMgr unpack restore failed (%s)", exc)
            raise

    # ---- back-compat API (no-ops in enabled mode under hook-based path) ---

    def allocate_inputs(self, op: OpRecord) -> None:
        """Compatibility shim. The enabled path uses module-level hooks.

        Kept callable in disabled mode to preserve the M1 fast-path test.
        Raises in enabled mode if invoked outside the context to flag misuse.
        """
        if self.disabled:
            return
        if not self._entered:
            raise RuntimeError(
                "OnDemandTensorMgr.allocate_inputs called outside ``with`` "
                "context. Use as a context manager — gathering happens via "
                "module hooks, not by calling allocate_inputs directly."
            )
        # No-op when entered: the pre-forward hook on the relevant module
        # has already gathered its params.

    def free_after(self, op: OpRecord) -> None:
        """Compatibility shim. The enabled path uses module-level hooks."""
        if self.disabled:
            return
        if not self._entered:
            raise RuntimeError(
                "OnDemandTensorMgr.free_after called outside ``with`` context."
            )
        # No-op when entered: the post-forward hook on the relevant module
        # has already released its params.

    # ---- introspection --------------------------------------------------

    def live_tensor_ids(self) -> Iterable[int]:
        return tuple(self._spills.keys())


__all__ = [
    "OnDemandTensorMgr",
    "_find_fused_kernel_containers",
    "_is_fused_method",
]
