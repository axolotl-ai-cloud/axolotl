"""Allocate-before-use / free-after tensor context for profiling models > device memory."""

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
    """Names of fused LoRA apply_* functions whose direct-attribute weight reads bypass per-Linear gather hooks; listed by name (not import) so a missing kernel module stays non-fatal."""
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
    """True iff ``attr`` is an instance-bound method whose underlying function is one of the fused-kernel apply_* entries."""
    if not isinstance(attr, types.MethodType):
        return False
    fn = getattr(attr, "__func__", None)
    name = getattr(fn, "__name__", None)
    return name in _fused_kernel_func_names()


def _find_fused_kernel_containers(model: "nn.Module") -> "list[nn.Module]":
    """Return modules with at least one fused-kernel method binding; deterministic ``model.modules()`` order so tests can rely on stable enumeration."""
    out: list["nn.Module"] = []
    for sub in model.modules():
        for attr_name in ("forward", "apply_qkv", "apply_o"):
            attr = getattr(sub, attr_name, None)
            if _is_fused_method(attr):
                out.append(sub)
                break
    return out


# PEFT LoRA trainable-factor attribute name fragments (substring match).
_PEFT_LORA_NAME_TAGS: frozenset[str] = frozenset(
    {
        "lora_A",
        "lora_B",
        "lora_embedding_A",
        "lora_embedding_B",
        "lora_magnitude_vector",
    }
)


def _has_peft_lora_factor(
    module: "nn.Module", *, recurse_children: bool = True
) -> bool:
    """True iff ``module`` directly owns a trainable LoRA factor."""
    # Bare nn.Parameter form.
    for name, p in module.named_parameters(recurse=False):
        if not p.requires_grad:
            continue
        if any(tag in name for tag in _PEFT_LORA_NAME_TAGS):
            return True
    if not recurse_children:
        return False
    # PEFT's ParameterDict/wrapped-Linear: child attr name carries the tag.
    for child_name, child in module.named_children():
        if not any(tag in child_name for tag in _PEFT_LORA_NAME_TAGS):
            continue
        for _pname, p in child.named_parameters(recurse=True):
            if p.requires_grad:
                return True
    return False


def _find_peft_lora_containers(model: "nn.Module") -> "list[nn.Module]":
    """Return modules that directly own trainable LoRA factors; excludes fused-kernel containers (their hooks already cover the same subtree). Deterministic ``model.modules()`` order."""
    fused = set(id(m) for m in _find_fused_kernel_containers(model))
    out: list["nn.Module"] = []
    for sub in model.modules():
        if id(sub) in fused:
            continue
        if not _has_peft_lora_factor(sub, recurse_children=True):
            continue
        out.append(sub)
    return out


@dataclass
class _ParamSpill:
    """Bookkeeping for one parameter that's been spilled to CPU."""

    param: Any  # torch.nn.Parameter — Any keeps import light
    cpu_storage: Any  # torch.Tensor on CPU (pinned if possible)
    original_device: Any  # torch.device the param was on at __enter__
    original_data: Any  # legacy: GPU tensor at __enter__; None on the new path


class OnDemandTensorMgr:
    """Context manager that materializes each leaf's params just-in-time."""

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
        # Per-param ref-count by id(param); tied params fire hooks per owner.
        self._active_param_users: dict[int, int] = {}
        self._handles: list[Any] = []
        self._sthook_ctx: Any = None
        self._entered = False
        self._n_pin_failures = 0
        # Populated by ``__enter__`` after fused-kernel detection. Tests
        # may inspect this to verify per-container hook installation.
        self._fused_containers: list["nn.Module"] = []
        # PEFT-LoRA containers needing subtree gather/release so param.data stays live across backward
        self._peft_lora_containers: list["nn.Module"] = []

    # ---- context-manager protocol --------------------------------------

    def __enter__(self) -> "OnDemandTensorMgr":
        """Spill parameters to pinned CPU and install the gather/spill hooks."""
        if self._entered:
            raise RuntimeError(
                "OnDemandTensorMgr cannot be re-entered before __exit__ "
                "completes. Each context-manager scope must pair exactly "
                "one __enter__ with one __exit__; nested re-entry on the "
                "same instance would duplicate hook registrations and "
                "corrupt spill bookkeeping."
            )
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

            # Enabled mode spills params only; fail fast on CPU buffers when target is CUDA.
            if target_device is not None:
                for buffer_name, buffer in self.model.named_buffers():
                    # Strict device equality catches cross-CUDA-index mismatches too.
                    if getattr(buffer, "device", None) != target_device:
                        raise RuntimeError(
                            f"OnDemandTensorMgr: buffer {buffer_name!r} on "
                            f"{getattr(buffer, 'device', None)!r} is not on "
                            f"target_device {target_device!r}. Move the buffer "
                            "to the target device or extend the spill hooks "
                            "to cover buffers."
                        )

            for sub in self.model.modules():
                # prepend=True so gather precedes the trace driver's allocated_before snapshot.
                self._handles.append(
                    sub.register_forward_pre_hook(self._pre_gather, prepend=True)
                )
                # FIFO post-release so it fires after the trace measures peak.
                self._handles.append(sub.register_forward_hook(self._post_release))
                # Symmetric backward pair: prepend pre-gather, FIFO post-release.
                self._handles.append(
                    sub.register_full_backward_pre_hook(
                        self._pre_gather_bwd, prepend=True
                    )
                )
                self._handles.append(
                    sub.register_full_backward_hook(self._post_release_bwd)
                )

            # Container-level gather for fused-kernel modules whose patched forward bypasses per-Linear hooks.
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
                # Backward subtree gather: fused autograd Function bypasses saved_tensors_hooks via ctx.weights.
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

            # PEFT-LoRA subtree gather keeps factors + base weight live for autograd shape-derivation.
            self._peft_lora_containers = _find_peft_lora_containers(self.model)
            if self._peft_lora_containers:
                LOG.debug(
                    "OnDemandTensorMgr: %d PEFT-LoRA container(s) "
                    "detected; installing per-container gather hooks",
                    len(self._peft_lora_containers),
                )
            for container in self._peft_lora_containers:
                self._handles.append(
                    container.register_forward_pre_hook(
                        self._pre_gather_subtree, prepend=True
                    )
                )
                self._handles.append(
                    container.register_forward_hook(self._post_release_subtree)
                )
                # Backward shape-derivation needs real weights; mirror fused-kernel pair.
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

            # Saved-tensors hooks spill to CPU; otherwise grad_fn pins gathered params alive.
            self._sthook_ctx = torch.autograd.graph.saved_tensors_hooks(
                self._pack_hook, self._unpack_hook
            )
            self._sthook_ctx.__enter__()
        except BaseException:
            # Mirror __exit__'s teardown so partial setup doesn't wedge params.
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
        """Undo partial __enter__ on the exception path; best-effort."""
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
                    # Legacy retain-storage path.
                    spill.original_data.copy_(
                        spill.cpu_storage.to(
                            spill.original_data.device, non_blocking=True
                        )
                    )
                    spill.param.data = spill.original_data
                elif getattr(spill.original_device, "type", None) == "cuda":
                    # Fresh GPU alloc; id(param) preserved.
                    spill.param.data = spill.cpu_storage.to(
                        spill.original_device, non_blocking=True
                    )
                else:
                    # CPU-original: cpu_storage IS the original.
                    spill.param.data = spill.cpu_storage
                # Move grad back to original device for caller's device-match.
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
            # Sync each CUDA target since bare synchronize() only waits on the current device.
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
        self._peft_lora_containers = []

    def __exit__(self, exc_type, exc, tb) -> None:
        """Remove hooks and restore parameters from their pinned-CPU spill copies."""
        self._entered = False
        if self.disabled:
            return

        # Remove hooks first; rebind except to _e to avoid clobbering exc parameter.
        for h in self._handles:
            try:
                h.remove()
            except Exception as _e:  # noqa: BLE001 - defensive
                LOG.debug("OnDemandTensorMgr: hook removal failed during exit (%s)", _e)
        self._handles.clear()

        # Exit saved_tensors_hooks before restoring params; in-flight backward already drained.
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
                # Move grad back to original device for caller's device-match.
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
                # Point param.data + grad at the CPU spill for the failure path.
                spill.param.data = spill.cpu_storage
                if (
                    spill.param.grad is not None
                    and getattr(spill.param.grad.device, "type", None) != "cpu"
                ):
                    spill.param.grad = spill.param.grad.to("cpu", non_blocking=True)
        # Sync each CUDA target since bare synchronize() only waits on the current device.
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
        self._peft_lora_containers = []

    # ---- spill / restore helpers ---------------------------------------

    def _spill_param_to_cpu(
        self, param: Any, target_device: "torch.device | None"
    ) -> None:
        """Move ``param`` to pinned CPU storage; leave a placeholder in .data."""
        import torch

        # Skip tied/shared params already spilled — repeat would clobber cpu_storage.
        if id(param) in self._spills:
            return

        original_device = param.device

        if original_device.type == "cpu":
            # Preserve original ref; pin_memory may return a new tensor that breaks tied weights.
            original_data = param.data
            try:
                pinned = original_data.pin_memory()
                cpu_storage = pinned
            except Exception:  # noqa: BLE001 - pinning is best-effort
                cpu_storage = original_data
                self._n_pin_failures += 1
            # Only set original_data when pinning produced a distinct tensor.
            spill_original = original_data if cpu_storage is not original_data else None
            self._spills[id(param)] = _ParamSpill(
                param=param,
                cpu_storage=cpu_storage,
                original_device=original_device,
                original_data=spill_original,
            )
            return

        # GPU-resident: copy GPU→CPU, drop GPU ref so allocator reclaims storage.
        try:
            cpu_storage = param.data.detach().to("cpu", copy=True)
            try:
                cpu_storage = cpu_storage.pin_memory()
            except Exception:  # noqa: BLE001 - pinning is best-effort
                self._n_pin_failures += 1
        except Exception as exc:  # noqa: BLE001 - defensive
            # Fail fast on cross-device spill failure; in-place fallback for same-device.
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

        # Capture dtype before reassigning param.data.
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
        """Normalize a device-like value to a ``torch.device``."""
        import torch

        if isinstance(device, torch.device):
            return device
        if isinstance(device, int):
            return torch.device("cuda", device)
        return torch.device(device)

    def _infer_model_device(self) -> "torch.device | None":
        """Best-effort model-device inference for default target alignment."""
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
        """Resolve the target device for gathered params."""
        if self.device is None:
            return None
        import torch

        if isinstance(self.device, torch.device):
            return self.device
        return self._normalize_device(self.device)

    def _pre_gather(self, module: "nn.Module", inputs: Any) -> None:
        """Copy the module's direct params to target_device; tied params bump refcount."""
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
                    # Surface the real cause; device-mismatch fallback would hide it.
                    raise
            self._active_param_users[pid] = 1

    def _post_release(self, module: "nn.Module", inputs: Any, output: Any) -> None:
        """Replace direct params with empty placeholders; outermost owner clears."""
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
        """Run ``_pre_gather`` over every submodule so the fused/PEFT container's whole subtree is GPU-resident before the patched forward reads weights by direct attribute access."""
        for sub in module.modules():
            self._pre_gather(sub, inputs)

    def _post_release_subtree(
        self, module: "nn.Module", inputs: Any, output: Any
    ) -> None:
        """Mirror of ``_pre_gather_subtree`` but walks submodules in reverse so the active-user refcounts unwind LIFO (matches the tied-param ownership pattern)."""
        for sub in reversed(list(module.modules())):
            self._post_release(sub, inputs, output)

    def _pre_gather_subtree_bwd(self, module: "nn.Module", grad_output: Any) -> None:
        """Backward-pre subtree gather; needed because fused autograd Functions stash raw weight refs on ``ctx`` (bypassing ``save_for_backward``), so the forward post-release left them as empty placeholders."""
        for sub in module.modules():
            self._pre_gather(sub, grad_output)

    def _post_release_subtree_bwd(
        self, module: "nn.Module", grad_input: Any, grad_output: Any
    ) -> None:
        """Backward-post subtree release; defers to ``_post_release_bwd`` per submodule so the ``inputs_have_grad`` premature-fire guard still applies (otherwise embeddings would clear their weight mid-AccumulateGrad)."""
        for sub in reversed(list(module.modules())):
            self._post_release_bwd(sub, grad_input, grad_output)

    def _pre_gather_bwd(self, module: "nn.Module", grad_output: Any) -> None:
        """Backward-pre gather; reuses forward logic."""
        self._pre_gather(module, grad_output)

    def _post_release_bwd(
        self, module: "nn.Module", grad_input: Any, grad_output: Any
    ) -> None:
        """Backward-post release; skip premature-fire when no input required grad."""
        if grad_input is not None and isinstance(grad_input, tuple):
            inputs_have_grad = any(g is not None for g in grad_input)
            if not inputs_have_grad:
                # Embeddings: hook fires before module backward runs; defer release to __exit__.
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
        # Normal path: grads accumulated → safe to release.
        self._post_release(module, grad_input, grad_output)

    # Saved-tensors spill/restore — backward is supported (unpack copies back to GPU).

    def _pack_hook(self, tensor: Any) -> Any:
        """Spill autograd-retained GPU tensors to CPU at save time."""
        try:
            if not getattr(tensor, "is_cuda", False):
                return tensor
            return tensor.detach().to("cpu", non_blocking=False)
        except Exception as _e:  # noqa: BLE001 - surface spill failures
            # Return-original would silently pin saved buffer alive on GPU.
            LOG.warning("OnDemandTensorMgr pack spill failed (%s)", _e)
            raise

    def _unpack_hook(self, packed: Any) -> Any:
        """Restore a spilled tensor on the configured GPU device."""
        try:
            # Non-tensor or already-GPU: nothing to do.
            device = getattr(packed, "device", None)
            if device is None:
                return packed
            if getattr(device, "type", None) != "cpu":
                return packed
            if self.device is None:
                return packed
            try:
                target = self._normalize_device(self.device)
            except Exception:  # noqa: BLE001 - defensive (torch import inside)
                return packed
            if target.type == "cpu":
                return packed
            return packed.to(target, non_blocking=True)
        except Exception as exc:  # noqa: BLE001 - defensive
            # Surface H2D failures so the real cause isn't hidden behind a downstream CUDA mismatch.
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
    "_find_peft_lora_containers",
    "_has_peft_lora_factor",
    "_is_fused_method",
]
