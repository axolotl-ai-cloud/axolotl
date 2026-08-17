"""Activation-checkpoint input ("hidden_states") offloading, ALST-style.

Gradient checkpointing already discards a layer's intermediate activations and
recomputes them in backward; it still keeps the *checkpoint input* (the layer's
hidden_states) resident. For long sequences those per-layer inputs
(num_layers x [seq, hidden]) dominate GPU memory. This offloads only that one
tensor per checkpoint to CPU and brings it back for the backward recompute —
minimal PCIe traffic (one tensor/layer, not every activation).

The copies are overlapped with compute on a side stream (forward async d2h with
bounded in-flight + backward h2d prefetch), so at long seq the transfer hides
behind the recompute.

It replaces torch's reentrant ``CheckpointFunction`` (use_reentrant=True), so it
is framework-agnostic (works under FSDP2; no DeepSpeed dependency). DTensor
inputs (sequence/context parallel) are offloaded too — ``DTensor.to`` round-trips
the local shard and preserves placements. Adapted from
torch.utils.checkpoint.CheckpointFunction and Snowflake ArcticTraining.
"""

import contextlib

import torch
from torch.utils.checkpoint import (
    _get_autocast_kwargs,
    _get_device_module,
    _infer_device_type,
    check_backward_validity,
    detach_variable,
    get_device_states,
    set_device_states,
)

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _offloadable(t: torch.Tensor) -> bool:
    # Offload any on-device activation, INCLUDING DTensors: under sequence/context
    # parallelism the layer input is a DTensor and offloading its sharded local
    # tensor is exactly what we want. DTensor.to("cpu")/.to(dev) round-trips the
    # local shard and preserves placements, so the plain path handles it. Model
    # params never reach here (they live inside run_function, not the offloaded
    # arg), so TRL-style param-storage filtering is unnecessary.
    return t.device.type != "cpu"


class _StreamOffloadManager:
    """Per-forward-pass coordinator that overlaps the input d2h/h2d copies with
    compute on a side stream. Forward: async-copy each layer input to CPU while
    later layers compute (bounded in-flight so only a few sources stay resident).
    Backward: bring inputs back on the side stream and prefetch the next one
    (inputs are consumed in reverse-of-offload order) to hide the reload."""

    def __init__(self, max_inflight=2):
        self.s1 = torch.cuda.Stream()
        self.max_inflight = max_inflight
        self.reset()

    def reset(self):
        self.next_id = 0
        self.inflight = {}  # id -> (gpu_src, event)  keep source alive until d2h done
        self.cpu = {}  # id -> (cpu_tensor, device, requires_grad)
        self.prefetched = {}  # id -> (gpu_tensor, event)

    # ---- forward ----
    def offload(self, t):
        s0 = torch.cuda.current_stream()
        self.s1.wait_stream(s0)  # input must be produced before we copy it
        with torch.cuda.stream(self.s1):
            cpu_t = t.detach().to("cpu", non_blocking=True)
        ev = self.s1.record_event()
        tid = self.next_id
        self.next_id += 1
        self.inflight[tid] = (t, ev)  # hold GPU source until copy completes
        self.cpu[tid] = (cpu_t, t.device, t.requires_grad)
        self._reap()
        return tid

    def _reap(self):
        for tid in [k for k, (_, ev) in self.inflight.items() if ev.query()]:
            del self.inflight[tid]
        while len(self.inflight) > self.max_inflight:
            tid = min(self.inflight)
            _, ev = self.inflight.pop(tid)
            ev.synchronize()

    # ---- backward ----
    def _bring_back(self, tid):
        cpu_t, device, _ = self.cpu[tid]
        with torch.cuda.stream(self.s1):
            gpu_t = cpu_t.to(device, non_blocking=True)
        return gpu_t, self.s1.record_event()

    def restore(self, tid):
        if tid in self.prefetched:
            gpu_t, ev = self.prefetched.pop(tid)
        else:
            gpu_t, ev = self._bring_back(tid)
        nxt = (
            tid - 1
        )  # next input needed (reverse order) — prefetch to overlap recompute
        if nxt >= 0 and nxt in self.cpu and nxt not in self.prefetched:
            self.prefetched[nxt] = self._bring_back(nxt)
        s0 = torch.cuda.current_stream()
        s0.wait_event(ev)
        # gpu_t was allocated on the side stream; tell the allocator it's now used
        # on the compute stream so its storage isn't reused before the recompute
        # consumes it (use-after-free guard, as in TRL's offloader).
        gpu_t.record_stream(s0)
        _, _, requires_grad = self.cpu[tid]
        out = gpu_t.detach().requires_grad_(requires_grad)
        del self.cpu[tid]
        if not self.cpu:
            self.reset()
        return out


_STREAM_MANAGER = None


def _get_stream_manager():
    global _STREAM_MANAGER
    if _STREAM_MANAGER is None:
        _STREAM_MANAGER = _StreamOffloadManager()
    return _STREAM_MANAGER


class HiddenStatesOffloadCheckpoint(torch.autograd.Function):
    """Reentrant checkpoint that offloads the layer input (hidden_states) to CPU,
    overlapping the transfer with compute via a side stream."""

    @staticmethod
    def forward(ctx, run_function, preserve_rng_state, *args):
        check_backward_validity(args)
        ctx.run_function = run_function
        ctx.preserve_rng_state = preserve_rng_state
        ctx.device_type = _infer_device_type(*args)
        ctx.device_autocast_kwargs, ctx.cpu_autocast_kwargs = _get_autocast_kwargs(
            ctx.device_type
        )
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            ctx.had_device_in_fwd = False
            device_module = _get_device_module(ctx.device_type)
            if getattr(device_module, "_initialized", False):
                ctx.had_device_in_fwd = True
                ctx.fwd_devices, ctx.fwd_device_states = get_device_states(*args)

        mgr = _get_stream_manager()
        ctx.inputs = []
        ctx.tensor_indices = []
        ctx.offload_tid = None  # manager id for the offloaded input
        ctx.offload_arg_pos = None
        tensor_inputs = []
        for i, arg in enumerate(args):
            if torch.is_tensor(arg):
                # Offload only the first tensor (the layer input / hidden_states);
                # other tensors (e.g. a shared [seq, seq] mask) stay on-device.
                if ctx.offload_tid is None and _offloadable(arg):
                    ctx.offload_tid = mgr.offload(arg)
                    ctx.offload_arg_pos = i
                    ctx.inputs.append(None)
                else:
                    tensor_inputs.append(arg)
                    ctx.tensor_indices.append(i)
                    ctx.inputs.append(None)
            else:
                ctx.inputs.append(arg)
        ctx.save_for_backward(*tensor_inputs)
        with torch.no_grad():
            outputs = run_function(*args)
        return outputs

    @staticmethod
    def backward(ctx, *args):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "use_reentrant=True checkpoint is incompatible with .grad()/passing "
                "`inputs` to .backward()."
            )
        mgr = _get_stream_manager()
        inputs = list(ctx.inputs)
        tensors = ctx.saved_tensors
        for i, idx in enumerate(ctx.tensor_indices):
            inputs[idx] = tensors[i]
        if ctx.offload_tid is not None:
            inputs[ctx.offload_arg_pos] = mgr.restore(ctx.offload_tid)

        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_device_in_fwd:
            rng_devices = ctx.fwd_devices
        with torch.random.fork_rng(
            devices=rng_devices,
            enabled=ctx.preserve_rng_state,
            device_type=ctx.device_type,
        ):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_device_in_fwd:
                    set_device_states(
                        ctx.fwd_devices,
                        ctx.fwd_device_states,
                        device_type=ctx.device_type,
                    )
            detached_inputs = detach_variable(tuple(inputs))
            device_autocast_ctx = (
                torch.amp.autocast(
                    device_type=ctx.device_type, **ctx.device_autocast_kwargs
                )
                if torch.amp.is_autocast_available(ctx.device_type)
                else contextlib.nullcontext()
            )
            with (
                torch.enable_grad(),
                device_autocast_ctx,
                torch.amp.autocast("cpu", **ctx.cpu_autocast_kwargs),
            ):
                outputs = ctx.run_function(*detached_inputs)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        outputs_with_grad = []
        args_with_grad = []
        for i in range(len(outputs)):
            if torch.is_tensor(outputs[i]) and outputs[i].requires_grad:
                outputs_with_grad.append(outputs[i])
                args_with_grad.append(args[i])
        if len(outputs_with_grad) == 0:
            raise RuntimeError("none of output has requires_grad=True")
        torch.autograd.backward(outputs_with_grad, args_with_grad)
        grads = tuple(
            inp.grad if isinstance(inp, torch.Tensor) else None
            for inp in detached_inputs
        )
        return (None, None) + grads


_ORIG_CHECKPOINT_FUNCTION = None


def patch_hidden_states_offload():
    """Replace torch's reentrant ``CheckpointFunction`` with the hidden_states
    offloading version. ``torch.utils.checkpoint.checkpoint(..., use_reentrant=True)``
    resolves ``CheckpointFunction`` from the module namespace at call time, so
    swapping the attribute is sufficient (and FSDP2-safe — only activations move)."""
    global _ORIG_CHECKPOINT_FUNCTION
    import torch.utils.checkpoint as ckpt

    if _ORIG_CHECKPOINT_FUNCTION is None:
        _ORIG_CHECKPOINT_FUNCTION = ckpt.CheckpointFunction
    ckpt.CheckpointFunction = HiddenStatesOffloadCheckpoint
    LOG.info("Patched checkpoint with hidden_states CPU offload (streamed)")


def unpatch_hidden_states_offload():
    import torch.utils.checkpoint as ckpt

    if _ORIG_CHECKPOINT_FUNCTION is not None:
        ckpt.CheckpointFunction = _ORIG_CHECKPOINT_FUNCTION
