"""
TiledMLP support for DDP, FSDP, and single GPU
"""

import contextlib
import os
import threading
from typing import List

import torch

# Opt-in fp32 accumulation for the tiled backward. The default accumulates
# at the param's own dtype, which matches what AccumulateGrad does in the
# unsharded backward and avoids materialising an fp32 buffer the size of
# every compute param. Set ``AXOLOTL_TILED_MLP_ACCUM_FP32=1`` to recover
# the previous fp32-accumulator behaviour when bf16 precision is the
# concern (e.g. very large N-shard sums where bf16 round-off accumulates).
_TILED_MLP_ACCUM_FP32 = os.environ.get("AXOLOTL_TILED_MLP_ACCUM_FP32", "0") == "1"


def _find_fsdp2_module(module):
    """Return the nearest FSDP2 :class:`FSDPModule` that owns ``module``.

    FSDP2 (``torch.distributed.fsdp.fully_shard``) registers per-module
    post-backward hooks that reshard parameters once their gradients have
    been produced. Inside :class:`TiledMLP.backward` we run several inner
    backwards over shards of the same input; if the wrapping FSDPModule
    reshards between iterations, the unsharded params are gone and the
    next tile recomputes against bogus shards. We have to disable reshard
    on the wrapping FSDPModule for the duration of the loop.

    The MLP itself is rarely the directly-wrapped module — production
    setups apply ``fully_shard`` at the decoder-layer level. Walk the
    global FSDP module-state registry to find the nearest ancestor whose
    parameter group contains us. Result is cached on the module so we pay
    the lookup once.

    Returns ``None`` if FSDP2 is not in use, or no wrapping FSDPModule
    contains ``module`` as a descendant.
    """
    cached = getattr(module, "_axolotl_fsdp2_owner", "__unset__")
    if cached != "__unset__":
        return cached

    try:
        from torch.distributed._composable_state import _module_state_mapping
        from torch.distributed.fsdp import FSDPModule
    except ImportError:
        module._axolotl_fsdp2_owner = None
        return None

    # MLP itself wrapped (covers the regression-guard unit test).
    if isinstance(module, FSDPModule):
        module._axolotl_fsdp2_owner = module
        return module

    # Walk the global FSDP registry looking for ancestors. The registry is
    # a WeakKeyDictionary so the snapshot is cheap and bounded by the
    # number of FSDP-wrapped modules in the process.
    target_id = id(module)
    candidates = []
    for owner in list(_module_state_mapping.keys()):
        if not isinstance(owner, FSDPModule):
            continue
        if owner is module:
            continue
        for sub in owner.modules():
            if id(sub) == target_id:
                candidates.append(owner)
                break

    if not candidates:
        result = None
    elif len(candidates) == 1:
        result = candidates[0]
    else:
        # When multiple FSDPModules are ancestors (e.g. fully_shard applied
        # to both decoder layer and the root), pick the deepest one — its
        # subtree is smallest. Counting modules is O(N) per candidate but
        # only runs once per MLP instance.
        result = min(candidates, key=lambda m: sum(1 for _ in m.modules()))

    module._axolotl_fsdp2_owner = result
    return result


@contextlib.contextmanager
def _defer_fsdp2_reshard(module):
    """Suspend FSDP2's post-backward reshard on the wrapping FSDPModule.

    The tiled backward calls :func:`torch.autograd.backward` once per shard.
    Each inner backward triggers FSDP2's per-module post-backward hooks,
    which would reshard parameters mid-loop. We pause that by toggling
    ``set_reshard_after_backward(False)`` on the wrapping FSDPModule, run
    the loop, restore the original setting, then issue a single explicit
    ``reshard()`` so the post-loop state matches normal FSDP2 semantics.

    No-op when ``module`` is not under FSDP2.
    """
    fsdp_mod = _find_fsdp2_module(module)
    if fsdp_mod is None:
        yield
        return

    # No public getter for ``reshard_after_backward`` in PyTorch 2.11;
    # read off the param group directly. The internal accessor surface
    # is documented in
    # ``torch.distributed.fsdp._fully_shard._fsdp_param_group.FSDPParamGroup``.
    state = fsdp_mod._get_fsdp_state()
    param_group = state._fsdp_param_group
    if param_group is None:
        # Nothing to defer (e.g. ignored module with no FSDP-managed params).
        yield
        return

    prev = param_group.reshard_after_backward
    fsdp_mod.set_reshard_after_backward(False, recurse=False)
    try:
        yield
    finally:
        # Restore so subsequent backward passes outside the tile loop
        # behave normally, then issue the deferred reshard once.
        fsdp_mod.set_reshard_after_backward(prev, recurse=False)
        fsdp_mod.reshard()


class DeepSpeedTiledMLPMoE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        fn,
        self,
        x,
        shards,
        compute_params,
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.self = self
        ctx.shards = shards
        ctx.compute_params = [p for p in compute_params if p.requires_grad]
        ctx.save_for_backward(x)

        x_shards = list(torch.chunk(x, chunks=shards, dim=1))
        with torch.no_grad():
            output_shards = [fn(self, x_shard) for x_shard in x_shards]

        ctx.is_tuple_output = isinstance(output_shards[0], tuple)
        if isinstance(output_shards[0], tuple):
            tuple_dim_idx = [1, 0]
            output_unsharded = tuple(
                torch.cat(
                    [output_shard[i] for output_shard in output_shards],
                    dim=tuple_dim_idx[i],
                )
                for i in range(len(output_shards[0]))
            )
        else:
            output_unsharded = torch.cat(output_shards, dim=1)

        return output_unsharded

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        self = ctx.self
        shards = ctx.shards
        compute_params = ctx.compute_params
        is_tuple_output = ctx.is_tuple_output

        x_requires_grad = x.requires_grad
        x = x.detach()
        # detach() unsets `x.requires_grad`, so restore it
        x.requires_grad_(x_requires_grad)

        incoming_grad = grads[0]
        x_grad = torch.zeros_like(x)
        x_shards = list(torch.chunk(x, chunks=shards, dim=1))

        shard_step = x_shards[0].numel()
        for i, x_shard in enumerate(x_shards):
            # Tell deepspeed not to add a new grad to its ipg bucket until the last shard is run
            if compute_params is not None:
                if i + 1 < shards:
                    for param in compute_params:
                        param.ds_grad_is_ready = False
                else:
                    # last shard, can add the grad
                    for param in compute_params:
                        param.ds_grad_is_ready = True

            x_shard.requires_grad_(x_requires_grad)

            shard_offset = i * shard_step
            x_shard.grad = (
                x_grad.view(-1)
                .narrow(0, shard_offset, x_shard.numel())
                .view_as(x_shard)
            )
            incoming_grad_shard = (
                incoming_grad.view(-1)
                .narrow(0, shard_offset, x_shard.numel())
                .view_as(x_shard)
            )
            with torch.enable_grad():
                output = fn(self, x_shard)
            if is_tuple_output:
                torch.autograd.backward(output[0], incoming_grad_shard)
            else:
                torch.autograd.backward(output, incoming_grad_shard)

        return (None, None, x_grad, None, None)


class TiledMLP(torch.autograd.Function):
    """
    TiledMLP implementation using gradient hooks
    """

    @staticmethod
    def forward(
        ctx,
        fn,
        self,
        x,
        shards,
        compute_params,
    ) -> torch.Tensor:
        ctx.fn = fn
        ctx.self = self
        ctx.shards = shards
        ctx.compute_params = [p for p in compute_params if p.requires_grad]
        ctx.save_for_backward(x)

        x_shards = list(torch.chunk(x, chunks=shards, dim=1))
        with torch.no_grad():
            output_shards = [fn(self, x_shard) for x_shard in x_shards]
        ctx.is_tuple_output = isinstance(output_shards[0], tuple)
        if isinstance(output_shards[0], tuple):
            tuple_dim_idx = [1, 0]
            output_unsharded = tuple(
                torch.cat(
                    [output_shard[i] for output_shard in output_shards],
                    dim=tuple_dim_idx[i],
                )
                for i in range(len(output_shards[0]))
            )
        else:
            output_unsharded = torch.cat(output_shards, dim=1)

        return output_unsharded

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        self = ctx.self
        shards = ctx.shards
        compute_params = ctx.compute_params
        is_tuple_output = ctx.is_tuple_output

        x_requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad_(x_requires_grad)

        incoming_grad = grads[0]
        x_grad = torch.zeros_like(x)
        x_shards = list(torch.chunk(x, chunks=shards, dim=1))

        # Snapshot existing ``.grad`` for each param and zero it; we will
        # accumulate the per-shard contributions into a per-param buffer
        # and write back at the end. The previous implementation used
        # ``param.register_hook`` per shard, which (a) re-installed hooks
        # every iteration so the N-th shard ran N stacked hooks and
        # double-counted contributions, and (b) scaled by ``1/N`` even
        # though sequence-dim sharding makes per-shard grads additive,
        # not averaged. The combined effect was a gradient roughly
        # 2x-2.5x the analytical value. Direct inline accumulation is
        # both simpler and correct, and avoids interactions with FSDP2's
        # own backward hooks.
        #
        # The accumulator defaults to the param's own dtype to match
        # what AccumulateGrad would do in the unsharded backward. The
        # earlier implementation accumulated in fp32, which doubled the
        # parameter-side memory footprint in bf16 MoE training where the
        # accumulator's ``[E, hidden, 2*intermediate]`` shape dominates.
        # Set ``AXOLOTL_TILED_MLP_ACCUM_FP32=1`` to opt back into fp32
        # accumulation when bf16 round-off is the concern.
        prev_grads = {}
        accum_grads = {}
        for p in compute_params:
            prev_grads[p] = p.grad
            accum_dtype = torch.float32 if _TILED_MLP_ACCUM_FP32 else p.dtype
            accum_grads[p] = torch.zeros_like(p, dtype=accum_dtype)
            p.grad = None

        shard_step = x_shards[0].numel()
        # Suspend FSDP2 post-backward reshard for the duration of the loop.
        # Without this, the first inner backward triggers FSDP2's reshard
        # hook on the wrapping FSDPModule and subsequent shards recompute
        # against only-local DTensor shards — silent grad corruption.
        # Single-GPU and DDP paths fall through to a no-op context manager.
        with _defer_fsdp2_reshard(self):
            for i, x_shard in enumerate(x_shards):
                x_shard.requires_grad_(x_requires_grad)

                shard_offset = i * shard_step
                x_shard.grad = (
                    x_grad.view(-1)
                    .narrow(0, shard_offset, x_shard.numel())
                    .view_as(x_shard)
                )
                incoming_grad_shard = (
                    incoming_grad.view(-1)
                    .narrow(0, shard_offset, x_shard.numel())
                    .view_as(x_shard)
                )

                with torch.enable_grad():
                    output = fn(self, x_shard)
                if is_tuple_output:
                    torch.autograd.backward(output[0], incoming_grad_shard)
                else:
                    torch.autograd.backward(output, incoming_grad_shard)

                # Capture this shard's contribution into the per-param
                # accumulator and clear ``.grad`` so the next shard starts
                # from zero. Skip the dtype cast when the accumulator
                # matches the param dtype (the default) — that cast was
                # the per-shard HBM-bandwidth tax on the bf16 path.
                for p in compute_params:
                    if p.grad is not None:
                        shard_grad = p.grad.detach()
                        if shard_grad.dtype != accum_grads[p].dtype:
                            shard_grad = shard_grad.to(accum_grads[p].dtype)
                        accum_grads[p].add_(shard_grad)
                        p.grad = None

        # Restore prior grad value (if any) and add the tiled contribution.
        for p in compute_params:
            tiled_contrib = accum_grads[p]
            if tiled_contrib.dtype != p.dtype:
                tiled_contrib = tiled_contrib.to(p.dtype)
            if prev_grads[p] is None:
                p.grad = tiled_contrib
            else:
                p.grad = prev_grads[p] + tiled_contrib

        return (None, None, x_grad, None, None)


class GradientAccumulator:
    """
    Manual gradient accumulator for TiledMLP with configurable precision.

    .. note::
        The production TiledMLP backward (above) accumulates inline and
        does not call this class — it is retained as a reference / opt-in
        path for callers that want hook-based accumulation. The defaults
        below match the inline path: param-dtype accumulator (matches
        ``AccumulateGrad`` in the unsharded backward) and ``1.0`` per-shard
        scaling (sequence-dim sharded grads are additive, not averaged).
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        total_shards: int,
        dtype: torch.dtype | None = None,
    ):
        self.params = params
        self.total_shards = total_shards
        # Default to the param's own dtype to avoid the 2x parameter-side
        # memory regression in bf16 MoE training where the accumulator
        # shape ``[E, hidden, 2*intermediate]`` dominates. fp32 accumulation
        # is opt-in via the ``dtype`` arg.
        if dtype is not None:
            self.grad_accumulation_dtype = dtype
        elif params:
            self.grad_accumulation_dtype = params[0].dtype
        else:
            self.grad_accumulation_dtype = torch.float32
        self.accumulated_grads = {}
        self.hooks = []
        self.lock = threading.Lock()
        # Sequence-dim shards partition the per-token sum; their
        # contributions are additive (``sum_t dL_t/dW``), not averaged.
        # The previous ``1/total_shards`` scaling produced a mean and was
        # a correctness bug for this sharding semantics.
        self.gradient_scale = 1.0

        # Initialize accumulated gradients in the specified dtype
        for param in self.params:
            if param.grad is not None:
                self.accumulated_grads[param] = param.grad.to(
                    self.grad_accumulation_dtype
                )
                param.grad = None
            else:
                self.accumulated_grads[param] = torch.zeros_like(
                    param, dtype=self.grad_accumulation_dtype
                )

    def install_hooks(self, is_last_shard: bool):
        """Install gradient hooks that accumulate gradients in higher precision"""

        def create_hook(param):
            def hook(grad):
                with self.lock:
                    # Skip the dtype cast when the accumulator already
                    # matches the grad dtype (the default after the
                    # param-dtype change above) — the redundant cast was
                    # the per-shard HBM bandwidth tax called out in the
                    # tiled-MLP regression analysis.
                    if grad.dtype == self.grad_accumulation_dtype:
                        scaled_grad = (
                            grad
                            if self.gradient_scale == 1.0
                            else grad * self.gradient_scale
                        )
                    else:
                        scaled_grad = (
                            grad.to(self.grad_accumulation_dtype) * self.gradient_scale
                        )

                    if param in self.accumulated_grads:
                        self.accumulated_grads[param] += scaled_grad
                    else:
                        self.accumulated_grads[param] = scaled_grad.clone()

                    # Only assign the accumulated gradient on the last shard
                    if is_last_shard:
                        if self.accumulated_grads[param].dtype != param.dtype:
                            param.grad = self.accumulated_grads[param].to(param.dtype)
                        else:
                            param.grad = self.accumulated_grads[param]
                        return param.grad
                    return None

            return hook

        # Install hooks on all parameters
        for param in self.params:
            if param.requires_grad:
                hook = param.register_hook(create_hook(param))
                self.hooks.append(hook)

    def cleanup(self):
        """Remove all installed hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        del self.accumulated_grads
