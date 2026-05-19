"""
TiledMLP support for DDP, FSDP, and single GPU
"""

import contextlib
import threading
from typing import List

import torch


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
        # accumulate the per-shard contributions in fp32 ourselves and
        # write back at the end. The previous implementation used
        # ``param.register_hook`` per shard, which (a) re-installed hooks
        # every iteration so the N-th shard ran N stacked hooks and
        # double-counted contributions, and (b) scaled by ``1/N`` even
        # though sequence-dim sharding makes per-shard grads additive,
        # not averaged. The combined effect was a gradient roughly
        # 2x-2.5x the analytical value. Direct inline accumulation is
        # both simpler and correct, and avoids interactions with FSDP2's
        # own backward hooks.
        prev_grads = {}
        accum_grads = {}
        for p in compute_params:
            prev_grads[p] = p.grad
            accum_grads[p] = torch.zeros_like(p, dtype=torch.float32)
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

                # Capture this shard's contribution into our fp32 accumulator
                # and clear ``.grad`` so the next shard starts from zero.
                for p in compute_params:
                    if p.grad is not None:
                        accum_grads[p].add_(p.grad.detach().to(torch.float32))
                        p.grad = None

        # Restore prior grad value (if any) and add the tiled contribution.
        for p in compute_params:
            tiled_contrib = accum_grads[p].to(p.dtype)
            if prev_grads[p] is None:
                p.grad = tiled_contrib
            else:
                p.grad = prev_grads[p] + tiled_contrib

        return (None, None, x_grad, None, None)


class GradientAccumulator:
    """
    Manual gradient accumulator for TiledMLP with configurable precision
    Accumulates in specified dtype and rescales the gradient at the end
    """

    def __init__(
        self,
        params: List[torch.nn.Parameter],
        total_shards: int,
        dtype: torch.dtype | None = None,
    ):
        self.params = params
        self.total_shards = total_shards
        self.grad_accumulation_dtype = dtype or torch.float32
        self.accumulated_grads = {}
        self.hooks = []
        self.lock = threading.Lock()
        self.gradient_scale = 1.0 / total_shards

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
                    grad_to_accum_dtype = grad.to(self.grad_accumulation_dtype)
                    scaled_grad = grad_to_accum_dtype * self.gradient_scale

                    if param in self.accumulated_grads:
                        self.accumulated_grads[param] += scaled_grad
                    else:
                        self.accumulated_grads[param] = scaled_grad.clone()

                    # Only assign the averaged gradient on the last shard
                    if is_last_shard:
                        param.grad = self.accumulated_grads[param].to(param.dtype)
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
