"""
TiledMLP support for DDP, FSDP, and single GPU
"""

import threading
from typing import List

import torch


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
        output_unsharded = torch.cat(output_shards, dim=1)

        return output_unsharded

    @staticmethod
    def backward(ctx, *grads) -> torch.Tensor:
        fn = ctx.fn
        (x,) = ctx.saved_tensors
        self = ctx.self
        shards = ctx.shards
        compute_params = ctx.compute_params

        x_requires_grad = x.requires_grad
        x = x.detach()
        x.requires_grad_(x_requires_grad)

        incoming_grad = grads[0]
        x_grad = torch.zeros_like(x)
        x_shards = list(torch.chunk(x, chunks=shards, dim=1))

        # Create a gradient accumulator for parameters
        grad_accumulator = GradientAccumulator(compute_params, shards, dtype=x.dtype)

        shard_step = x_shards[0].numel()
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

            # Install hooks for this shard
            is_last_shard = i + 1 == shards
            grad_accumulator.install_hooks(is_last_shard)

            with torch.enable_grad():
                output = fn(self, x_shard)
            torch.autograd.backward(output, incoming_grad_shard)

        # Clean up hooks
        grad_accumulator.cleanup()
        del grad_accumulator

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
