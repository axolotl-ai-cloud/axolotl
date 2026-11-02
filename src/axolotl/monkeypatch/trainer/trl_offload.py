"""Keep saved-tensor layouts intact under TRL's activation offloader.

``OffloadActivations`` copies a non-contiguous saved tensor to CPU as a contiguous one, so
backward receives it with other strides than forward saved. Two backward kernels depend on
the saved layout and compute wrong gradients (or read out of bounds) from such a copy:

* SDPA's memory-efficient attention backward, given its transposed ``query`` / ``out``;
* TRL's fused log-prob/entropy kernel (``selective_log_softmax`` on CUDA), which indexes
  the saved ``logits`` / ``index`` / ``row_mask`` with strides recorded in forward, e.g.
  for the ``logits[:, :-1]`` view DPO and KTO pass.
"""

import inspect

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def _contiguous_order(tensor) -> list[int] | None:
    """The dim order whose permute views a dense, non-contiguous tensor as contiguous."""
    if type(tensor) is not torch.Tensor or tensor.is_contiguous():
        return None
    if 0 in tensor.stride():
        return None
    order = sorted(range(tensor.dim()), key=lambda dim: -tensor.stride(dim))
    return order if tensor.permute(order).is_contiguous() else None


def patch_trl_offload_dense_layout() -> None:
    """Offload transposed saved tensors as their contiguous permutation and permute back."""
    from trl.models.activation_offloading import OffloadActivations

    if getattr(OffloadActivations, "_axolotl_dense_layout", False):
        return

    original_init = OffloadActivations.__init__

    def patched_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        pack, unpack = self.pack_hook, self.unpack_hook
        orders: dict[int, list[int]] = {}

        def pack_hook(tensor):
            order = _contiguous_order(tensor)
            tensor_id = pack(tensor if order is None else tensor.permute(order))
            # ids restart every step, so a reused id must not keep a stale order
            if order is None:
                orders.pop(tensor_id, None)
            else:
                orders[tensor_id] = order
            return tensor_id

        def unpack_hook(tensor_id):
            tensor = unpack(tensor_id)
            order = orders.pop(tensor_id, None)
            if order is None:
                return tensor
            inverse = [0] * len(order)
            for position, dim in enumerate(order):
                inverse[dim] = position
            return tensor.permute(inverse)

        self.pack_hook, self.unpack_hook = pack_hook, unpack_hook

    OffloadActivations.__init__ = patched_init
    OffloadActivations._axolotl_dense_layout = True
    LOG.info(
        "Patched TRL OffloadActivations to return transposed saved tensors with their "
        "original strides (fixes wrong SDPA memory-efficient attention gradients)"
    )


class _UnpackedCtx:
    """The backward ctx with ``saved_tensors`` unpacked once and layouts recomputed."""

    def __init__(self, ctx, saved_tensors):
        self._ctx = ctx
        self.saved_tensors = saved_tensors

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def patch_trl_logprob_entropy_saved_layout() -> None:
    """Index TRL's fused log-prob saved tensors with the strides backward receives."""
    try:
        from trl.kernels import logprob_entropy
    except ImportError:
        return
    function = getattr(logprob_entropy, "_LogProbEntropyFunction", None)
    layout = getattr(logprob_entropy, "_layout", None)
    if function is None or layout is None:
        return
    if getattr(function, "_axolotl_saved_layout", False):
        return
    original_backward = function.backward
    try:
        uses_forward_layout = "ctx.layout" in inspect.getsource(original_backward)
    except (OSError, TypeError):
        uses_forward_layout = True
    if not uses_forward_layout:
        return

    def backward(ctx, *grads):
        # one unpack: TRL's offloader releases each saved tensor when it is unpacked
        saved = ctx.saved_tensors
        logits, index, row_mask = saved[0], saved[1], saved[-1]
        unpacked = _UnpackedCtx(ctx, saved)
        unpacked.layout = layout(logits)
        if hasattr(ctx, "index_strides"):
            unpacked.index_strides = layout(index.unsqueeze(-1))[2:]
        if hasattr(ctx, "mask_strides"):
            unpacked.mask_strides = (
                (0, 0) if row_mask is None else layout(row_mask.unsqueeze(-1))[2:]
            )
        return original_backward(unpacked, *grads)

    function.backward = staticmethod(backward)
    function._axolotl_saved_layout = True
    LOG.info(
        "Patched TRL's fused log-prob/entropy backward to read saved tensors with "
        "their unpacked layout (fixes wrong gradients under activation offloading)"
    )
