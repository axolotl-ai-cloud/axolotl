"""Keep saved-tensor layouts intact under TRL's activation offloader.

``OffloadActivations`` copies a non-contiguous saved tensor to CPU as a contiguous one, so
backward receives it with other strides than forward saved. SDPA's memory-efficient
attention backward depends on the saved layout of its transposed ``query`` / ``out`` and
computes wrong gradients from such a copy.
"""

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
