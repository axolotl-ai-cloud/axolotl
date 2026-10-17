"""CPU-side detection of ops that need device values on the host."""

import torch
from torch.utils._python_dispatch import TorchDispatchMode

aten = torch.ops.aten

_HOST_VALUE_OPS = {
    aten.nonzero.default,
    aten.masked_select.default,
    aten._local_scalar_dense.default,
    aten.is_nonzero.default,
}
_INDEX_OPS = {aten.index.Tensor, aten.index_put.default, aten.index_put_.default}


class HostSyncRecorder(TorchDispatchMode):
    """Record dispatched ops that would block on a CUDA device: `.item()`-style scalar
    reads, `nonzero`, and boolean-mask indexing, whose output size depends on data."""

    def __init__(self):
        super().__init__()
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if func in _HOST_VALUE_OPS:
            self.ops.append(str(func))
        elif func in _INDEX_OPS and any(
            torch.is_tensor(index) and index.dtype == torch.bool
            for index in (args[1] or ())
        ):
            self.ops.append(f"{func} (bool index)")
        return func(*args, **(kwargs or {}))
