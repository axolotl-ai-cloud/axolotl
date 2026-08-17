"""
Patch for torchao optim subclasses that crash under torch.compile.

torchao 0.17.0 PR #3934 added an "appearance dtype" to OptimState{4,8}bit and
OptimStateFp8, allowing them to report as e.g. bf16 while internally storing
quantized codes.  Three issues:

1. aten.view.default doesn't propagate the appearance dtype, so views (e.g. from
   DTensor.from_local()) revert to float32 while the base is bf16.  torch.compile's
   fake-tensor metadata check then fails (AssertionError: torch.bfloat16 != torch.float32).

2. aten._to_copy doesn't clone internal tensors, so same-device dtype changes
   (e.g. .float()) create an accidental view relationship with the same issue.

3. aten.view.dtype is unimplemented, so if the dtype-view path IS taken, it crashes
   with NotImplementedError.

Fix: propagate dtype in view.default (primary), clone in _to_copy, register view.dtype.

Upstream fix: https://github.com/pytorch/ao/pull/4216
"""

import torch
from torch.utils._python_dispatch import return_and_correct_aliasing

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

aten = torch.ops.aten


def _needs_view_dtype_patch(cls):
    """Check if a subclass is missing aten.view.dtype."""
    op_table = getattr(cls, "_ATEN_OP_TABLE", {}).get(cls, {})
    return aten.view.dtype not in op_table


def patch_torchao_optim_state_8bit():
    """Patch torchao optim subclasses for torch.compile compatibility."""
    try:
        from torchao.optim.subclass_8bit import OptimState8bit
    except ImportError:
        return

    # Patch view.default to propagate appearance dtype
    @OptimState8bit.implements(aten.view.default)
    def _(func, types, args, kwargs):
        x, shape = args
        return OptimState8bit(
            x.codes.view(shape), x.scale, x.qmap, x.signed, dtype=x.dtype
        )

    # Patch _to_copy to clone internal tensors (breaks accidental view)
    @OptimState8bit.implements(aten._to_copy.default)
    def _(func, types, args, kwargs):
        dtype = kwargs.get("dtype", args[0].dtype)
        device = kwargs.get("device", None)
        out = OptimState8bit(
            args[0].codes.to(device=device).clone(),
            args[0].scale.to(device=device).clone(),
            args[0].qmap.to(device=device).clone(),
            args[0].signed,
            dtype=dtype,
        )
        return return_and_correct_aliasing(func, args, kwargs, out)

    if _needs_view_dtype_patch(OptimState8bit):

        @OptimState8bit.implements(aten.view.dtype)
        def _(func, types, args, kwargs):
            x, dtype = args
            return OptimState8bit(x.codes, x.scale, x.qmap, x.signed, dtype=dtype)

    LOG.debug("Patched OptimState8bit for torch.compile compatibility")

    try:
        from torchao.optim.subclass_4bit import OptimState4bit
    except ImportError:
        OptimState4bit = None

    if OptimState4bit is not None:

        @OptimState4bit.implements(aten.view.default)
        def _(func, types, args, kwargs):
            x, shape = args
            if tuple(x.shape) == tuple(shape):
                return OptimState4bit(
                    x.codes, x.scale, x.qmap, x.signed, x._shape, dtype=x.dtype
                )
            if len(shape) == 1 and shape[0] == -1:
                return OptimState4bit(
                    x.codes, x.scale, x.qmap, x.signed, (x.numel(),), dtype=x.dtype
                )
            raise ValueError(
                f"{x.__class__.__name__} only supports .view() with same shape or shape=[-1]"
            )

        @OptimState4bit.implements(aten._to_copy.default)
        def _(func, types, args, kwargs):
            dtype = kwargs.get("dtype", args[0].dtype)
            device = kwargs.get("device", None)
            out = OptimState4bit(
                args[0].codes.to(device=device).clone(),
                args[0].scale.to(device=device).clone(),
                args[0].qmap.to(device=device).clone(),
                args[0].signed,
                args[0].shape,
                dtype=dtype,
            )
            return return_and_correct_aliasing(func, args, kwargs, out)

        if _needs_view_dtype_patch(OptimState4bit):

            @OptimState4bit.implements(aten.view.dtype)
            def _(func, types, args, kwargs):
                x, dtype = args
                return OptimState4bit(
                    x.codes, x.scale, x.qmap, x.signed, x.shape, dtype=dtype
                )

        LOG.debug("Patched OptimState4bit for torch.compile compatibility")

    try:
        from torchao.optim.subclass_fp8 import OptimStateFp8
    except ImportError:
        OptimStateFp8 = None

    if OptimStateFp8 is not None:

        @OptimStateFp8.implements(aten.view.default)
        def _(func, types, args, kwargs):
            x, shape = args
            return OptimStateFp8(x.codes.view(shape), x.scale, dtype=x.dtype)

        @OptimStateFp8.implements(aten._to_copy.default)
        def _(func, types, args, kwargs):
            dtype = kwargs.get("dtype", args[0].dtype)
            device = kwargs.get("device", None)
            out = OptimStateFp8(
                args[0].codes.to(device=device).clone(),
                args[0].scale.to(device=device).clone(),
                dtype=dtype,
            )
            return return_and_correct_aliasing(func, args, kwargs, out)

        if _needs_view_dtype_patch(OptimStateFp8):

            @OptimStateFp8.implements(aten.view.dtype)
            def _(func, types, args, kwargs):
                x, dtype = args
                return OptimStateFp8(x.codes, x.scale, dtype=dtype)

        LOG.debug("Patched OptimStateFp8 for torch.compile compatibility")
