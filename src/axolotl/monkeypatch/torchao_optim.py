"""
Patch for torchao optim subclasses missing aten.view.dtype dispatch.

torchao 0.17.0 PR #3934 added an "appearance dtype" to OptimState{4,8}bit and
OptimStateFp8, allowing them to report as e.g. bf16 while internally storing
quantized codes.  However, it did not implement aten.view.dtype, which PyTorch's
dynamo fake-tensor machinery calls (in meta_utils.py) when a tensor subclass view
has a different dtype than its base.

This causes torch.compile'd optimizer steps to crash with:
  NotImplementedError: OptimState8bit dispatch: ... aten.view.dtype

The fix registers the missing op so it just updates the appearance dtype, matching
the existing aten._to_copy.default behavior.

Upstream issue: https://github.com/pytorch/ao/issues/XXXX
"""

import logging

import torch

logger = logging.getLogger(__name__)

aten = torch.ops.aten


def _needs_patch(cls):
    """Check if a subclass is missing aten.view.dtype."""
    op_table = getattr(cls, "_ATEN_OP_TABLE", {}).get(cls, {})
    return aten.view.dtype not in op_table


def patch_torchao_optim_state_8bit():
    """Register aten.view.dtype for torchao optim subclasses if missing."""
    try:
        from torchao.optim.subclass_8bit import OptimState8bit
    except ImportError:
        return

    if _needs_patch(OptimState8bit):

        @OptimState8bit.implements(aten.view.dtype)
        def _(func, types, args, kwargs):
            x, dtype = args
            return OptimState8bit(x.codes, x.scale, x.qmap, x.signed, dtype=dtype)

        logger.debug("Patched OptimState8bit with aten.view.dtype support")

    try:
        from torchao.optim.subclass_4bit import OptimState4bit
    except ImportError:
        OptimState4bit = None

    if OptimState4bit is not None and _needs_patch(OptimState4bit):

        @OptimState4bit.implements(aten.view.dtype)
        def _(func, types, args, kwargs):
            x, dtype = args
            return OptimState4bit(
                x.codes, x.scale, x.qmap, x.signed, x.shape, dtype=dtype
            )

        logger.debug("Patched OptimState4bit with aten.view.dtype support")

    try:
        from torchao.optim.subclass_fp8 import OptimStateFp8
    except ImportError:
        OptimStateFp8 = None

    if OptimStateFp8 is not None and _needs_patch(OptimStateFp8):

        @OptimStateFp8.implements(aten.view.dtype)
        def _(func, types, args, kwargs):
            x, dtype = args
            return OptimStateFp8(x.codes, x.scale, dtype=dtype)

        logger.debug("Patched OptimStateFp8 with aten.view.dtype support")
