"""Make torchao FP4 tensors support ``aten.add`` by dequantizing.

PEFT (>= 0.19) applies ``target_parameters`` LoRA to an expert weight via
``torch.nn.utils.parametrize``: it registers a parametrization whose forward computes
``base + delta`` (the merged ``scaling * B @ A``). ``register_parametrization`` evaluates
this immediately. When the base weight is a frozen torchao ``NVFP4Tensor`` / ``MXTensor``
(e.g. `block_diffusion.frozen_fp4_experts`), ``aten.add`` is unimplemented for the tensor
subclass, so the registration raises::

    NotImplementedError: NVFP4Tensor dispatch: ... aten.add ...

Registering a dequantize-then-add for these tensors makes the parametrization produce a
bf16 merged weight (differentiable through ``delta`` to the LoRA A/B), so ScatterMoE then
runs the merged weight via its standard path. Idempotent; safe to call repeatedly.
"""

from __future__ import annotations


def _make_add_handler():
    import torch

    fp4_names = {"NVFP4Tensor", "MXTensor"}

    def _is_fp4(t):
        return type(t).__name__ in fp4_names

    def _fp4_add(func, types, args, kwargs):
        a, b = args[0], args[1]
        # Use the dense operand's dtype (the LoRA delta is bf16/fp16) so the result is a plain
        # tensor; dequantize only the FP4 subclass operand(s).
        dense_dtype = next(
            (t.dtype for t in (a, b) if isinstance(t, torch.Tensor) and not _is_fp4(t)),
            torch.bfloat16,
        )
        if _is_fp4(a):
            a = a.dequantize(dense_dtype)
        if _is_fp4(b):
            b = b.dequantize(dense_dtype)
        # A true in-place add_ is impossible once the FP4 operand is dequantized to a fresh tensor,
        # so both add and add_ resolve to this out-of-place add (kwargs forward keyword-only alpha).
        return torch.ops.aten.add.Tensor(a, b, **kwargs)

    return _fp4_add


def patch_torchao_fp4_add() -> None:
    """Register dequantize-add for ``aten.add``/``aten.add_`` on NVFP4Tensor and MXTensor.

    ``add.Tensor`` covers PEFT's out-of-place ``base + delta`` parametrization (training);
    ``add_.Tensor`` covers ``ParamWrapper.merge`` (``merge_and_unload`` produces a bf16
    weight — a true 4-bit merge would require re-quantization).
    """
    import torch

    handler = _make_add_handler()
    ops = [torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor]

    for module_path, cls_name in (
        ("torchao.prototype.mx_formats.nvfp4_tensor", "NVFP4Tensor"),
        ("torchao.prototype.mx_formats.mx_tensor", "MXTensor"),
    ):
        try:
            module = __import__(module_path, fromlist=[cls_name])
            cls = getattr(module, cls_name)
        except (ImportError, AttributeError):
            continue
        # torchao's op table is nested: cls._ATEN_OP_TABLE[cls][op] (see
        # torchao.utils._dispatch__torch_dispatch__).
        registered = getattr(cls, "_ATEN_OP_TABLE", {}).get(cls, {})
        for op in ops:
            if op not in registered:
                cls.implements([op])(handler)
