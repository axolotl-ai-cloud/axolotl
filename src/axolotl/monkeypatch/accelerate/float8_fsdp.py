"""FSDP2 support for torchao ``Float8Tensor`` (frozen blockwise-FP8 weights, dim-0 sharded).

A weight-only ``Float8Tensor`` (e.g. a pre-quantized blockwise-FP8 checkpoint wrapped for
1-byte storage) can't be FSDP2-sharded as torchao ships it: its ``aten.split`` unpacks
``(tensor, size, dim)`` unconditionally, so FSDP2's ``torch.chunk(param, n)`` (dim defaults
to 0) raises ``ValueError: not enough values to unpack``; and even with an explicit dim,
torchao only implements the rowwise / per-tensor split cases — a 128×128-blocked weight
hits ``else: raise AssertionError("not yet implemented")``. It also ships no FSDP2
all-gather hooks.

This patches, for a 2-D ``[N, K]`` weight with a blockwise ``[N//b0, K//b1]`` scale sharded
on dim 0 (the output-feature axis):
  * ``aten.split`` with a defaulted dim — qdata splits by ``s``; the scale splits by
    ``s // block_size[0]`` (the scale is coarser than qdata by ``block_size[0]``),
  * ``new_zeros`` / ``as_strided`` / ``copy_`` / ``clone`` / ``detach`` / ``narrow`` /
    ``view`` on the inner ``(qdata, scale)``,
  * ``fsdp_pre_all_gather`` / ``fsdp_post_all_gather`` so FSDP2 all-gathers the inner tensors
    and rebuilds the subclass instead of using the flat ``view(-1)`` buffer path.

Idempotent; a no-op if torchao is absent. Dim-0 sharding only (assumes the shard size is a
multiple of ``block_size[0]`` — true for FSDP of these projections, whose N is a multiple of
128×world_size). The blockwise split case is worth upstreaming to torchao.
"""

from __future__ import annotations

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)
_PATCHED = False


def _float8_cls():
    try:
        from torchao.quantization import Float8Tensor

        return Float8Tensor
    except ImportError:
        return None


def _cstride(shape):
    st = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        st[i] = st[i + 1] * shape[i + 1]
    return st


def patch_float8_fsdp():
    global _PATCHED
    if _PATCHED:
        return
    Float8Tensor = _float8_cls()
    if Float8Tensor is None:
        return
    aten = torch.ops.aten
    implements = Float8Tensor.implements

    def _rebuild(ref, qdata, scale, block_size=None):
        return Float8Tensor(
            qdata,
            scale,
            list(block_size) if block_size is not None else list(ref.block_size),
            ref.mm_config,
            ref.act_quant_kwargs,
            ref.kernel_preference,
            dtype=ref.dtype,
        )

    @implements([aten.split.Tensor])
    def _split(func, types, args, kwargs):
        x, split_size = args[0], args[1]
        dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
        if dim != 0:
            raise NotImplementedError(
                f"Float8Tensor FSDP split only on dim 0, got {dim}"
            )
        blk0 = x.block_size[0]
        qd = func(x.qdata, split_size, 0)
        sc = func(x.scale, max(1, split_size // blk0), 0)
        assert len(qd) == len(sc), f"split mismatch q={len(qd)} s={len(sc)}"
        return [_rebuild(x, qd[i], sc[i]) for i in range(len(qd))]

    @implements([aten.clone.default, aten.detach.default])
    def _clone_detach(func, types, args, kwargs):
        x = args[0]
        return _rebuild(x, func(x.qdata, **kwargs), func(x.scale, **kwargs))

    @implements([aten.new_zeros.default])
    def _new_zeros(func, types, args, kwargs):
        x, size = args[0], list(args[1])
        qd = func(x.qdata, size, **kwargs)
        sc_size = [max(1, size[i] // x.block_size[i]) for i in range(len(size))]
        sc = func(x.scale, sc_size, **kwargs)
        return _rebuild(x, qd, sc)

    @implements([aten.narrow.default])
    def _narrow(func, types, args, kwargs):
        x, dim, start, length = args[0], args[1], args[2], args[3]
        if dim != 0:
            raise NotImplementedError(f"Float8Tensor narrow only dim 0, got {dim}")
        blk0 = x.block_size[0]
        qd = func(x.qdata, 0, start, length)
        sc = func(x.scale, 0, start // blk0, max(1, length // blk0))
        return _rebuild(x, qd, sc)

    @implements([aten.view.default])
    def _view(func, types, args, kwargs):
        x, size = args[0], list(args[1])
        if size == [-1]:
            # FSDP's flat-buffer view; unused for the subclass-extension all-gather path.
            return _rebuild(
                x,
                x.qdata.reshape(1, -1),
                x.scale.reshape(1, -1),
                block_size=[1, x.scale.numel()],
            )
        sc_size = [size[i] // x.block_size[i] for i in range(len(size))]
        return _rebuild(x, func(x.qdata, size), func(x.scale, sc_size))

    @implements([aten.as_strided.default])
    def _as_strided(func, types, args, kwargs):
        x = args[0]
        qs, ss = list(x.qdata.shape), list(x.scale.shape)
        qd = func(x.qdata, qs, _cstride(qs), 0)
        sc = func(x.scale, ss, _cstride(ss), 0)
        return _rebuild(x, qd, sc)

    @implements([aten.copy_.default])
    def _copy_(func, types, args, kwargs):
        dst, src = args[0], args[1]
        func(dst.qdata, src.qdata)
        func(dst.scale, src.scale)
        return dst

    def fsdp_pre_all_gather(
        self, mesh, outer_size=None, outer_stride=None, module=None, mp_policy=None
    ):
        return (self.qdata, self.scale), (tuple(self.block_size),)

    def fsdp_post_all_gather(
        self, all_gather_outputs, metadata, param_dtype, *, out=None
    ):
        (block_size,) = metadata
        qdata, scale = all_gather_outputs
        if out is not None:
            out.qdata, out.scale = qdata, scale
            return
        return _rebuild(self, qdata, scale, list(block_size)), all_gather_outputs

    Float8Tensor.fsdp_pre_all_gather = fsdp_pre_all_gather
    Float8Tensor.fsdp_post_all_gather = fsdp_post_all_gather

    _PATCHED = True
    LOG.info(
        "Installed FSDP2 support (split + all-gather hooks) on torchao Float8Tensor"
    )
