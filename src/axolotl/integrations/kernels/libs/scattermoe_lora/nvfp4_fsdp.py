"""FSDP2 support for torchao ``NVFP4Tensor`` (frozen, expert-sharded on dim 0).

torchao's NVFP4Tensor (prototype) ships no FSDP2 hooks, so `fully_shard` on a module with
NVFP4 expert params fails: the flat-param sharding path calls ops the subclass doesn't
implement (``aten.split`` etc.). This adds:

  * ``aten.split`` / ``aten.clone`` / ``aten.detach`` / ``aten.copy_`` operating on the
    inner ``(qdata, scale, per_tensor_scale)`` along dim 0 (the expert axis FSDP shards),
  * ``fsdp_pre_all_gather`` / ``fsdp_post_all_gather`` so FSDP2 uses the subclass-extension
    path (all-gathering the inner tensors and reconstructing the subclass) instead of the
    flat-buffer ``view(-1)`` path, which a block-scaled FP4 layout can't satisfy.

Idempotent. Sharding is restricted to dim 0 (the [E, N, K] expert axis) — block-axis
sharding is not supported (nor needed for expert-parallel/FSDP of MoE weights).
"""

from __future__ import annotations

import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)
_PATCHED = False


def _nvfp4_cls():
    try:
        from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor

        return NVFP4Tensor
    except ImportError:
        return None


def _rebuild(ref, qdata, scale, per_tensor_scale):
    NVFP4Tensor = type(ref)
    return NVFP4Tensor(
        qdata,
        scale,
        ref.block_size,
        ref.orig_dtype,
        per_tensor_scale,
        ref.act_per_tensor_scale,
        ref.is_swizzled_scales,
        ref.use_triton_kernel,
        ref.act_quant_kwargs,
    )


def patch_nvfp4_fsdp():
    global _PATCHED
    if _PATCHED:
        return
    NVFP4Tensor = _nvfp4_cls()
    if NVFP4Tensor is None:
        return
    aten = torch.ops.aten
    implements = NVFP4Tensor.implements

    def _pts_along_dim0(x, n):
        """Per-tensor scale handling under a dim-0 split: per-expert -> split, else share."""
        pts = x.per_tensor_scale
        if pts is not None and pts.dim() >= 1 and pts.shape[0] == x.shape[0]:
            return list(torch.split(pts, n, 0)) if isinstance(n, int) else None
        return None

    @implements([aten.split.Tensor])
    def _split(func, types, args, kwargs):
        x, split_size = args[0], args[1]
        dim = args[2] if len(args) > 2 else kwargs.get("dim", 0)
        if dim != 0:
            raise NotImplementedError(
                f"NVFP4Tensor FSDP split only on dim 0, got {dim}"
            )
        qd = func(x.qdata, split_size, 0)
        sc = func(x.scale, split_size, 0)
        pts = _pts_along_dim0(x, split_size)
        return [
            _rebuild(x, qd[i], sc[i], pts[i] if pts is not None else x.per_tensor_scale)
            for i in range(len(qd))
        ]

    @implements([aten.clone.default, aten.detach.default])
    def _clone_detach(func, types, args, kwargs):
        x = args[0]
        return x._apply_fn_to_data(lambda t: func(t, **kwargs))

    @implements([aten.new_zeros.default])
    def _new_zeros(func, types, args, kwargs):
        # FSDP pads on dim 0 (experts). Scale may be SWIZZLED (its trailing dims are not
        # K//block), so preserve qdata/scale trailing shapes and vary only dim 0.
        x, size = args[0], list(args[1])
        E = size[0]
        qd = func(x.qdata, [E, *x.qdata.shape[1:]], **kwargs)
        sc = func(x.scale, [E, *x.scale.shape[1:]], **kwargs)
        # Preserve a per-expert per_tensor_scale buffer (vary dim 0) so the subsequent
        # copy_/all-gather can carry it; dropping it here loses it for the whole param.
        pts = x.per_tensor_scale
        if pts is not None and pts.dim() >= 1 and pts.shape[0] == x.shape[0]:
            pts = func(pts, [E, *pts.shape[1:]], **kwargs)
        return _rebuild(x, qd, sc, pts)

    @implements([aten.narrow.default])
    def _narrow(func, types, args, kwargs):
        x, dim, start, length = args[0], args[1], args[2], args[3]
        if dim != 0:
            raise NotImplementedError(f"NVFP4 narrow only dim 0, got {dim}")
        qd = func(x.qdata, 0, start, length)
        sc = func(x.scale, 0, start, length)
        pts = x.per_tensor_scale
        if pts is not None and pts.dim() >= 1 and pts.shape[0] == x.shape[0]:
            pts = func(pts, 0, start, length)
        return _rebuild(x, qd, sc, pts)

    @implements([aten.view.default])
    def _view(func, types, args, kwargs):
        x, size = args[0], list(args[1])
        if size == [-1]:
            # FSDP's flat-buffer view. A 1-D NVFP4 is invalid (block layout needs >=2D),
            # and this buffer is unused for subclass-extension params (the all-gather goes
            # through fsdp_pre/post_all_gather), so return a valid 2-D [1, numel] NVFP4.
            return _rebuild(
                x, x.qdata.reshape(1, -1), x.scale.reshape(1, -1), x.per_tensor_scale
            )
        K = size[-1]
        qd = func(x.qdata, size[:-1] + [K // 2])
        sc = func(x.scale, size[:-1] + [K // x.block_size])
        return _rebuild(x, qd, sc, x.per_tensor_scale)

    @implements([aten.slice.Tensor])
    def _slice(func, types, args, kwargs):
        x = args[0]
        dim = args[1] if len(args) > 1 else 0
        if dim == 0:  # expert-axis slice (FSDP shard); torchao only handles rank 2
            start = args[2] if len(args) > 2 else None
            end = args[3] if len(args) > 3 else None
            step = args[4] if len(args) > 4 else 1
            if step != 1:
                raise NotImplementedError("NVFP4 slice step must be 1")
            qd = func(x.qdata, 0, start, end, 1)
            sc = func(x.scale, 0, start, end, 1)
            pts = x.per_tensor_scale
            if pts is not None and pts.dim() >= 1 and pts.shape[0] == x.shape[0]:
                pts = func(pts, 0, start, end, 1)
            return _rebuild(x, qd, sc, pts)
        from torchao.prototype.mx_formats.nvfp4_tensor import nvfp4_slice

        return nvfp4_slice(func, types, args, kwargs)

    def _cstride(shape):
        st = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            st[i] = st[i + 1] * shape[i + 1]
        return st

    @implements([aten.as_strided.default])
    def _as_strided(func, types, args, kwargs):
        # FSDP reconstructs the contiguous unsharded param via as_strided(orig_size,
        # contiguous_stride, 0). The reconstructed NVFP4 already has the right qdata/scale
        # (incl. swizzled scale) shapes; as_strided them to their own contiguous shapes (a
        # view op must return NEW tensors, so we can't return x as-is).
        x = args[0]
        qshape, sshape = list(x.qdata.shape), list(x.scale.shape)
        qd = func(x.qdata, qshape, _cstride(qshape), 0)
        sc = func(x.scale, sshape, _cstride(sshape), 0)
        return _rebuild(x, qd, sc, x.per_tensor_scale)

    @implements([aten.copy_.default])
    def _copy_(func, types, args, kwargs):
        dst, src = args[0], args[1]
        func(dst.qdata, src.qdata)
        func(dst.scale, src.scale)
        if src.per_tensor_scale is not None:
            if (
                dst.per_tensor_scale is not None
                and dst.per_tensor_scale.shape == src.per_tensor_scale.shape
            ):
                func(dst.per_tensor_scale, src.per_tensor_scale)
            else:
                # dst lost its per_tensor buffer (e.g. a new_zeros without one); adopt src's.
                dst.per_tensor_scale = src.per_tensor_scale.clone()
        return dst

    # FSDP2 subclass-extension hooks avoid the flat-buffer view(-1) path.
    def fsdp_pre_all_gather(
        self, mesh, outer_size=None, outer_stride=None, module=None, mp_policy=None
    ):
        inputs = (self.qdata, self.scale)
        if self.per_tensor_scale is not None:
            inputs = inputs + (self.per_tensor_scale,)
        meta = (self.per_tensor_scale is not None,)
        return inputs, meta

    def fsdp_post_all_gather(
        self, all_gather_outputs, metadata, param_dtype, *, out=None
    ):
        (has_pts,) = metadata
        if has_pts:
            qdata, scale, pts = all_gather_outputs
        else:
            qdata, scale = all_gather_outputs
            pts = None
        if out is not None:
            # reconstruct in-place into the existing unsharded param
            out.qdata, out.scale, out.per_tensor_scale = qdata, scale, pts
            return
        return _rebuild(self, qdata, scale, pts), all_gather_outputs

    NVFP4Tensor.fsdp_pre_all_gather = fsdp_pre_all_gather
    NVFP4Tensor.fsdp_post_all_gather = fsdp_post_all_gather

    _PATCHED = True
    LOG.info("Installed FSDP2 support (split + all-gather hooks) on NVFP4Tensor")
