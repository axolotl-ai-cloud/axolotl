# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fused Triton kernels for the SinkGD optimizer family (opt-in: ``sinkgd_fused_kernel``).

One kernel per SR-Sinkhorn iteration replaces the compiled loop's ~4 passes: each CTA applies
the *incoming* column scale on load (the scale is never materialized — it folds forward into
the next kernel, the power-iteration vectors, and the final apply), reduces and applies the
row scale, writes once, and atomically accumulates the next iteration's column sum-of-squares.
Under FSDP2 rows-sharding the cross-rank ``[N]`` all-reduce of that buffer slots between
kernels — the same collective count as the compiled distributed path.

Two grid layouts, auto-selected per shape (``_plan_splits``): the *tall* layout gives each CTA
full rows (row reductions stay in-register); the *wide* layout splits columns across a second
grid axis with fp32-atomic row reductions, for wide-short local shards (heavy sharding) where
the tall grid cannot fill the GPU. Same memory traffic either way.

Numerics are equivalent to the compiled path at bf16 rounding scale (scales are carried in
fp32 end-to-end, so the fused path is marginally *more* precise), but not byte-identical —
hence opt-in.
"""

import torch
import torch.distributed as dist

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # pragma: no cover
    HAVE_TRITON = False


if HAVE_TRITON:

    @triton.jit
    def _iter_tall(
        x_ptr,
        out_ptr,
        colsq_in_ptr,
        colsq_out_ptr,
        M,
        N,
        sqrt_n,
        sqrt_m,
        eps,
        stride_b,
        stride_m,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        base = pid_b * stride_b
        rowsq = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n0 in range(0, N, BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            csq = tl.load(colsq_in_ptr + pid_b * N + cols, mask=col_mask, other=1.0)
            cscale = sqrt_m / tl.maximum(tl.sqrt(csq), eps)
            tile = tl.load(
                x_ptr + base + rows[:, None] * stride_m + cols[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            v = tile.to(tl.float32) * cscale[None, :]
            rowsq += tl.sum(v * v, axis=1)
        rscale = sqrt_n / tl.maximum(tl.sqrt(rowsq), eps)
        for n0 in range(0, N, BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            csq = tl.load(colsq_in_ptr + pid_b * N + cols, mask=col_mask, other=1.0)
            cscale = sqrt_m / tl.maximum(tl.sqrt(csq), eps)
            m2 = row_mask[:, None] & col_mask[None, :]
            tile = tl.load(
                x_ptr + base + rows[:, None] * stride_m + cols[None, :],
                mask=m2,
                other=0.0,
            )
            v = tile.to(tl.float32) * cscale[None, :] * rscale[:, None]
            tl.store(
                out_ptr + base + rows[:, None] * stride_m + cols[None, :],
                v.to(out_ptr.dtype.element_ty),
                mask=m2,
            )
            tl.atomic_add(
                colsq_out_ptr + pid_b * N + cols, tl.sum(v * v, axis=0), mask=col_mask
            )

    @triton.jit
    def _rowsq_wide(
        x_ptr,
        colsq_in_ptr,
        rowsq_ptr,
        M,
        N,
        sqrt_m,
        eps,
        cols_per,
        stride_b,
        stride_m,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        base = pid_b * stride_b
        c0 = pid_n * cols_per
        acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
        for n0 in range(c0, min(c0 + cols_per, N), BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            csq = tl.load(colsq_in_ptr + pid_b * N + cols, mask=col_mask, other=1.0)
            cscale = sqrt_m / tl.maximum(tl.sqrt(csq), eps)
            tile = tl.load(
                x_ptr + base + rows[:, None] * stride_m + cols[None, :],
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            )
            v = tile.to(tl.float32) * cscale[None, :]
            acc += tl.sum(v * v, axis=1)
        tl.atomic_add(rowsq_ptr + pid_b * M + rows, acc, mask=row_mask)

    @triton.jit
    def _scale_wide(
        x_ptr,
        out_ptr,
        colsq_in_ptr,
        rowsq_ptr,
        colsq_out_ptr,
        M,
        N,
        sqrt_n,
        sqrt_m,
        eps,
        cols_per,
        stride_b,
        stride_m,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        base = pid_b * stride_b
        rsq = tl.load(rowsq_ptr + pid_b * M + rows, mask=row_mask, other=1.0)
        rscale = sqrt_n / tl.maximum(tl.sqrt(rsq), eps)
        c0 = pid_n * cols_per
        for n0 in range(c0, min(c0 + cols_per, N), BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            csq = tl.load(colsq_in_ptr + pid_b * N + cols, mask=col_mask, other=1.0)
            cscale = sqrt_m / tl.maximum(tl.sqrt(csq), eps)
            m2 = row_mask[:, None] & col_mask[None, :]
            tile = tl.load(
                x_ptr + base + rows[:, None] * stride_m + cols[None, :],
                mask=m2,
                other=0.0,
            )
            v = tile.to(tl.float32) * cscale[None, :] * rscale[:, None]
            tl.store(
                out_ptr + base + rows[:, None] * stride_m + cols[None, :],
                v.to(out_ptr.dtype.element_ty),
                mask=m2,
            )
            tl.atomic_add(
                colsq_out_ptr + pid_b * N + cols, tl.sum(v * v, axis=0), mask=col_mask
            )

    @triton.jit
    def _matvec1_partials(
        x_ptr,
        colsq_ptr,
        p_ptr,
        u_ptr,
        v_ptr,
        sums_ptr,
        M,
        N,
        sqrt_m,
        eps,
        cols_per,
        stride_b,
        stride_m,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        WITH_PARTIALS: tl.constexpr,
        ATOMIC_V: tl.constexpr,
    ):
        # v = (X . colscale) @ u on the CTA's rows; optionally the three projection
        # partials ||p||^2, <p, Xd>, ||Xd||^2 (per-matrix scalar atomics) in the same pass.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        base = pid_b * stride_b
        c0 = pid_n * cols_per
        acc_v = tl.zeros((BLOCK_M,), dtype=tl.float32)
        s0 = 0.0
        s1 = 0.0
        s2 = 0.0
        for n0 in range(c0, min(c0 + cols_per, N), BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            csq = tl.load(colsq_ptr + pid_b * N + cols, mask=col_mask, other=1.0)
            cscale = sqrt_m / tl.maximum(tl.sqrt(csq), eps)
            m2 = row_mask[:, None] & col_mask[None, :]
            xt = (
                tl.load(
                    x_ptr + base + rows[:, None] * stride_m + cols[None, :],
                    mask=m2,
                    other=0.0,
                ).to(tl.float32)
                * cscale[None, :]
            )
            ut = tl.load(u_ptr + pid_b * N + cols, mask=col_mask, other=0.0)
            acc_v += tl.sum(xt * ut[None, :], axis=1)
            if WITH_PARTIALS:
                pt = tl.load(
                    p_ptr + base + rows[:, None] * stride_m + cols[None, :],
                    mask=m2,
                    other=0.0,
                ).to(tl.float32)
                s0 += tl.sum(pt * pt)
                s1 += tl.sum(pt * xt)
                s2 += tl.sum(xt * xt)
        if ATOMIC_V:
            tl.atomic_add(v_ptr + pid_b * M + rows, acc_v, mask=row_mask)
        else:
            tl.store(v_ptr + pid_b * M + rows, acc_v, mask=row_mask)
        if WITH_PARTIALS:
            tl.atomic_add(sums_ptr + pid_b * 3 + 0, s0)
            tl.atomic_add(sums_ptr + pid_b * 3 + 1, s1)
            tl.atomic_add(sums_ptr + pid_b * 3 + 2, s2)

    @triton.jit
    def _matvec2(
        x_ptr,
        colsq_ptr,
        v_ptr,
        uu_ptr,
        M,
        N,
        sqrt_m,
        eps,
        cols_per,
        stride_b,
        stride_m,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # uu = v^T (X . colscale), accumulated per column via atomics.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        base = pid_b * stride_b
        vt = tl.load(v_ptr + pid_b * M + rows, mask=row_mask, other=0.0)
        c0 = pid_n * cols_per
        for n0 in range(c0, min(c0 + cols_per, N), BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            csq = tl.load(colsq_ptr + pid_b * N + cols, mask=col_mask, other=1.0)
            cscale = sqrt_m / tl.maximum(tl.sqrt(csq), eps)
            m2 = row_mask[:, None] & col_mask[None, :]
            xt = (
                tl.load(
                    x_ptr + base + rows[:, None] * stride_m + cols[None, :],
                    mask=m2,
                    other=0.0,
                ).to(tl.float32)
                * cscale[None, :]
            )
            tl.atomic_add(
                uu_ptr + pid_b * N + cols,
                tl.sum(xt * vt[:, None], axis=0),
                mask=col_mask,
            )

    @triton.jit
    def _apply(
        p_ptr,
        x_ptr,
        colsq_ptr,
        a_ptr,
        scale2_ptr,
        M,
        N,
        decay,
        sqrt_m,
        eps,
        cols_per,
        stride_b,
        stride_m,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        # p = (p * decay - a * (x . colscale)) * scale2 — one form covers all modes:
        # base/spec: scale2 = 1, decay = 1 - lr*wd, a = lr[*target/sigma];
        # md sphere: decay = 1, a = lr*tn/sigma, scale2 = tn/fro (analytic projection).
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        pid_b = tl.program_id(2)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = rows < M
        base = pid_b * stride_b
        a = tl.load(a_ptr + pid_b)
        s2c = tl.load(scale2_ptr + pid_b)
        c0 = pid_n * cols_per
        for n0 in range(c0, min(c0 + cols_per, N), BLOCK_N):
            cols = n0 + tl.arange(0, BLOCK_N)
            col_mask = cols < N
            csq = tl.load(colsq_ptr + pid_b * N + cols, mask=col_mask, other=1.0)
            cscale = sqrt_m / tl.maximum(tl.sqrt(csq), eps)
            m2 = row_mask[:, None] & col_mask[None, :]
            pp = p_ptr + base + rows[:, None] * stride_m + cols[None, :]
            xt = (
                tl.load(
                    x_ptr + base + rows[:, None] * stride_m + cols[None, :],
                    mask=m2,
                    other=0.0,
                ).to(tl.float32)
                * cscale[None, :]
            )
            pt = tl.load(pp, mask=m2, other=0.0).to(tl.float32)
            tl.store(
                pp, ((pt * decay - a * xt) * s2c).to(p_ptr.dtype.element_ty), mask=m2
            )


_SM_COUNT = None


def _num_sms() -> int:
    global _SM_COUNT  # noqa: PLW0603
    if _SM_COUNT is None:
        _SM_COUNT = torch.cuda.get_device_properties(
            torch.cuda.current_device()
        ).multi_processor_count
    return _SM_COUNT


def _plan_splits(B, M, N, block_m, block_n, target_waves=2):
    """Column splits so the grid fills ~target_waves x SMs. Wide only pays when the grid is
    row-starved AND rows are long (short rows: the extra kernel+memset per iteration loses)."""
    row_ctas = triton.cdiv(M, block_m) * B
    want = target_waves * _num_sms()
    if row_ctas >= want or N < 4096:
        return 1, N
    nsplits = min(triton.cdiv(want, row_ctas), triton.cdiv(N, block_n))
    cols_per = triton.cdiv(triton.cdiv(N, block_n), nsplits) * block_n
    return triton.cdiv(N, cols_per), cols_per


def fused_available() -> bool:
    return HAVE_TRITON and torch.cuda.is_available()


@torch.no_grad()
def fused_sinkgd_step(
    p,
    grad,
    lr,
    weight_decay,
    iters,
    eps,
    *,
    mode="base",
    u=None,
    target_norm=None,
    spectral_target="muon",
    sn_iters=1,
    m_global=None,
    process_group=None,
    block_m=16,
    block_n=512,
):
    """One fused SinkGD step; single-device when ``process_group`` is None, else the
    rows-sharded FSDP2 step (``p``/``grad`` are local shards, ``m_global`` the full row dim,
    cross-rank reduction of the column/power-iteration vectors over the group).

    mode: ``base`` | ``spec`` | ``md``. ``u`` is the persisted ``[*lead, N]`` power-iteration
    vector (spec/md), ``target_norm`` the ``[*lead]`` sphere radius (md). ``lr`` is a float
    that already includes ``alpha_eff``; weight decay applies to base/spec (md is on the
    sphere, decay is meaningless there — ignored to match the compiled MD path).
    """
    if not p.is_contiguous():
        # kernels address p with dense strides and write it in place; a .contiguous()
        # copy here would silently drop the update
        raise ValueError("fused_sinkgd_step requires a contiguous param")
    orig = grad.shape
    g = grad.reshape(-1, *orig[-2:]) if grad.ndim > 2 else grad.unsqueeze(0)
    pw = p.reshape(-1, *orig[-2:]) if p.ndim > 2 else p.unsqueeze(0)
    if not g.is_contiguous():
        g = g.contiguous()
    B, m_local, n = g.shape
    m_glob = m_global if m_global is not None else m_local
    sqrt_n, sqrt_m = float(n) ** 0.5, float(m_glob) ** 0.5
    dev = g.device

    x_buf = torch.empty_like(g)
    colsq_a = torch.full((B, n), float(m_glob), device=dev, dtype=torch.float32)
    colsq_b = torch.empty(B, n, device=dev, dtype=torch.float32)
    nsplits, cols_per = _plan_splits(B, m_local, n, block_m, block_n)
    grid = (triton.cdiv(m_local, block_m), nsplits, B)

    src = g
    rowsq = None
    for _ in range(iters):
        colsq_b.zero_()
        if nsplits == 1:
            _iter_tall[(grid[0], B)](
                src,
                x_buf,
                colsq_a,
                colsq_b,
                m_local,
                n,
                sqrt_n,
                sqrt_m,
                eps,
                m_local * n,
                n,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )
        else:
            if rowsq is None:
                rowsq = torch.empty(B, m_local, device=dev, dtype=torch.float32)
            rowsq.zero_()
            _rowsq_wide[grid](
                src,
                colsq_a,
                rowsq,
                m_local,
                n,
                sqrt_m,
                eps,
                cols_per,
                m_local * n,
                n,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )
            _scale_wide[grid](
                src,
                x_buf,
                colsq_a,
                rowsq,
                colsq_b,
                m_local,
                n,
                sqrt_n,
                sqrt_m,
                eps,
                cols_per,
                m_local * n,
                n,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
            )
        if process_group is not None:
            dist.all_reduce(colsq_b, group=process_group)
        src = x_buf
        colsq_a, colsq_b = colsq_b, colsq_a

    decay = 1.0 - lr * weight_decay
    if mode == "base":
        a = torch.full((B,), lr, device=dev, dtype=torch.float32)
        scale2 = torch.ones(B, device=dev, dtype=torch.float32)
        _apply[grid](
            pw,
            x_buf,
            colsq_a,
            a,
            scale2,
            m_local,
            n,
            decay,
            sqrt_m,
            eps,
            cols_per,
            m_local * n,
            n,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
        return

    # spec/md: power iteration on the (colscale-folded) update, sn_iters rounds
    uw = u.reshape(-1, n)
    v = torch.empty(B, m_local, device=dev, dtype=torch.float32)
    sums = torch.zeros(B, 3, device=dev, dtype=torch.float32)
    sigma = None
    for it in range(sn_iters):
        with_partials = mode == "md" and it == 0
        if nsplits > 1:
            v.zero_()
        _matvec1_partials[grid](
            x_buf,
            colsq_a,
            pw,
            uw,
            v,
            sums,
            m_local,
            n,
            sqrt_m,
            eps,
            cols_per,
            m_local * n,
            n,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
            WITH_PARTIALS=with_partials,
            ATOMIC_V=nsplits > 1,
        )
        vsq = (v * v).sum(dim=-1)
        if with_partials:
            packed = torch.cat([sums, vsq.unsqueeze(-1)], dim=-1)
            if process_group is not None:
                dist.all_reduce(packed, group=process_group)
            sums = packed[:, :3]
            vsq = packed[:, 3]
        elif process_group is not None:
            dist.all_reduce(vsq, group=process_group)
        v /= vsq.sqrt().clamp_min(eps).unsqueeze(-1)
        uu = torch.zeros(B, n, device=dev, dtype=torch.float32)
        _matvec2[grid](
            x_buf,
            colsq_a,
            v,
            uu,
            m_local,
            n,
            sqrt_m,
            eps,
            cols_per,
            m_local * n,
            n,
            BLOCK_M=block_m,
            BLOCK_N=block_n,
        )
        if process_group is not None:
            dist.all_reduce(uu, group=process_group)
        sigma = uu.norm(dim=-1, keepdim=True).clamp_min(eps)
        uw.copy_(uu / sigma)
    sig = sigma.squeeze(-1)

    if mode == "spec":
        tgt = (m_glob / n) ** 0.5 if spectral_target == "muon" else 1.0
        a = (lr * tgt / sig).contiguous()
        scale2 = torch.ones_like(a)
    else:
        tn = target_norm.reshape(-1).float().to(dev)
        a = (lr * tn / sig).contiguous()
        fro = (
            (sums[:, 0] - 2 * a * sums[:, 1] + a * a * sums[:, 2]).clamp_min(eps).sqrt()
        )
        scale2 = (tn / fro).contiguous()
        decay = 1.0  # sphere projection subsumes any decay
    _apply[grid](
        pw,
        x_buf,
        colsq_a,
        a,
        scale2,
        m_local,
        n,
        decay,
        sqrt_m,
        eps,
        cols_per,
        m_local * n,
        n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
    )
