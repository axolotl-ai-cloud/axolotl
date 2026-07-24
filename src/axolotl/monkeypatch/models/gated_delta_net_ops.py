"""Opaque custom ops wrapping the FLA GatedDeltaNet training kernels: they take
``position_ids`` and derive ``cu_seqlens`` eagerly inside the opaque region so
the data-dependent ``aten.nonzero`` never enters the compile graph.
"""

from __future__ import annotations

import torch

__all__ = ["fla_ops_available", "fla_ops_build_error"]

_OPS_BUILT = False
_OPS_BUILD_ERROR: str | None = None


def _check_varlen_batch_size(x: torch.Tensor, position_ids: torch.Tensor | None):
    # The wrapped raw FLA functions only guard this with an `assert` (stripped under -O), which would make B>1 silent garbage.
    if position_ids is not None and x.shape[0] != 1:
        raise ValueError(
            f"The batch size is expected to be 1 rather than {x.shape[0]} when using "
            "packed (varlen) inputs. Please flatten variable-length inputs before processing."
        )


def _cu_seqlens_from_position_ids(position_ids: torch.Tensor) -> torch.Tensor:
    if position_ids.ndim == 3:  # MRoPE [axes, B, T]: all axes share temporal pos
        position_ids = position_ids[0]
    pos = position_ids.reshape(-1)
    tensor_kwargs = {"dtype": torch.int32, "device": pos.device}
    indices_q = (pos == 0).nonzero().view(-1)
    return torch.cat(
        (
            indices_q.to(**tensor_kwargs),
            torch.tensor(pos.size(), **tensor_kwargs),
        )
    )


def _build_ops() -> None:
    """Register the custom ops once. Raises if FLA is unavailable."""
    global _OPS_BUILT
    if _OPS_BUILT:
        return

    from fla.modules.convolution import causal_conv1d_bwd, causal_conv1d_fwd
    from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
    from fla.ops.gated_delta_rule.chunk import (
        chunk_gated_delta_rule_bwd,
        chunk_gated_delta_rule_fwd,
    )

    def _cu(position_ids: torch.Tensor | None) -> torch.Tensor | None:
        if position_ids is None:
            return None
        return _cu_seqlens_from_position_ids(position_ids)

    @torch.library.custom_op("axolotl_gdn::gdn_conv", mutates_args=())
    def _gdn_conv(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        activation: str | None,
        position_ids: torch.Tensor | None,
    ) -> torch.Tensor:
        _check_varlen_batch_size(x, position_ids)
        y, _ = causal_conv1d_fwd(
            x=x.contiguous(),
            weight=weight.contiguous(),
            bias=bias.contiguous() if bias is not None else None,
            residual=None,
            initial_state=None,
            output_final_state=False,
            activation=activation,
            cu_seqlens=_cu(position_ids),
        )
        return y

    @_gdn_conv.register_fake
    def _(x, weight, bias, activation, position_ids):
        return torch.empty(x.shape, dtype=x.dtype, device=x.device)

    @torch.library.custom_op("axolotl_gdn::gdn_conv_bwd", mutates_args=())
    def _gdn_conv_bwd(
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        dy: torch.Tensor,
        activation: str | None,
        position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dx, dw, db, _, _ = causal_conv1d_bwd(
            x=x.contiguous(),
            dy=dy.contiguous(),
            dht=None,
            weight=weight.contiguous(),
            bias=bias.contiguous() if bias is not None else None,
            residual=None,
            initial_state=None,
            activation=activation,
            cu_seqlens=_cu(position_ids),
        )
        if db is None:
            db = weight.new_empty(0)
        return dx, dw, db

    @_gdn_conv_bwd.register_fake
    def _(x, weight, bias, dy, activation, position_ids):
        db_shape = bias.shape if bias is not None else (0,)
        db_dtype = bias.dtype if bias is not None else weight.dtype
        return (
            torch.empty(x.shape, dtype=x.dtype, device=x.device),
            torch.empty(weight.shape, dtype=weight.dtype, device=weight.device),
            torch.empty(db_shape, dtype=db_dtype, device=weight.device),
        )

    def _gdn_conv_setup(ctx, inputs, output):
        x, weight, bias, activation, position_ids = inputs
        ctx.activation = activation
        ctx.has_bias = bias is not None
        ctx.save_for_backward(x, weight, bias, position_ids)

    def _gdn_conv_backward(ctx, dy):
        x, weight, bias, position_ids = ctx.saved_tensors
        dx, dw, db = torch.ops.axolotl_gdn.gdn_conv_bwd(
            x, weight, bias, dy, ctx.activation, position_ids
        )
        return dx, dw, (db if ctx.has_bias else None), None, None

    _gdn_conv.register_autograd(_gdn_conv_backward, setup_context=_gdn_conv_setup)

    @torch.library.custom_op("axolotl_gdn::gdn_chunk", mutates_args=())
    def _gdn_chunk(
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        position_ids: torch.Tensor | None,
        cast_g: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        _check_varlen_batch_size(q, position_ids)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        # cast_g mirrors each architecture's eager kernel call: qwen3_5 casts g to the
        # input dtype before chunk_gated_delta_rule; qwen3_next passes g (f32) unchanged.
        # The op's g input stays f32 either way, so the returned dg is f32 — matching eager grad flow.
        g, beta = g.contiguous(), beta.contiguous()
        if cast_g:
            g = g.to(q.dtype)
        qn, q_rstd = l2norm_fwd(q)
        kn, k_rstd = l2norm_fwd(k)
        g_cum, o, A, _ = chunk_gated_delta_rule_fwd(
            q=qn,
            k=kn,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=None,
            output_final_state=False,
            cu_seqlens=_cu(position_ids),
        )
        return o.to(q.dtype), qn, q_rstd, kn, k_rstd, g_cum, A

    @_gdn_chunk.register_fake
    def _(q, k, v, g, beta, scale, position_ids, cast_g):
        def emp(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=q.device)

        return (
            emp(v.shape, q.dtype),
            emp(q.shape, q.dtype),
            emp(q.shape[:-1], torch.float32),
            emp(k.shape, k.dtype),
            emp(k.shape[:-1], torch.float32),
            emp(g.shape, torch.float32),  # g_cum (chunk_local_cumsum output_dtype)
            emp((*k.shape[:-1], 64), k.dtype),  # A (chunk 64, solve_tril -> k dtype)
        )

    @torch.library.custom_op("axolotl_gdn::gdn_chunk_bwd", mutates_args=())
    def _gdn_chunk_bwd(
        qn: torch.Tensor,
        q_rstd: torch.Tensor,
        kn: torch.Tensor,
        k_rstd: torch.Tensor,
        v: torch.Tensor,
        g_cum: torch.Tensor,
        beta: torch.Tensor,
        A: torch.Tensor,
        do: torch.Tensor,
        scale: float,
        position_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Mirror FLA's input_guard: v arrives as a non-contiguous split/reshape view, and the stride-hardcoded kernels mis-specialize without contiguization.
        qn, kn, v = qn.contiguous(), kn.contiguous(), v.contiguous()
        g_cum, beta, A = g_cum.contiguous(), beta.contiguous(), A.contiguous()
        dq, dk, dv, db, dg, _ = chunk_gated_delta_rule_bwd(
            q=qn,
            k=kn,
            v=v,
            g=g_cum,
            beta=beta,
            A=A,
            scale=scale,
            initial_state=None,
            do=do.contiguous(),
            dht=None,
            cu_seqlens=_cu(position_ids),
        )
        dq = l2norm_bwd(qn, q_rstd, dq)
        dk = l2norm_bwd(kn, k_rstd, dk)
        return (
            dq.to(qn.dtype),
            dk.to(kn.dtype),
            dv.to(v.dtype),
            dg.to(g_cum.dtype),
            db.to(beta.dtype),
        )

    @_gdn_chunk_bwd.register_fake
    def _(qn, q_rstd, kn, k_rstd, v, g_cum, beta, A, do, scale, position_ids):
        def emp(shape, dtype):
            return torch.empty(shape, dtype=dtype, device=qn.device)

        return (
            emp(qn.shape, qn.dtype),
            emp(kn.shape, kn.dtype),
            emp(v.shape, v.dtype),
            emp(g_cum.shape, g_cum.dtype),
            emp(beta.shape, beta.dtype),
        )

    def _gdn_chunk_setup(ctx, inputs, output):
        _q, _k, v, _g, beta, scale, position_ids, cast_g = inputs
        _, qn, q_rstd, kn, k_rstd, g_cum, A = output
        ctx.scale = scale
        ctx.cast_g = cast_g
        ctx.save_for_backward(qn, q_rstd, kn, k_rstd, v, g_cum, beta, A, position_ids)

    def _gdn_chunk_backward(ctx, do, *unused_intermediate_grads):
        qn, q_rstd, kn, k_rstd, v, g_cum, beta, A, position_ids = ctx.saved_tensors
        dq, dk, dv, dg, db = torch.ops.axolotl_gdn.gdn_chunk_bwd(
            qn, q_rstd, kn, k_rstd, v, g_cum, beta, A, do, ctx.scale, position_ids
        )
        dg = dg.to(g_cum.dtype)
        if ctx.cast_g:
            # qwen3_5: eager casts g to bf16, so its dg lands on the bf16 grid — reproduce that f32->bf16->f32 round-trip.
            dg = dg.to(qn.dtype).to(g_cum.dtype)
        return dq, dk, dv, dg, db, None, None, None

    _gdn_chunk.register_autograd(_gdn_chunk_backward, setup_context=_gdn_chunk_setup)

    _OPS_BUILT = True


def fla_ops_available() -> bool:
    """Build the ops if needed; True when the FLA-backed custom ops exist."""
    global _OPS_BUILD_ERROR
    if _OPS_BUILT:
        return True
    if _OPS_BUILD_ERROR is not None:
        return False
    try:
        _build_ops()
    except Exception as exc:  # pragma: no cover - depends on fla install
        _OPS_BUILD_ERROR = f"{type(exc).__name__}: {exc}"
        return False
    return True


def fla_ops_build_error() -> str | None:
    """The cached exception from a failed op build, or None."""
    return _OPS_BUILD_ERROR


# Shared FusedRMSNormGated compile boundary (used by every GatedDeltaNet model, qwen3_5 + qwen3_next).
# FLA's eager FusedRMSNormGated backward calls aten.as_strided in a way torch.compile can't meta-trace.
_FLA_RMSNORM_GATED_OP = None


def _build_fla_rmsnorm_gated_op():
    # Backward is its own opaque op: AOT-autograd traces the backward graph too, so the FLA Triton recompute must also be hidden or it hits FakeTensors.
    @torch.library.custom_op("axolotl_fla::rmsnorm_gated_bwd", mutates_args=())
    def _bwd_op(
        grad: torch.Tensor,
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        activation: str,
        eps: float,
    ) -> list[torch.Tensor]:
        from fla.modules.fused_norm_gate import rms_norm_gated

        with torch.enable_grad():
            xd, gd, wd = (t.detach().requires_grad_(True) for t in (x, g, weight))
            y = rms_norm_gated(xd, gd, wd, None, activation, eps=eps)
            dx, dg, dw = torch.autograd.grad(y, (xd, gd, wd), grad)
        return [dx, dg, dw]

    @_bwd_op.register_fake
    def _(grad, x, g, weight, activation, eps):
        return [torch.empty_like(x), torch.empty_like(g), torch.empty_like(weight)]

    @torch.library.custom_op("axolotl_fla::rmsnorm_gated", mutates_args=())
    def _op(
        x: torch.Tensor,
        g: torch.Tensor,
        weight: torch.Tensor,
        activation: str,
        eps: float,
    ) -> torch.Tensor:
        from fla.modules.fused_norm_gate import rms_norm_gated

        return rms_norm_gated(x, g, weight, None, activation, eps=eps).contiguous()

    @_op.register_fake
    def _(x, g, weight, activation, eps):
        return torch.empty_like(x, memory_format=torch.contiguous_format)

    def _setup(ctx, inputs, output):
        x, g, weight, activation, eps = inputs
        ctx.save_for_backward(x, g, weight)
        ctx.activation, ctx.eps = activation, eps

    def _bwd(ctx, grad):
        x, g, weight = ctx.saved_tensors
        dx, dg, dw = _bwd_op(grad.contiguous(), x, g, weight, ctx.activation, ctx.eps)
        return dx, dg, dw, None, None

    _op.register_autograd(_bwd, setup_context=_setup)
    return _op


def _fla_rmsnorm_gated_compiled_forward(
    self, x, g, residual=None, prenorm=False, residual_in_fp32=False
):
    # Class-level patch: non-GatedDeltaNet FLA models (residual/prenorm/bias/weight=None variants) must take the plain eager path.
    if (
        residual is None
        and not prenorm
        and self.bias is None
        and self.weight is not None
    ):
        return _FLA_RMSNORM_GATED_OP(x, g, self.weight, self.activation, self.eps)
    from fla.modules.fused_norm_gate import rms_norm_gated

    return rms_norm_gated(
        x,
        g,
        self.weight,
        self.bias,
        self.activation,
        residual=residual,
        eps=self.eps,
        prenorm=prenorm,
        residual_in_fp32=residual_in_fp32,
    )


def install_rmsnorm_gated_compile_boundary(
    fused_rms_norm_gated_cls, logger=None
) -> None:
    """Wrap FusedRMSNormGated.forward in an opaque op once (class-level, never reverted; assumes one-model-per-process)."""
    if getattr(fused_rms_norm_gated_cls, "_axolotl_compile_boundary", False):
        return
    global _FLA_RMSNORM_GATED_OP
    try:
        if _FLA_RMSNORM_GATED_OP is None:
            _FLA_RMSNORM_GATED_OP = _build_fla_rmsnorm_gated_op()
        fused_rms_norm_gated_cls.forward = _fla_rmsnorm_gated_compiled_forward
        fused_rms_norm_gated_cls._axolotl_compile_boundary = True
    except Exception:  # pragma: no cover
        try:
            import torch._dynamo as _dyn

            fused_rms_norm_gated_cls.forward = _dyn.disable(
                fused_rms_norm_gated_cls.forward
            )
            fused_rms_norm_gated_cls._axolotl_compile_boundary = True
        except Exception as exc:
            if logger is not None:
                logger.warning(
                    f"Could not install a compile boundary for FusedRMSNormGated "
                    f"({exc}); torch.compile may graph-break in the decoder loop"
                )
