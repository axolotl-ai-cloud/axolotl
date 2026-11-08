# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Clamped SwiGLU ``silu(min(gate, L)) * clamp(up, -L, L)`` as a quack gated epilogue.

The forward and backward wrap quack's own ``swiglu`` / ``dswiglu`` (scalar and packed F32x2
forms) around the clamp. Gradients follow ``torch.clamp``: they pass strictly inside the bound
and are zero at and beyond it.
"""

from __future__ import annotations


def build_clamped_swiglu(activation, limit: float):
    """``(swiglu, dswiglu)`` clamped at ``limit``, built on quack ``activation``'s SwiGLU."""
    import cutlass
    from cutlass import Float32, const_expr
    from cutlass.cutlass_dsl import dsl_user_op

    hi = float(limit)

    def clamp_gate(x):
        return cutlass.min(x, Float32(hi))

    def clamp_up(y):
        return cutlass.max(cutlass.min(y, Float32(hi)), Float32(-hi))

    def gate_grad(dx, x):
        return cutlass.select_(x < Float32(hi), dx, Float32(0.0))

    def up_grad(dy, y):
        dy = cutlass.select_(y < Float32(hi), dy, Float32(0.0))
        return cutlass.select_(y > Float32(-hi), dy, Float32(0.0))

    @dsl_user_op
    def swiglu_clamped(x, y, *, loc=None, ip=None):
        if const_expr(not isinstance(x, tuple)):
            return activation.swiglu(clamp_gate(x), clamp_up(y))
        return activation.swiglu(
            (clamp_gate(x[0]), clamp_gate(x[1])), (clamp_up(y[0]), clamp_up(y[1]))
        )

    @dsl_user_op
    def dswiglu_clamped(x, y, dout, *, loc=None, ip=None):
        if const_expr(not isinstance(x, tuple)):
            dx, dy, out = activation.dswiglu(clamp_gate(x), clamp_up(y), dout)
            return gate_grad(dx, x), up_grad(dy, y), out
        dx, dy, out = activation.dswiglu(
            (clamp_gate(x[0]), clamp_gate(x[1])),
            (clamp_up(y[0]), clamp_up(y[1])),
            dout,
        )
        return (
            (gate_grad(dx[0], x[0]), gate_grad(dx[1], x[1])),
            (up_grad(dy[0], y[0]), up_grad(dy[1], y[1])),
            out,
        )

    return swiglu_clamped, dswiglu_clamped
