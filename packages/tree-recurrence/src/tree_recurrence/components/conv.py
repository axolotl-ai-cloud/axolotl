# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""Short causal convs as ``ConvFn``s for ``tree_conv``.

``tree_conv`` puts each segment's path history in front of it, so the conv needs
no boundaries and any kernel will do.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch

from tree_recurrence.driver import ConvFn


def channels_first_conv(
    causal_conv1d_fn: Callable[..., torch.Tensor],
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> ConvFn:
    """A ``causal_conv1d_fn(x [B, D, T], weight [D, W], bias, activation=...)``, as
    in the causal-conv1d package and transformers' modeling modules."""

    def conv(x, cu_seqlens):
        kwargs = {} if activation is None else {"activation": activation}
        out = causal_conv1d_fn(x.transpose(1, 2), weight, bias, **kwargs)
        return out.transpose(1, 2)

    return conv


def fla_conv(
    causal_conv1d: Callable[..., Any],
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = None,
) -> ConvFn:
    """FLA's varlen ``causal_conv1d(x [B, T, D], weight, bias, activation,
    cu_seqlens)``, which returns the output and a cache."""

    def conv(x, cu_seqlens):
        out, _ = causal_conv1d(
            x=x, weight=weight, bias=bias, activation=activation, cu_seqlens=cu_seqlens
        )
        return out

    return conv
