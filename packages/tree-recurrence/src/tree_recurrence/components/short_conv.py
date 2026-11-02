# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Axolotl AI

"""The gated short-conv component (LFM2, LFM2-MoE).

LFM2's recurrent layers are a gated short causal conv with no scan, so a tree row
only needs each segment's conv to read the last inputs of its path: every shared
token runs once.
"""

from __future__ import annotations

from collections.abc import Callable

import torch

from tree_recurrence.components.conv import channels_first_conv
from tree_recurrence.driver import tree_conv
from tree_recurrence.plan import TreePlan


def gated_short_conv(
    mixer: torch.nn.Module,
    causal_conv1d_fn: Callable[..., torch.Tensor],
    hidden_states: torch.Tensor,
    plan: TreePlan,
) -> torch.Tensor:
    """``out_proj(C * conv(B * x))`` with ``B, C, x = in_proj(hidden_states)``, the
    mixer of transformers' ``Lfm2ShortConv``, on a tree row."""
    b, c, x = mixer.in_proj(hidden_states).chunk(3, dim=-1)
    weight = mixer.conv.weight.squeeze(1)
    conv = channels_first_conv(causal_conv1d_fn, weight, mixer.conv.bias)
    return mixer.out_proj(c * tree_conv(conv, b * x, plan, weight.shape[-1]))
