# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Non-gated (relu²) expert support for the MoE kernels — nemotron_h latent experts.

CPU-tier coverage: layout acceptance, activation detection/resolution, the
non-gated activation itself, and the sonicmoe grouped fallback math (float64,
per-expert loop path) against a per-token oracle.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from axolotl.integrations.kernels.libs.sonicmoe.nongated import (
    _NonGatedGroupedMLP,
    sonicmoe_nongated_forward,
)
from axolotl.integrations.kernels.libs.sonicmoe.nvfp4 import (
    gated_activation,
    resolve_gated_activation,
)


class TestLayoutAcceptance:
    def _mod(self, **kw):
        base = {
            "is_transposed": False,
            "is_concatenated": True,
            "has_bias": False,
            "has_gate": False,
            "up_proj": torch.zeros(2, 4, 8),
            "down_proj": torch.zeros(2, 8, 4),
        }
        base.update(kw)
        return SimpleNamespace(**base)

    def test_nongated_up_down_accepted(self):
        from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
            scattermoe_supports_layout,
        )

        assert scattermoe_supports_layout(self._mod())

    def test_nongated_transposed_rejected(self):
        from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
            scattermoe_supports_layout,
        )

        assert not scattermoe_supports_layout(self._mod(is_transposed=True))
        assert not scattermoe_supports_layout(self._mod(has_bias=True))

    def test_gated_layout_still_accepted(self):
        from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
            scattermoe_supports_layout,
        )

        mod = self._mod(has_gate=True)
        del mod.up_proj
        mod.gate_up_proj = torch.zeros(2, 8, 8)
        assert scattermoe_supports_layout(mod)


class TestActResolution:
    def test_detect_relu2(self):
        from transformers.activations import ACT2FN

        from axolotl.integrations.kernels.libs.scattermoe_lora.experts import (
            _detect_act_type,
        )

        mod = SimpleNamespace(act_fn=ACT2FN["relu2"])
        assert _detect_act_type(mod) == "relu2"

    def test_resolve_mlp_hidden_act_fallback(self):
        # nemotron_h has no hidden_act; mlp_hidden_act must be honored
        cfg = SimpleNamespace(mlp_hidden_act="relu2")
        assert resolve_gated_activation(cfg) == "relu2"

    def test_resolve_hidden_act_precedence(self):
        cfg = SimpleNamespace(hidden_act="silu", mlp_hidden_act="relu2")
        assert resolve_gated_activation(cfg) == "silu"


class TestNonGatedActivation:
    def test_relu2_matches_eager(self):
        h = torch.randn(64, 32, dtype=torch.float64)
        out = gated_activation(h, "relu2", concat=True, gated=False)
        assert torch.allclose(out, F.relu(h).square())

    def test_unsupported_act_raises(self):
        with pytest.raises(ValueError, match="non-gated"):
            gated_activation(torch.randn(4, 4), "swiglu2", concat=True, gated=False)

    def test_gated_path_unchanged(self):
        h = torch.randn(16, 8, dtype=torch.float64)
        out = gated_activation(h, "silu", concat=True)
        gate, up = h.chunk(2, dim=-1)
        assert torch.allclose(out, F.silu(gate) * up)


def _oracle(x, idx, wts, w1, w2):
    """Per-token non-gated relu² MoE oracle."""
    T, K = idx.shape
    out = torch.zeros(T, w2.shape[1], dtype=x.dtype)
    for t in range(T):
        for k in range(K):
            e = int(idx[t, k])
            h = x[t] @ w1[e].T
            a = F.relu(h).square()
            out[t] = out[t] + wts[t, k] * (a @ w2[e].T)
    return out


class TestNonGatedGroupedMLP:
    def test_forward_matches_oracle(self):
        torch.manual_seed(0)
        E, L, I, T, K = 4, 8, 12, 16, 2
        x = torch.randn(T, L, dtype=torch.float64)
        w1 = torch.randn(E, I, L, dtype=torch.float64) * 0.3
        w2 = torch.randn(E, L, I, dtype=torch.float64) * 0.3
        idx = torch.randint(0, E, (T, K))
        wts = torch.rand(T, K, dtype=torch.float64)

        out = sonicmoe_nongated_forward(x, idx, wts, w1, w2, E)
        assert torch.allclose(out, _oracle(x, idx, wts, w1, w2), atol=1e-10)

    def test_gradients_match_autograd_reference(self):
        torch.manual_seed(1)
        E, L, I, T = 3, 6, 10, 12
        offs = torch.tensor([5, 9, 12], dtype=torch.int32)
        x = torch.randn(T, L, dtype=torch.float64, requires_grad=True)
        w1 = (torch.randn(E, I, L, dtype=torch.float64) * 0.3).requires_grad_()
        w2 = (torch.randn(E, L, I, dtype=torch.float64) * 0.3).requires_grad_()

        y = _NonGatedGroupedMLP.apply(x, w1, w2, offs)
        g = torch.randn_like(y)
        y.backward(g)
        got = (x.grad.clone(), w1.grad.clone(), w2.grad.clone())

        for t in (x, w1, w2):
            t.grad = None
        start, outs = 0, []
        for e, end in enumerate(offs.tolist()):
            outs.append(F.relu(x[start:end] @ w1[e].T).square() @ w2[e].T)
            start = end
        torch.cat(outs).backward(g)
        for a, b in zip(got, (x.grad, w1.grad, w2.grad), strict=True):
            assert torch.allclose(a, b, atol=1e-10)

    def test_empty_expert_segments(self):
        # experts 0 and 2 receive no tokens
        E, L, I, T = 3, 4, 6, 5
        offs = torch.tensor([0, 5, 5], dtype=torch.int32)
        x = torch.randn(T, L, dtype=torch.float64)
        w1 = torch.randn(E, I, L, dtype=torch.float64)
        w2 = torch.randn(E, L, I, dtype=torch.float64)
        y = _NonGatedGroupedMLP.apply(x, w1, w2, offs)
        assert torch.allclose(y, F.relu(x @ w1[1].T).square() @ w2[1].T)
