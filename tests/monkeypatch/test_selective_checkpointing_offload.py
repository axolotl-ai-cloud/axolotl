"""Tests for phase-2 SAC CPU offload."""

import pytest
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from axolotl.monkeypatch.selective_checkpointing_offload import (
    SacOffloadEngine,
    build_sac_offload_context_fn,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


class TestSacOffloadFunctional:
    @requires_cuda
    def test_grads_match_baseline_with_offload(self):
        device = "cuda"
        batch, heads, seq, dim = 2, 4, 512, 64

        def make_inputs():
            gen = torch.Generator(device="cpu").manual_seed(7)
            qkv = torch.randn(
                3, batch, heads, seq, dim, dtype=torch.float32, generator=gen
            )
            return [t.to(device).detach().clone().requires_grad_(True) for t in qkv]

        def attn_block(q, k, v):
            out = F.scaled_dot_product_attention(q, k, v)
            return out.relu() @ v.transpose(-2, -1)

        q0, k0, v0 = make_inputs()
        attn_block(q0, k0, v0).sum().backward()

        engine = SacOffloadEngine(min_offload_bytes=1024)
        context_fn = build_sac_offload_context_fn(["attention"], engine=engine)

        q1, k1, v1 = make_inputs()
        out = checkpoint(
            attn_block, q1, k1, v1, use_reentrant=False, context_fn=context_fn
        )
        out.sum().backward()
        torch.cuda.synchronize()

        assert engine.stats.offloaded_tensors > 0, "nothing was offloaded"
        assert engine.stats.restored_tensors == engine.stats.offloaded_tensors
        torch.testing.assert_close(q0.grad, q1.grad)
        torch.testing.assert_close(k0.grad, k1.grad)
        torch.testing.assert_close(v0.grad, v1.grad)

    @requires_cuda
    def test_multi_region_prefetch_and_reuse(self):
        device = "cuda"
        n_layers, seq, hidden = 4, 256, 128

        gen = torch.Generator(device="cpu").manual_seed(11)
        weights = [
            torch.randn(hidden, hidden, generator=gen).to(device).requires_grad_(True)
            for _ in range(n_layers)
        ]

        def layer(x, w):
            q = (x @ w).view(1, 2, seq, hidden // 2)
            out = F.scaled_dot_product_attention(q, q, q)
            return out.reshape(1, seq, hidden).relu()

        def run(context_fn=None):
            gen2 = torch.Generator(device="cpu").manual_seed(12)
            x = torch.randn(1, seq, hidden, generator=gen2).to(device)
            for w in weights:
                if w.grad is not None:
                    w.grad = None
            h = x
            for w in weights:
                if context_fn is not None:
                    h = checkpoint(
                        layer, h, w, use_reentrant=False, context_fn=context_fn
                    )
                else:
                    h = layer(h, w)
            h.sum().backward()
            torch.cuda.synchronize()
            return [w.grad.clone() for w in weights]

        baseline = run()

        engine = SacOffloadEngine(min_offload_bytes=1024)
        context_fn = build_sac_offload_context_fn(["attention"], engine=engine)
        # two steps: second step exercises the pinned-buffer pool reuse
        run(context_fn)
        grads = run(context_fn)

        assert engine.stats.offloaded_tensors >= 2 * n_layers
        assert engine.stats.restored_tensors == engine.stats.offloaded_tensors
        for g0, g1 in zip(baseline, grads, strict=True):
            torch.testing.assert_close(g0, g1)
