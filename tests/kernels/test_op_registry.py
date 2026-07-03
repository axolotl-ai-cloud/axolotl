"""Tests for custom-op registration of axolotl kernels."""

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA"
)


class _OpRecorder(TorchDispatchMode):
    def __init__(self):
        self.seen = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(str(func))
        return func(*args, **(kwargs or {}))


class TestOpsRegistered:
    def test_flash_d512_ops_exist(self):
        import axolotl.monkeypatch.attention.flash_attn_d512  # noqa: F401

        assert torch.ops.axolotl.flash_attn_d512_fwd is not None
        assert torch.ops.axolotl.flash_attn_d512_bwd is not None

    def test_activation_ops_exist(self):
        import axolotl.kernels.geglu  # noqa: F401
        import axolotl.kernels.swiglu  # noqa: F401

        assert torch.ops.axolotl.swiglu_fwd is not None
        assert torch.ops.axolotl.swiglu_bwd is not None
        assert torch.ops.axolotl.geglu_fwd is not None
        assert torch.ops.axolotl.geglu_bwd is not None

    def test_rms_norm_gated_ops_exist(self):
        import axolotl.kernels.rms_norm_gated  # noqa: F401

        assert torch.ops.axolotl.rms_norm_gated_fwd is not None
        assert torch.ops.axolotl.rms_norm_gated_bwd is not None

    def test_trainer_utils_ops_exist(self):
        import axolotl.monkeypatch.trainer.utils  # noqa: F401

        assert torch.ops.axolotl.selective_log_softmax_fwd is not None
        assert torch.ops.axolotl.selective_log_softmax_bwd is not None
        assert torch.ops.axolotl.entropy_from_logits is not None

    def test_ebft_ops_exist(self):
        import axolotl.core.trainers.ebft.kernels  # noqa: F401

        assert torch.ops.axolotl.ebft_fused_log_softmax_gather is not None
        assert torch.ops.axolotl.ebft_fused_reinforce_loss is not None
        assert torch.ops.axolotl.ebft_fused_cosine_similarity is not None
        assert torch.ops.axolotl.ebft_fused_diversity_penalty is not None


class TestDispatchVisibility:
    @requires_cuda
    def test_swiglu_visible_to_dispatch_mode(self):
        from axolotl.kernels.swiglu import swiglu_forward

        gate = torch.randn(1, 8, 64, device="cuda")
        up = torch.randn(1, 8, 64, device="cuda")
        with _OpRecorder() as rec:
            swiglu_forward(gate, up)
        assert any("axolotl.swiglu_fwd" in s for s in rec.seen)

    @requires_cuda
    def test_flash_d512_visible_to_dispatch_mode(self):
        from axolotl.monkeypatch.attention.flash_attn_d512 import flash_d512

        q, k, v = (
            torch.randn(1, 2, 64, 512, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )
        with _OpRecorder() as rec:
            flash_d512(q, k, v, causal=True)
        assert any("axolotl.flash_attn_d512_fwd" in s for s in rec.seen)


class TestFakeTensor:
    @requires_cuda
    def test_flash_d512_fwd_fake_shapes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        import axolotl.monkeypatch.attention.flash_attn_d512  # noqa: F401

        with FakeTensorMode():
            q = torch.empty(2, 4, 128, 512, device="cuda", dtype=torch.bfloat16)
            k = torch.empty_like(q)
            v = torch.empty_like(q)
            pos = torch.zeros(2, 128, device="cuda", dtype=torch.int32)
            o, lse = torch.ops.axolotl.flash_attn_d512_fwd(
                q, k, v, pos, True, False, 0.044
            )
            assert o.shape == q.shape and o.dtype == q.dtype
            assert lse.shape == (8, 128) and lse.dtype == torch.float32

    @requires_cuda
    def test_swiglu_fwd_fake_shapes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        import axolotl.kernels.swiglu  # noqa: F401

        with FakeTensorMode():
            gate = torch.empty(1, 8, 64, device="cuda", dtype=torch.bfloat16)
            up = torch.empty_like(gate)
            out = torch.ops.axolotl.swiglu_fwd(gate, up)
            assert out.shape == gate.shape and out.dtype == gate.dtype


class TestNewOpsDispatchVisibility:
    @requires_cuda
    def test_rms_norm_gated_visible_to_dispatch_mode(self):
        from axolotl.kernels.rms_norm_gated import FusedRMSNormGated

        mod = FusedRMSNormGated(64).to("cuda", torch.bfloat16)
        x = torch.randn(2, 8, 64, device="cuda", dtype=torch.bfloat16)
        g = torch.randn_like(x)
        with _OpRecorder() as rec:
            mod(x, gate=g)
        assert any("axolotl.rms_norm_gated_fwd" in s for s in rec.seen)

    @requires_cuda
    def test_selective_log_softmax_visible_to_dispatch_mode(self):
        from axolotl.monkeypatch.trainer.utils import selective_log_softmax

        logits = torch.randn(2, 16, 512, device="cuda", dtype=torch.bfloat16)
        index = torch.randint(0, 512, (2, 16), device="cuda")
        with _OpRecorder() as rec:
            selective_log_softmax(logits, index)
        assert any("axolotl.selective_log_softmax_fwd" in s for s in rec.seen)

    @requires_cuda
    def test_entropy_visible_to_dispatch_mode(self):
        from axolotl.monkeypatch.trainer.utils import entropy_from_logits

        logits = torch.randn(2, 16, 512, device="cuda", dtype=torch.bfloat16)
        with _OpRecorder() as rec:
            entropy_from_logits(logits)
        assert any("axolotl.entropy_from_logits" in s for s in rec.seen)

    @requires_cuda
    def test_ebft_ops_visible_to_dispatch_mode(self):
        from axolotl.core.trainers.ebft.kernels import (
            fused_cosine_similarity,
            fused_diversity_penalty,
            fused_log_softmax_gather,
            fused_reinforce_loss,
        )

        logits = torch.randn(2, 8, 512, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(0, 512, (2, 8), device="cuda")
        logps = torch.randn(64, device="cuda")
        advs = torch.randn(64, device="cuda")
        mask = torch.rand(64, device="cuda") > 0.5
        a = torch.randn(4, 128, device="cuda")
        emb = torch.randn(2, 4, 64, device="cuda")
        with _OpRecorder() as rec:
            fused_log_softmax_gather(logits, labels)
            fused_reinforce_loss(logps, advs, mask)
            fused_cosine_similarity(a, a.clone())
            fused_diversity_penalty(emb)
        for op in (
            "ebft_fused_log_softmax_gather",
            "ebft_fused_reinforce_loss",
            "ebft_fused_cosine_similarity",
            "ebft_fused_diversity_penalty",
        ):
            assert any(f"axolotl.{op}" in s for s in rec.seen), op


class TestNewOpsFakeTensor:
    @requires_cuda
    def test_rms_norm_gated_fwd_fake_shapes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        import axolotl.kernels.rms_norm_gated  # noqa: F401

        with FakeTensorMode():
            x = torch.empty(16, 64, device="cuda", dtype=torch.bfloat16)
            g = torch.empty_like(x)
            w = torch.empty(64, device="cuda", dtype=torch.bfloat16)
            y, rstd = torch.ops.axolotl.rms_norm_gated_fwd(x, g, w, 1e-6, 0.0, 64, 4)
            assert y.shape == x.shape and y.dtype == x.dtype
            assert rstd.shape == (16,) and rstd.dtype == torch.float32

    @requires_cuda
    def test_selective_log_softmax_fwd_fake_shapes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        import axolotl.monkeypatch.trainer.utils  # noqa: F401

        with FakeTensorMode():
            logits = torch.empty(32, 512, device="cuda", dtype=torch.bfloat16)
            index = torch.empty(32, 1, device="cuda", dtype=torch.int64)
            out, lse = torch.ops.axolotl.selective_log_softmax_fwd(
                logits, index, 1, 1, 512, 4096, 8192
            )
            assert out.shape == (32, 1) and out.dtype == torch.float32
            assert lse.shape == (32,) and lse.dtype == torch.float32

    @requires_cuda
    def test_entropy_fake_shapes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        import axolotl.monkeypatch.trainer.utils  # noqa: F401

        with FakeTensorMode():
            logits = torch.empty(2, 16, 512, device="cuda", dtype=torch.bfloat16)
            out = torch.ops.axolotl.entropy_from_logits(logits, 128)
            assert out.shape == (32,) and out.dtype == torch.float32

    @requires_cuda
    def test_ebft_fake_shapes(self):
        from torch._subclasses.fake_tensor import FakeTensorMode

        import axolotl.core.trainers.ebft.kernels  # noqa: F401

        with FakeTensorMode():
            logits = torch.empty(2, 8, 512, device="cuda", dtype=torch.bfloat16)
            labels = torch.empty(2, 8, device="cuda", dtype=torch.int64)
            out = torch.ops.axolotl.ebft_fused_log_softmax_gather(logits, labels)
            assert out.shape == (2, 8) and out.dtype == torch.float32

            logps = torch.empty(64, device="cuda")
            mask = torch.empty(64, device="cuda", dtype=torch.bool)
            loss = torch.ops.axolotl.ebft_fused_reinforce_loss(logps, logps, mask)
            assert loss.shape == () and loss.dtype == torch.float32

            a = torch.empty(4, 8, 128, device="cuda")
            sim = torch.ops.axolotl.ebft_fused_cosine_similarity(a, a)
            assert sim.shape == (4, 8) and sim.dtype == torch.float32

            emb = torch.empty(2, 4, 64, device="cuda")
            div = torch.ops.axolotl.ebft_fused_diversity_penalty(emb)
            assert div.shape == (2, 4) and div.dtype == torch.float32


class TestCompile:
    @requires_cuda
    def test_swiglu_compiles_fullgraph(self):
        from axolotl.kernels.swiglu import swiglu_forward

        def fn(gate, up):
            return swiglu_forward(gate, up).sum()

        gate = torch.randn(1, 8, 64, device="cuda")
        up = torch.randn(1, 8, 64, device="cuda")
        eager = fn(gate, up)
        compiled = torch.compile(fn, fullgraph=True)(gate, up)
        torch.testing.assert_close(eager, compiled)

    @requires_cuda
    def test_flash_d512_compiles_fullgraph(self):
        from axolotl.monkeypatch.attention.flash_attn_d512 import flash_d512

        def fn(q, k, v):
            return flash_d512(q, k, v, causal=True).sum()

        q, k, v = (
            torch.randn(1, 2, 64, 512, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )
        eager = fn(q, k, v)
        compiled = torch.compile(fn, fullgraph=True)(q, k, v)
        torch.testing.assert_close(eager, compiled, atol=2e-2, rtol=2e-2)


class TestOpcheck:
    """torch.library.opcheck validates schema, fake fidelity, aliasing, and
    functionalization in one shot."""

    @requires_cuda
    def test_swiglu_ops(self):
        from axolotl.kernels.swiglu import _swiglu_bwd_op, swiglu_forward

        gate = torch.randn(1, 8, 64, device="cuda")
        up = torch.randn(1, 8, 64, device="cuda")
        go = torch.randn(1, 8, 64, device="cuda")
        torch.library.opcheck(swiglu_forward, (gate.clone(), up.clone()))
        torch.library.opcheck(_swiglu_bwd_op, (go, gate, up))

    @requires_cuda
    def test_geglu_ops(self):
        from axolotl.kernels.geglu import _geglu_bwd_op, geglu_forward

        gate = torch.randn(1, 8, 64, device="cuda")
        up = torch.randn(1, 8, 64, device="cuda")
        go = torch.randn(1, 8, 64, device="cuda")
        torch.library.opcheck(geglu_forward, (gate.clone(), up.clone()))
        torch.library.opcheck(_geglu_bwd_op, (go, gate, up))

    @requires_cuda
    def test_flash_d512_fwd(self):
        from axolotl.monkeypatch.attention.flash_attn_d512 import _flash_d512_fwd_op

        q, k, v = (
            torch.randn(1, 1, 32, 512, device="cuda", dtype=torch.bfloat16)
            for _ in range(3)
        )
        pos = torch.zeros(1, 32, device="cuda", dtype=torch.int32)
        torch.library.opcheck(_flash_d512_fwd_op, (q, k, v, pos, True, False, 0.044))


class TestMutatingBwdUnderCompile:
    @requires_cuda
    def test_swiglu_backward_functionalizes_correctly(self):
        """The mutate-then-read-back wrapper must survive compile's
        auto-functionalization (downstream uses of mutated args are remapped)."""
        from axolotl.kernels.swiglu import swiglu_backward

        def fn(go, gate, up):
            h, dg, du = swiglu_backward(go, gate, up)
            return h.sum() + dg.sum() * 2 + du.sum() * 3

        go, gate, up = (torch.randn(1, 16, 128, device="cuda") for _ in range(3))
        eager = fn(go.clone(), gate.clone(), up.clone())
        compiled = torch.compile(fn, fullgraph=True)(
            go.clone(), gate.clone(), up.clone()
        )
        torch.testing.assert_close(eager, compiled)


class TestFallbackShim:
    def test_unregistered_op_passthrough(self):
        from axolotl.kernels.op_registry import _UnregisteredOp

        def raw(x):
            return x + 1

        shim = _UnregisteredOp(raw)
        assert shim(torch.tensor(1)).item() == 2
        # register_fake is a no-op that must not raise
        shim.register_fake(lambda x: x)

    def test_failed_registration_falls_back(self):
        from typing import Callable

        from axolotl.kernels.op_registry import _UnregisteredOp, register_kernel_op

        # a callable arg is not an op-schema type: registration fails and the
        # decorator degrades to the raw callable
        @register_kernel_op("test_bad_schema_op")
        def _bad(x: torch.Tensor, fn: Callable) -> torch.Tensor:
            return fn(x)

        assert isinstance(_bad, _UnregisteredOp)
        assert _bad(torch.tensor(1), lambda t: t + 2).item() == 3
        _bad.register_fake(lambda x, fn: x)
