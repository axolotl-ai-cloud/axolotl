"""CPU checks for routing FA4 (flash_attn.cute) through axolotl's custom ops under SAC."""

import logging
import sys
import types

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.checkpoint import checkpoint

import axolotl.monkeypatch.attention.flash_attn_4 as fa4
import axolotl.monkeypatch.attention.flash_attn_4_ops as fa4_ops
from axolotl.monkeypatch.selective_checkpointing import (
    SacPolicyState,
    _is_flash_attention_forward,
    _window_arg_index,
    build_sac_context_fn,
)
from axolotl.monkeypatch.selective_checkpointing_offload import (
    build_sac_offload_context_fn,
)

pytestmark = pytest.mark.skipif(
    not fa4_ops._REGISTERED, reason="torch.library custom ops unavailable"
)

# Mirrors flash_attn/cute/interface.py: the public entry points resolve the autograd
# Functions from the module namespace at call time, and backward reads only ctx.
FAKE_INTERFACE = """
import torch

CALLS = {"fwd": 0, "bwd": 0}


def _mask(sq, sk, causal, window):
    i = torch.arange(sq)[:, None] + (sk - sq)
    j = torch.arange(sk)[None, :]
    keep = torch.ones(sq, sk, dtype=torch.bool)
    left, right = window
    if causal:
        right = 0
    if left is not None and left >= 0:
        keep &= j >= i - left
    if right is not None and right >= 0:
        keep &= j <= i + right
    return keep


def _scores(q, k, scale, causal, window, softcap):
    k = k.repeat_interleave(q.shape[1] // k.shape[1], dim=1)
    s = torch.einsum("qhd,khd->hqk", q, k) * scale
    if softcap:
        s = softcap * torch.tanh(s / softcap)
    return s.masked_fill(~_mask(q.shape[0], k.shape[0], causal, window), float("-inf"))


def _fwd(q, k, v, scale, causal, window, softcap):
    s = _scores(q, k, scale, causal, window, softcap)
    lse = torch.logsumexp(s, dim=-1)
    v = v.repeat_interleave(q.shape[1] // v.shape[1], dim=1)
    out = torch.einsum("hqk,khd->qhd", torch.exp(s - lse[..., None]), v)
    return out, lse


def _bwd(q, k, v, out, dout, lse, scale, causal, window):
    group = q.shape[1] // k.shape[1]
    p = torch.exp(_scores(q, k, scale, causal, window, 0.0) - lse[..., None])
    ke, ve = (t.repeat_interleave(group, dim=1) for t in (k, v))
    dv = torch.einsum("hqk,qhd->khd", p, dout)
    dp = torch.einsum("qhd,khd->hqk", dout, ve)
    delta = (dout * out).sum(-1).transpose(0, 1)
    ds = p * (dp - delta[..., None])
    dq = torch.einsum("hqk,khd->qhd", ds, ke) * scale
    dk = torch.einsum("hqk,qhd->khd", ds, q) * scale
    fold = lambda t: t.view(t.shape[0], k.shape[1], group, t.shape[-1]).sum(2)
    return dq, fold(dk), fold(dv)


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, softmax_scale=None, causal=False,
                window_size=(None, None), learnable_sink=None, softcap=0.0,
                pack_gqa=None):
        CALLS["fwd"] += 1
        scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        outs, lses = zip(*(
            _fwd(q[b], k[b], v[b], scale, causal, window_size, softcap)
            for b in range(q.shape[0])
        ))
        out, lse = torch.stack(outs), torch.stack(lses)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.softmax_scale = scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        CALLS["bwd"] += 1
        q, k, v, out, lse = ctx.saved_tensors
        grads = [
            _bwd(q[b], k[b], v[b], out[b], dout[b], lse[b], ctx.softmax_scale,
                 ctx.causal, ctx.window_size)
            for b in range(q.shape[0])
        ]
        dq, dk, dv = (torch.stack(g) for g in zip(*grads))
        return dq, dk, dv, *((None,) * 6)


class FlashAttnVarlenFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q=None,
                seqused_k=None, page_table=None, softmax_scale=None, causal=False,
                window_size=(None, None), learnable_sink=None, softcap=0.0,
                pack_gqa=None):
        CALLS["fwd"] += 1
        scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
        bounds = cu_seqlens_q.tolist()
        outs, lses = zip(*(
            _fwd(q[a:b], k[a:b], v[a:b], scale, causal, window_size, softcap)
            for a, b in zip(bounds[:-1], bounds[1:])
        ))
        out, lse = torch.cat(outs), torch.cat(lses, dim=1)
        ctx.save_for_backward(q, k, v, out, lse, cu_seqlens_q, cu_seqlens_k,
                              seqused_q, seqused_k)
        ctx.softmax_scale = scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.softcap = softcap
        return out, lse

    @staticmethod
    def backward(ctx, dout, *args):
        CALLS["bwd"] += 1
        q, k, v, out, lse, cu_q, cu_k, used_q, used_k = ctx.saved_tensors
        assert used_q is None and used_k is None
        bounds = cu_q.tolist()
        grads = [
            _bwd(q[a:b], k[a:b], v[a:b], out[a:b], dout[a:b], lse[:, a:b],
                 ctx.softmax_scale, ctx.causal, ctx.window_size)
            for a, b in zip(bounds[:-1], bounds[1:])
        ]
        dq, dk, dv = (torch.cat(g) for g in zip(*grads))
        return dq, dk, dv, *((None,) * 11)


def flash_attn_func(q, k, v, softmax_scale=None, causal=False,
                    window_size=(None, None), learnable_sink=None, softcap=0.0,
                    pack_gqa=None):
    return FlashAttnFunc.apply(q, k, v, softmax_scale, causal, window_size,
                               learnable_sink, softcap, pack_gqa)


def flash_attn_varlen_func(q, k, v, cu_seqlens_q=None, cu_seqlens_k=None,
                           seqused_q=None, seqused_k=None, page_table=None,
                           softmax_scale=None, causal=False, window_size=(None, None),
                           learnable_sink=None, softcap=0.0, pack_gqa=None):
    return FlashAttnVarlenFunc.apply(q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_q,
                                     seqused_k, page_table, softmax_scale, causal,
                                     window_size, learnable_sink, softcap, pack_gqa)
"""

HEADS, KV_HEADS, HEAD_DIM, HIDDEN = 4, 2, 8, 16


@pytest.fixture(name="cute")
def fixture_cute(monkeypatch):
    interface = types.ModuleType("flash_attn.cute.interface")
    exec(FAKE_INTERFACE, interface.__dict__)  # nosec B102
    cute = types.ModuleType("flash_attn.cute")
    cute.interface = interface
    cute.flash_attn_func = interface.flash_attn_func
    cute.flash_attn_varlen_func = interface.flash_attn_varlen_func
    parent = types.ModuleType("flash_attn")
    parent.cute = cute
    monkeypatch.setitem(sys.modules, "flash_attn", parent)
    monkeypatch.setitem(sys.modules, "flash_attn.cute", cute)
    monkeypatch.setitem(sys.modules, "flash_attn.cute.interface", interface)
    monkeypatch.setattr(fa4_ops, "_ORIGINALS", {})
    monkeypatch.setattr(fa4_ops, "_NEEDS_INPUT_GRAD", {})
    fa4_ops._PENDING.clear()
    assert fa4_ops.install_fa4_ops(cute)
    yield cute
    fa4_ops._PENDING.clear()


@pytest.fixture(name="op_calls")
def fixture_op_calls(monkeypatch):
    calls = {"fwd": 0}
    impl = fa4_ops._forward_impl

    def counting(*args, **kwargs):
        calls["fwd"] += 1
        return impl(*args, **kwargs)

    monkeypatch.setattr(fa4_ops, "_forward_impl", counting)
    return calls


def _weights(seed=0):
    gen = torch.Generator().manual_seed(seed)
    shapes = [
        (2, 6, HIDDEN),
        (HIDDEN, HEADS * HEAD_DIM),
        (HIDDEN, KV_HEADS * HEAD_DIM),
        (HIDDEN, KV_HEADS * HEAD_DIM),
        (HEADS * HEAD_DIM, HIDDEN),
    ]
    return [torch.randn(s, generator=gen).requires_grad_(True) for s in shapes]


def _project(x, wq, wk, wv):
    b, s, _ = x.shape
    return (
        (x @ wq).view(b, s, HEADS, HEAD_DIM),
        (x @ wk).view(b, s, KV_HEADS, HEAD_DIM),
        (x @ wv).view(b, s, KV_HEADS, HEAD_DIM),
    )


def _dense_block(cute, state=None, layer_type=None, **attn_kwargs):
    def block(x, wq, wk, wv, wo):
        if state is not None:
            state.current_layer_type = layer_type
        q, k, v = _project(x, wq, wk, wv)
        out = cute.flash_attn_func(q, k, v, causal=True, **attn_kwargs)[0]
        if state is not None:
            state.current_layer_type = None
        return (out.reshape(x.shape[0], x.shape[1], -1) @ wo).sum()

    return block


def _varlen_block(cute, **attn_kwargs):
    def block(x, wq, wk, wv, wo):
        q, k, v = (t.flatten(0, 1) for t in _project(x, wq, wk, wv))
        cu = torch.tensor([0, 5, 12], dtype=torch.int32)
        out = cute.flash_attn_varlen_func(
            q, k, v, cu_seqlens_q=cu, cu_seqlens_k=cu, causal=True, **attn_kwargs
        )[0]
        return (out.reshape(x.shape[0], x.shape[1], -1) @ wo).sum()

    return block


def _run(cute, block, context_fn=None, checkpointed=True):
    calls = cute.interface.CALLS
    calls.update(fwd=0, bwd=0)
    inputs = _weights()
    if not checkpointed:
        loss = block(*inputs)
    elif context_fn is None:
        loss = checkpoint(block, *inputs, use_reentrant=False)
    else:
        loss = checkpoint(block, *inputs, use_reentrant=False, context_fn=context_fn)
    forward_calls = calls["fwd"]
    loss.backward()
    return forward_calls, dict(calls), [t.grad for t in inputs]


def _assert_exact(grads, reference):
    for got, want in zip(grads, reference, strict=True):
        torch.testing.assert_close(got, want, rtol=0, atol=0)


@pytest.mark.parametrize("varlen", [False, True])
def test_saved_fa4_forward_is_not_recomputed(cute, op_calls, varlen):
    make = _varlen_block if varlen else _dense_block
    block = make(cute)
    _, direct, reference = _run(cute, block, checkpointed=False)
    assert direct == {"fwd": 1, "bwd": 1}
    assert op_calls["fwd"] == 0

    state = SacPolicyState()
    forward, calls, grads = _run(
        cute, block, build_sac_context_fn(["attention"], state=state)
    )
    assert forward == 1 and calls == {"fwd": 1, "bwd": 1}
    assert op_calls["fwd"] == 1
    name = "axolotl::flash_attn_4_varlen_fwd" if varlen else "axolotl::flash_attn_4_fwd"
    assert state.saved_op_names == {name}
    _assert_exact(grads, reference)
    assert not fa4_ops._PENDING


@pytest.mark.parametrize("varlen", [False, True])
def test_unsaved_fa4_is_recomputed_with_the_same_grads(cute, varlen):
    block = (_varlen_block if varlen else _dense_block)(cute)
    _, _, reference = _run(cute, block, checkpointed=False)
    _, calls, grads = _run(cute, block)
    assert calls == {"fwd": 2, "bwd": 1}
    _assert_exact(grads, reference)
    _, calls, grads = _run(cute, block, build_sac_context_fn([]))
    assert calls == {"fwd": 2, "bwd": 1}
    _assert_exact(grads, reference)


def test_sliding_window_calls_are_recomputed(cute):
    block = _dense_block(cute, window_size=(3, 0))
    _, _, reference = _run(cute, block, checkpointed=False)
    state = SacPolicyState()
    _, calls, grads = _run(
        cute, block, build_sac_context_fn(["attention"], state=state)
    )
    assert calls == {"fwd": 2, "bwd": 1}
    assert state.sliding_op_names == {"axolotl::flash_attn_4_fwd"}
    _assert_exact(grads, reference)

    _, calls, grads = _run(
        cute, block, build_sac_context_fn(["attention"], save_sliding_window=True)
    )
    assert calls == {"fwd": 1, "bwd": 1}
    _assert_exact(grads, reference)


def test_sliding_layer_types_are_recomputed(cute):
    state = SacPolicyState()
    block = _dense_block(cute, state=state, layer_type="sliding_attention")
    _, _, reference = _run(cute, block, checkpointed=False)
    _, calls, grads = _run(
        cute, block, build_sac_context_fn(["attention"], state=state)
    )
    assert calls == {"fwd": 2, "bwd": 1}
    _assert_exact(grads, reference)

    full = _dense_block(cute, state=state, layer_type="full_attention")
    _, calls, _ = _run(cute, full, build_sac_context_fn(["attention"], state=state))
    assert calls == {"fwd": 1, "bwd": 1}


@pytest.mark.parametrize("varlen", [False, True])
def test_offloaded_fa4_save_is_not_recomputed(cute, varlen):
    block = (_varlen_block if varlen else _dense_block)(cute)
    _, _, reference = _run(cute, block, checkpointed=False)
    _, calls, grads = _run(cute, block, build_sac_offload_context_fn(["attention"]))
    assert calls == {"fwd": 1, "bwd": 1}
    _assert_exact(grads, reference)


def test_static_args_reach_fa4_unchanged(cute, monkeypatch):
    seen = {}
    original = cute.interface.FlashAttnFunc.func
    forward = original.forward

    def spy(ctx, *args, **kwargs):
        seen.update(kwargs)
        return forward(ctx, *args, **kwargs)

    monkeypatch.setattr(original, "forward", staticmethod(spy))
    block = _dense_block(cute, softmax_scale=0.3, softcap=12.5, window_size=(None, 0))
    _, calls, _ = _run(cute, block, build_sac_context_fn(["attention"]))
    assert calls["fwd"] == 1
    assert seen["softmax_scale"] == 0.3 and seen["softcap"] == 12.5
    assert seen["window_size"] == (None, 0) and seen["causal"] is True


def test_lost_ctx_reruns_the_forward_for_backward(cute):
    block = _dense_block(cute)
    _, _, reference = _run(cute, block, checkpointed=False)
    calls = cute.interface.CALLS
    calls.update(fwd=0, bwd=0)
    inputs = _weights()
    loss = checkpoint(
        block,
        *inputs,
        use_reentrant=False,
        context_fn=build_sac_context_fn(["attention"]),
    )
    fa4_ops._PENDING.clear()
    loss.backward()
    assert calls == {"fwd": 2, "bwd": 1}
    _assert_exact([t.grad for t in inputs], reference)


def test_unsupported_args_call_fa4_directly(cute, op_calls):
    sink = torch.zeros(HEADS, requires_grad=True)
    block = _dense_block(cute, learnable_sink=sink)
    _, calls, _ = _run(cute, block, build_sac_context_fn(["attention"]))
    assert calls == {"fwd": 2, "bwd": 1}
    assert op_calls["fwd"] == 0


def test_outside_sac_regions_fa4_is_called_directly(cute, op_calls):
    _run(cute, _dense_block(cute))
    _run(cute, _dense_block(cute), checkpointed=False)
    with torch.no_grad():
        q = torch.randn(1, 4, HEADS, HEAD_DIM)
        k = torch.randn(1, 4, KV_HEADS, HEAD_DIM)
        cute.flash_attn_func(q, k, k)
    assert op_calls["fwd"] == 0


def test_install_is_idempotent_and_skipped_without_registered_ops(cute, monkeypatch):
    route = cute.interface.FlashAttnFunc
    assert route._axolotl_fa4_route
    assert fa4_ops.install_fa4_ops(cute)
    assert cute.interface.FlashAttnFunc is route

    interface = types.ModuleType("flash_attn.cute.interface")
    exec(FAKE_INTERFACE, interface.__dict__)  # nosec B102
    original = interface.FlashAttnFunc
    monkeypatch.setattr(fa4_ops, "_REGISTERED", False)
    assert not fa4_ops.install_fa4_ops(interface)
    assert interface.FlashAttnFunc is original


def test_ops_match_the_sac_attention_rules():
    assert _is_flash_attention_forward("axolotl::flash_attn_4_fwd")
    assert _is_flash_attention_forward("axolotl::flash_attn_4_varlen_fwd")
    assert not _is_flash_attention_forward("axolotl::flash_attn_4_bwd")
    assert not _is_flash_attention_forward("axolotl::flash_attn_4_varlen_bwd")
    assert _window_arg_index(torch.ops.axolotl.flash_attn_4_fwd.default) == 3
    assert _window_arg_index(torch.ops.axolotl.flash_attn_4_varlen_fwd.default) == 7


def test_fake_shapes():
    static = '{"params": {}, "tuples": []}'
    with FakeTensorMode():
        q = torch.empty(2, 6, HEADS, HEAD_DIM)
        k = torch.empty(2, 6, KV_HEADS, HEAD_DIM)
        v = torch.empty(2, 6, KV_HEADS, 4)
        out, lse, token = torch.ops.axolotl.flash_attn_4_fwd(q, k, v, -1, -1, static)
        assert out.shape == (2, 6, HEADS, 4) and out.dtype == q.dtype
        assert lse.shape == (2, HEADS, 6) and lse.dtype == torch.float32
        assert token.shape == () and token.dtype == torch.int64

        qv, kv, vv = (t.flatten(0, 1) for t in (q, k, v))
        cu = torch.empty(3, dtype=torch.int32)
        out, lse, _ = torch.ops.axolotl.flash_attn_4_varlen_fwd(
            qv, kv, vv, cu, cu, None, None, -1, -1, static
        )
        assert out.shape == (12, HEADS, 4) and lse.shape == (HEADS, 12)

        grads = torch.ops.axolotl.flash_attn_4_bwd(
            out.new_empty(2, 6, HEADS, 4), None, q, k, v, q, lse, token, static
        )
        assert [g.shape for g in grads] == [q.shape, k.shape, v.shape]
        grads = torch.ops.axolotl.flash_attn_4_varlen_bwd(
            out, None, qv, kv, vv, cu, cu, None, None, out, lse, token, static
        )
        assert [g.shape for g in grads] == [qv.shape, kv.shape, vv.shape]


def test_configure_fa4_routes_and_keeps_the_first_call_log(cute, monkeypatch, caplog):
    monkeypatch.setattr(fa4, "_quack_supported", lambda: (True, None))
    monkeypatch.setattr(fa4_ops, "_ORIGINALS", {})
    interface = types.ModuleType("flash_attn.cute.interface")
    exec(FAKE_INTERFACE, interface.__dict__)  # nosec B102
    cute.interface = interface
    cute.flash_attn_func = interface.flash_attn_func
    cute.flash_attn_varlen_func = interface.flash_attn_varlen_func
    with caplog.at_level(logging.INFO, logger=fa4.LOG.name):
        fa4.configure_fa4()
        assert interface.FlashAttnFunc._axolotl_fa4_route
        assert interface.FlashAttnVarlenFunc._axolotl_fa4_route
        state = SacPolicyState()
        _, calls, _ = _run(
            cute, _dense_block(cute), build_sac_context_fn(["attention"], state=state)
        )
        _run(cute, _dense_block(cute), checkpointed=False)
    assert calls == {"fwd": 1, "bwd": 1}
    first_calls = [
        r for r in caplog.records if "first flash_attn_func call" in r.getMessage()
    ]
    assert len(first_calls) == 1


def test_fake_fa4_matches_sdpa(cute):
    """Keeps the stand-in honest: its forward and backward are causal GQA attention."""
    gen = torch.Generator().manual_seed(1)
    q = torch.randn(2, 6, HEADS, HEAD_DIM, generator=gen, dtype=torch.float64)
    k = torch.randn(2, 6, KV_HEADS, HEAD_DIM, generator=gen, dtype=torch.float64)
    v = torch.randn(2, 6, KV_HEADS, HEAD_DIM, generator=gen, dtype=torch.float64)
    dout = torch.randn(2, 6, HEADS, HEAD_DIM, generator=gen, dtype=torch.float64)
    leaves = [t.clone().requires_grad_(True) for t in (q, k, v)]
    (cute.flash_attn_func(*leaves, causal=True)[0] * dout).sum().backward()
    ref = [t.clone().requires_grad_(True) for t in (q, k, v)]
    out = torch.nn.functional.scaled_dot_product_attention(
        *(t.transpose(1, 2) for t in ref), is_causal=True, enable_gqa=True
    ).transpose(1, 2)
    (out * dout).sum().backward()
    for got, want in zip(leaves, ref, strict=True):
        torch.testing.assert_close(got.grad, want.grad)
