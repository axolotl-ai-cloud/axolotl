"""Correctness tests for the cu_seqlens varlen SDPA packing path (``sdpa_varlen``)."""

import pytest
import torch
import torch.nn.functional as F

from axolotl.monkeypatch.attention.sdpa_varlen import (
    patch_sdpa_varlen,
    unpatch_sdpa_varlen,
    varlen_available,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen_attn needs CUDA"),
    pytest.mark.skipif(
        not varlen_available(), reason="torch.nn.attention.varlen needs torch >= 2.11"
    ),
]

DEV = "cuda"


class _Mod:
    def __init__(self, sliding_window=None, num_key_value_groups=4):
        self.sliding_window = sliding_window
        self.num_key_value_groups = num_key_value_groups


def _block_mask(position_ids, dtype, sliding=None):
    B, S = position_ids.shape
    doc = (position_ids == 0).cumsum(-1)
    same = doc[:, :, None] == doc[:, None, :]
    idx = torch.arange(S, device=DEV)
    allow = same & (idx[:, None] >= idx[None, :])[None]
    if sliding:
        allow = allow & ((idx[:, None] - idx[None, :]) < sliding)[None]
    m = torch.zeros(B, 1, S, S, device=DEV, dtype=dtype)
    m.masked_fill_(~allow[:, None], torch.finfo(dtype).min)
    return m


def _ref(q, k, v, scaling, position_ids, sliding=None):
    Hq, Hkv = q.shape[1], k.shape[1]
    kr = k.repeat_interleave(Hq // Hkv, 1)
    vr = v.repeat_interleave(Hq // Hkv, 1)
    o = F.scaled_dot_product_attention(
        q, kr, vr, attn_mask=_block_mask(position_ids, q.dtype, sliding), scale=scaling
    )
    return o.transpose(1, 2)  # [B,S,H,D]


def _pos(doclens_per_row):
    rows = [torch.cat([torch.arange(ln) for ln in dl]) for dl in doclens_per_row]
    S = max(len(r) for r in rows)
    return torch.stack([F.pad(r, (0, S - len(r))) for r in rows]).to(DEV)


@pytest.fixture
def patched():
    assert patch_sdpa_varlen()
    yield
    unpatch_sdpa_varlen()


@pytest.mark.parametrize(
    "doclens,sliding,label",
    [
        ([[2048, 2048]], None, "B1-2docs"),
        ([[1000, 1500, 1596]], None, "B1-3docs"),
        ([[2048, 2048]], 1024, "B1-sliding"),
        ([[2000, 2096], [1500, 1000, 1596]], None, "B2-padded"),
    ],
)
def test_varlen_matches_block_mask_sdpa(patched, doclens, sliding, label):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    sdpa = ALL_ATTENTION_FUNCTIONS["sdpa"]
    pos = _pos(doclens)
    B, S = pos.shape
    Hq, Hkv, D = 16, 4, 256
    torch.manual_seed(0)
    q = torch.randn(B, Hq, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(B, Hkv, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(B, Hkv, S, D, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    qr, kr, vr = (t.detach().clone().requires_grad_() for t in (q, k, v))
    mod = _Mod(sliding_window=sliding, num_key_value_groups=Hq // Hkv)
    scaling = D**-0.5

    out, _ = sdpa(
        mod, q, k, v, None, scaling=scaling, position_ids=pos, sliding_window=sliding
    )
    out.float().pow(2).mean().backward()
    ref = _ref(qr, kr, vr, scaling, pos, sliding)
    ref.float().pow(2).mean().backward()

    assert F.cosine_similarity(out.float().flatten(), ref.float().flatten(), 0) > 0.999
    assert (
        F.cosine_similarity(q.grad.float().flatten(), qr.grad.float().flatten(), 0)
        > 0.999
    )


def _assert_defers_to_stock(mod, q, k, v, pos):
    """The wrapper must produce exactly the stock-SDPA result (it just calls through)."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    wrapper = ALL_ATTENTION_FUNCTIONS["sdpa"]
    original = wrapper._axolotl_sdpa_original
    scaling = q.shape[-1] ** -0.5
    out_w, _ = wrapper(mod, q, k, v, None, scaling=scaling, position_ids=pos)
    out_o, _ = original(mod, q, k, v, None, scaling=scaling, position_ids=pos)
    assert torch.equal(out_w, out_o)


def test_varlen_matches_flash_attention_2_e2e():
    """The varlen path matches FA2 (the canonical varlen-packing impl) on a real packed forward."""
    from transformers import LlamaConfig, LlamaModel

    def build(impl):
        cfg = LlamaConfig(
            hidden_size=512,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=2,
            intermediate_size=1024,
            vocab_size=256,
            head_dim=64,
            _attn_implementation=impl,
        )
        torch.manual_seed(1)
        return LlamaModel(cfg).to(DEV).to(torch.bfloat16).eval()

    docs = [20, 30, 14]
    ids = torch.cat([torch.randint(0, 256, (1, d), device=DEV) for d in docs], 1)
    pos = torch.cat([torch.arange(d) for d in docs]).to(DEV)[None]
    try:
        with torch.no_grad():
            fa2 = build("flash_attention_2")(
                input_ids=ids, position_ids=pos
            ).last_hidden_state
    except Exception:  # pylint: disable=broad-except
        pytest.skip("flash_attention_2 unavailable")

    assert patch_sdpa_varlen()
    try:
        with torch.no_grad():
            varlen = build("sdpa")(input_ids=ids, position_ids=pos).last_hidden_state
    finally:
        unpatch_sdpa_varlen()
    assert (
        F.cosine_similarity(varlen.float().flatten(), fa2.float().flatten(), 0) > 0.999
    )


def test_falls_back_when_not_packed(patched):
    """Single-document rows must defer to stock SDPA (no varlen path)."""
    S, Hq, Hkv, D = 512, 16, 4, 256
    pos = torch.arange(S, device=DEV)[None]  # one document -> no packing
    torch.manual_seed(0)
    q = torch.randn(1, Hq, S, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(1, Hkv, S, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(1, Hkv, S, D, device=DEV, dtype=torch.bfloat16)
    _assert_defers_to_stock(_Mod(num_key_value_groups=Hq // Hkv), q, k, v, pos)


def test_falls_back_on_large_head_dim(patched):
    """head_dim > 256 can't use Flash varlen; must defer to stock SDPA."""
    pos = _pos([[256, 256]])
    B, S = pos.shape
    Hq, Hkv, D = 8, 2, 512  # head_dim 512
    torch.manual_seed(0)
    q = torch.randn(B, Hq, S, D, device=DEV, dtype=torch.bfloat16)
    k = torch.randn(B, Hkv, S, D, device=DEV, dtype=torch.bfloat16)
    v = torch.randn(B, Hkv, S, D, device=DEV, dtype=torch.bfloat16)
    _assert_defers_to_stock(_Mod(num_key_value_groups=Hq // Hkv), q, k, v, pos)
