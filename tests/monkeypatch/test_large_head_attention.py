"""CPU-only tests for the generic large-head-dim attention capability (#5/large_head_attention)."""

import torch

import axolotl.monkeypatch.attention.flash_attn_d512 as flash_mod
from axolotl.monkeypatch.attention import large_head as lh


class Cfg(dict):
    def __getattr__(self, k):
        return self.get(k)


class _Mod:
    num_key_value_groups = 1


def _packed_pos(docs):
    return torch.cat([torch.arange(d) for d in docs])[None]


def test_resolve_policy_intent_alias_default():
    assert (
        lh.resolve_large_head_policy(Cfg(large_head_attention="triton_flash"))
        == "triton_flash"
    )
    assert lh.resolve_large_head_policy(Cfg(large_head_attention="sdpa")) == "sdpa"
    assert (
        lh.resolve_large_head_policy(Cfg(flash_attn_d512=True)) == "auto"
    )  # legacy alias
    assert lh.resolve_large_head_policy(Cfg()) == "sdpa"  # default


def test_route_declines_for_sdpa_policy():
    q = torch.zeros(1, 2, 8, 512)
    assert (
        lh.flash_d512_route(_Mod(), q, q, q, None, _packed_pos([4, 4]), policy="sdpa")
        is None
    )


def test_route_declines_for_unknown_policy():
    # A config typo must NOT silently route through the Triton kernel: only auto/triton_flash do.
    q = torch.zeros(1, 2, 8, 512)
    assert (
        lh.flash_d512_route(
            _Mod(), q, q, q, None, _packed_pos([4, 4]), policy="trtion_flsah"
        )
        is None
    )


def test_set_policy_resets_to_default():
    lh.set_large_head_policy("auto")
    assert lh.get_large_head_policy() == "auto"
    lh.set_large_head_policy(None)  # a later run without the field must reset to sdpa
    assert lh.get_large_head_policy() == "sdpa"


def test_route_declines_for_small_head_dim():
    q = torch.zeros(1, 2, 8, 128)  # head_dim 128 -> not a large-head case
    assert (
        lh.flash_d512_route(_Mod(), q, q, q, None, _packed_pos([4, 4]), policy="auto")
        is None
    )


def test_route_declines_single_doc_under_auto():
    q = torch.zeros(1, 2, 8, 512)
    single = torch.arange(8)[None]  # one document -> auto prefers SDPA is_causal
    assert lh.flash_d512_route(_Mod(), q, q, q, None, single, policy="auto") is None


def test_route_declines_custom_scale():
    q = torch.zeros(1, 2, 8, 512)
    assert (
        lh.flash_d512_route(_Mod(), q, q, q, 0.123, _packed_pos([4, 4]), policy="auto")
        is None
    )


def test_route_success_packed(monkeypatch):
    # stub the GPU kernel so the routing/layout logic is testable on CPU
    captured = {}

    def fake_flash(query, k, v, causal, position_ids=None):
        captured["causal"] = causal
        captured["pid_resets"] = int((position_ids == 0).sum())
        return torch.ones_like(query)  # [B, H, S, D]

    monkeypatch.setattr(flash_mod, "flash_d512", fake_flash)
    B, H, S, D = 1, 4, 8, 512
    q = torch.randn(B, H, S, D)
    out = lh.flash_d512_route(
        _Mod(), q, q, q, D**-0.5, _packed_pos([4, 4]), policy="auto"
    )
    assert out is not None
    attn, weights = out
    assert weights is None
    assert attn.shape == (B, S, H, D)  # transposed to sdpa_attention_forward's contract
    assert captured["causal"] is True and captured["pid_resets"] == 2


def test_route_gqa_repeats_kv(monkeypatch):
    seen = {}

    def fake_flash(query, k, v, causal, position_ids=None):
        seen["kv_heads"] = k.shape[1]
        return torch.ones_like(query)

    monkeypatch.setattr(flash_mod, "flash_d512", fake_flash)

    class GQA:
        num_key_value_groups = 4

    q = torch.randn(1, 16, 8, 512)
    kv = torch.randn(1, 4, 8, 512)  # 4 kv heads, 16 q heads
    lh.flash_d512_route(GQA(), q, kv, kv, 512**-0.5, _packed_pos([4, 4]), policy="auto")
    assert seen["kv_heads"] == 16  # repeated to match q heads


def test_patch_and_unpatch_sdpa(monkeypatch):
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    lh.set_large_head_policy("sdpa")
    assert lh.patch_sdpa_large_head() is False  # policy sdpa -> no wrap
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original

    assert lh.patch_sdpa_large_head("auto") is True  # wraps
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is not original
    assert lh.patch_sdpa_large_head("auto") is True  # idempotent (still wrapped)
    lh.unpatch_sdpa_large_head()
    assert ALL_ATTENTION_FUNCTIONS["sdpa"] is original
    lh.set_large_head_policy("sdpa")
