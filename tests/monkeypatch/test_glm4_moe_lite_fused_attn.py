"""GLM fused attention parity, dispatch, and cache fallback."""

import copy

import pytest
import torch

from axolotl.utils.dict import DictDefault

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)


def _config(interleave=True, q_rank=64):
    from transformers.models.glm4_moe_lite.configuration_glm4_moe_lite import (
        Glm4MoeLiteConfig,
    )

    return Glm4MoeLiteConfig(
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        q_lora_rank=q_rank,
        kv_lora_rank=64,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
        rope_interleave=interleave,
        attention_dropout=0.0,
    )


def _relative_close(actual, expected):
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.03, atol=0.015)
    relative = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-8)
    assert relative < 0.015


@requires_cuda
@pytest.mark.parametrize("interleave", [False, True])
@pytest.mark.parametrize("q_rank", [None, 64])
@pytest.mark.parametrize("backend", ["sdpa", "kernels-community/flash-attn2@v3"])
def test_attention_outputs_and_all_parameter_gradients(interleave, q_rank, backend):
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        Glm4MoeLiteAttention,
        Glm4MoeLiteRotaryEmbedding,
    )

    from axolotl.monkeypatch.models.glm4_moe_lite.fused_attn import _make_fused_forward

    if "/" in backend:
        from transformers.integrations.hub_kernels import load_and_register_attn_kernel

        load_and_register_attn_kernel(backend)
    torch.manual_seed(42)
    config = _config(interleave, q_rank)
    config._attn_implementation = backend
    original = Glm4MoeLiteAttention(config, 0).cuda().bfloat16()
    fused = copy.deepcopy(original)
    fused.forward = _make_fused_forward(type(original).forward).__get__(fused)
    x = torch.randn(1, 34, 128, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    y = x.detach().clone().requires_grad_()
    positions = torch.arange(17, device="cuda").repeat(2).unsqueeze(0)
    cos, sin = Glm4MoeLiteRotaryEmbedding(config).cuda()(x, positions)
    mask = None
    if backend == "sdpa":
        index = torch.arange(34, device="cuda")
        mask = (
            (index[:, None] >= index[None, :])
            & (index[:, None] // 17 == index[None, :] // 17)
        )[None, None]
    expected = original(x, (cos, sin), mask, position_ids=positions)[0]
    actual = fused(y, (cos, sin), mask, position_ids=positions)[0]
    _relative_close(actual, expected)
    gradient = torch.randn_like(expected)
    expected.backward(gradient)
    actual.backward(gradient)
    _relative_close(y.grad, x.grad)
    for (name, reference), (_, candidate) in zip(
        original.named_parameters(), fused.named_parameters(), strict=True
    ):
        assert candidate.grad is not None, name
        _relative_close(candidate.grad, reference.grad)


@requires_cuda
def test_cached_decoding_preserves_compressed_cache(monkeypatch):
    from transformers import DynamicCache
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        Glm4MoeLiteAttention,
        Glm4MoeLiteRotaryEmbedding,
    )

    from axolotl.kernels import glm4_moe_lite as kernels
    from axolotl.monkeypatch.models.glm4_moe_lite.fused_attn import _make_fused_forward

    def unexpected(*args, **kwargs):
        pytest.fail("Cached decoding must retain compressed MLA cache semantics")

    monkeypatch.setattr(kernels, "fused_mla_prepare", unexpected)
    config = _config()
    config._attn_implementation = "sdpa"
    original = Glm4MoeLiteAttention(config, 0).cuda().bfloat16().eval()
    fused = copy.deepcopy(original)
    fused.forward = _make_fused_forward(type(original).forward).__get__(fused)
    caches = [DynamicCache(config=config), DynamicCache(config=config)]
    rotary = Glm4MoeLiteRotaryEmbedding(config).cuda()
    with torch.no_grad():
        for step in range(2):
            x = torch.randn(1, 1, 128, device="cuda", dtype=torch.bfloat16)
            embeddings = rotary(x, torch.tensor([[step]], device="cuda"))
            expected = original(x, embeddings, None, past_key_values=caches[0])[0]
            actual = fused(x, embeddings, None, past_key_values=caches[1])[0]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert caches[1].get_seq_length() == 2
    assert caches[1].layers[0].keys.shape[-1] == config.kv_lora_rank
    assert caches[1].layers[0].values.shape[-1] == config.qk_rope_head_dim


def test_cpu_fallback():
    from axolotl.monkeypatch.models.glm4_moe_lite.fused_attn import _make_fused_forward

    calls = []

    def original(self, hidden_states, position_embeddings, attention_mask, **kwargs):
        calls.append(kwargs)
        return hidden_states, None

    x = torch.randn(1, 2, 3)
    cos = sin = torch.ones(1, 2, 2)
    assert _make_fused_forward(original)(object(), x, (cos, sin), None)[0] is x
    assert len(calls) == 1


def test_dispatch_is_opt_in_and_patch_is_idempotent(monkeypatch):
    from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
        Glm4MoeLiteAttention,
    )

    from axolotl.loaders.patch_manager import PatchManager

    original = Glm4MoeLiteAttention.forward
    monkeypatch.setattr(Glm4MoeLiteAttention, "forward", original)
    monkeypatch.delattr(
        Glm4MoeLiteAttention, "_axolotl_fused_attn_patched", raising=False
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    cfg = DictDefault(
        model_config_type="glm4_moe_lite",
        context_parallel_size=1,
        fused_attn_kernel=False,
    )
    manager = PatchManager(cfg, None)
    manager._apply_model_support_pre_load_hook()
    assert Glm4MoeLiteAttention.forward is original
    cfg.fused_attn_kernel = True
    try:
        manager._apply_model_support_pre_load_hook()
        patched = Glm4MoeLiteAttention.forward
        assert patched is not original
        manager._apply_model_support_pre_load_hook()
        assert Glm4MoeLiteAttention.forward is patched
    finally:
        del Glm4MoeLiteAttention._axolotl_fused_attn_patched
