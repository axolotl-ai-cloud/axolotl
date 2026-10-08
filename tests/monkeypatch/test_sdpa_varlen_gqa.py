"""CPU dispatch checks for native GQA and the legacy varlen API."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.attention.varlen as native

from axolotl.monkeypatch.attention.sdpa_varlen import _build_varlen_forward


@pytest.mark.parametrize("native_gqa", [True, False])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
def test_gqa_dispatch_and_gradients(monkeypatch, native_gqa, kv_heads):
    calls = []

    def compute(q, k, v, cu_q, cu_k, max_q, max_k, window_size, enable_gqa):
        calls.append((q.shape[1], k.shape[1], v.shape[1], enable_gqa))
        assert cu_q.tolist() == cu_k.tolist() == [0, 2, 4]
        return torch.cat(
            [
                torch.nn.functional.scaled_dot_product_attention(
                    q[start:end].transpose(0, 1),
                    k[start:end].transpose(0, 1),
                    v[start:end].transpose(0, 1),
                    is_causal=True,
                    enable_gqa=enable_gqa,
                ).transpose(0, 1)
                for start, end in ((0, 2), (2, 4))
            ]
        )

    def modern(q, k, v, cu_q, cu_k, max_q, max_k, *, window_size, enable_gqa=False):
        return compute(q, k, v, cu_q, cu_k, max_q, max_k, window_size, enable_gqa)

    def legacy(q, k, v, cu_q, cu_k, max_q, max_k, *, window_size):
        return compute(q, k, v, cu_q, cu_k, max_q, max_k, window_size, False)

    monkeypatch.setattr(native, "varlen_attn", modern if native_gqa else legacy)
    wrapper = _build_varlen_forward(lambda *a, **kw: pytest.fail("fallback"))
    inputs = [
        torch.randn(1, h, 4, 8, dtype=torch.float16, requires_grad=True)
        for h in (4, kv_heads, kv_heads)
    ]
    reference_inputs = [x.detach().clone().requires_grad_() for x in inputs]
    # The kernel is replaced with CPU SDPA; only the wrapper's CUDA dispatch is simulated.
    monkeypatch.setattr(torch.Tensor, "is_cuda", property(lambda self: True))
    output, _ = wrapper(
        SimpleNamespace(), *inputs, None, position_ids=torch.tensor([[0, 1, 0, 1]])
    )
    q, k, v = reference_inputs
    reference = torch.nn.functional.scaled_dot_product_attention(
        q,
        k.repeat_interleave(4 // kv_heads, 1),
        v.repeat_interleave(4 // kv_heads, 1),
        attn_mask=torch.block_diag(*[torch.ones(2, 2, dtype=torch.bool).tril()] * 2),
    ).transpose(1, 2)
    torch.testing.assert_close(output, reference, atol=2e-3, rtol=2e-3)
    output.float().square().sum().backward()
    reference.float().square().sum().backward()
    for actual, expected in zip(inputs, reference_inputs, strict=True):
        torch.testing.assert_close(actual.grad, expected.grad, atol=1e-2, rtol=1e-2)
    expected_heads = kv_heads if native_gqa else 4
    assert calls == [(4, expected_heads, expected_heads, native_gqa and kv_heads != 4)]


@pytest.mark.parametrize("architecture", ["llama", "mistral", "qwen3"])
def test_transformers_dispatch_preserves_kv_heads(monkeypatch, architecture):
    from transformers import AutoConfig, AutoModelForCausalLM
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    config = AutoConfig.for_model(
        architecture,
        vocab_size=32,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        use_cache=False,
    )
    config._attn_implementation = "sdpa"
    model = AutoModelForCausalLM.from_config(config)
    original = ALL_ATTENTION_FUNCTIONS["sdpa"]
    calls = []

    def record(module, query, key, value, attention_mask, **kwargs):
        calls.append((query.shape[1], key.shape[1], value.shape[1]))
        return original(module, query, key, value, attention_mask, **kwargs)

    monkeypatch.setitem(ALL_ATTENTION_FUNCTIONS._global_mapping, "sdpa", record)
    model(torch.tensor([[1, 2, 3]]))
    assert calls == [(4, 2, 2)]
