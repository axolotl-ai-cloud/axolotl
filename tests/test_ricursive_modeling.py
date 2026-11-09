"""Ripple (`ricursive`) in-tree modeling: packing, checkpointing, caching."""

import pytest
import torch

from axolotl.model_support.ricursive.configuration_ricursive import RicursiveConfig
from axolotl.model_support.ricursive.modeling_ricursive import RicursiveForCausalLM

# window 4 < document lengths, so sliding masks cross the packing boundary;
# layer 4 is the global NoPE layer
CONFIG_KWARGS = dict(
    vocab_size=128,
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=5,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=8,
    sliding_window=4,
    use_valres=True,
    ffn_clamp_threshold=0.5,
    embed_scale=4.0,
)


def make_model(**overrides):
    torch.manual_seed(0)
    return RicursiveForCausalLM(RicursiveConfig(**{**CONFIG_KWARGS, **overrides}))


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
def test_packed_rows_match_separate_documents(attn_implementation):
    model = make_model(attn_implementation=attn_implementation).eval()
    first, second = torch.randint(0, 128, (1, 9)), torch.randint(0, 128, (1, 7))
    position_ids = torch.cat([torch.arange(9), torch.arange(7)])[None]

    with torch.no_grad():
        packed = model(
            input_ids=torch.cat([first, second], dim=1),
            position_ids=position_ids,
            use_cache=False,
        ).logits
        separate = torch.cat(
            [model(input_ids=first).logits, model(input_ids=second).logits], dim=1
        )

    torch.testing.assert_close(packed, separate)


@pytest.mark.parametrize("use_reentrant", [False, True])
def test_gradient_checkpointing_matches_full_backward(use_reentrant):
    model = make_model().train()
    input_ids = torch.randint(0, 128, (2, 12))

    def grads():
        model.zero_grad()
        model(input_ids=input_ids, labels=input_ids).loss.backward()
        return {name: p.grad.clone() for name, p in model.named_parameters()}

    expected = grads()
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
    )
    actual = grads()

    # the embedding gradient includes the value-residual path into every layer
    for name, grad in expected.items():
        torch.testing.assert_close(actual[name], grad, msg=name)


def test_cached_generation_matches_uncached():
    model = make_model().eval()
    prompt = torch.randint(0, 128, (1, 6))

    cached = model.generate(prompt, max_new_tokens=8, do_sample=False)
    uncached = model.generate(
        prompt, max_new_tokens=8, do_sample=False, use_cache=False
    )

    assert torch.equal(cached, uncached)
