"""Tests for the memoized num_parameters."""

from unittest.mock import patch

import pytest
from transformers import LlamaConfig, LlamaForCausalLM
from transformers.modeling_utils import ModuleUtilsMixin

from axolotl.monkeypatch.transformers_num_parameters import patch_num_parameters


@pytest.fixture
def tiny_model():
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )
    return LlamaForCausalLM(config)


@pytest.fixture
def patched():
    installed = ModuleUtilsMixin.num_parameters
    original = installed
    if getattr(original, "_axolotl_cached", False):
        original = original.__wrapped__
    patch_num_parameters()
    yield original
    ModuleUtilsMixin.num_parameters = installed


def test_matches_original_for_every_argument_set(tiny_model, patched):
    for only_trainable in (False, True):
        for exclude_embeddings in (False, True):
            expected = patched(
                tiny_model,
                only_trainable=only_trainable,
                exclude_embeddings=exclude_embeddings,
            )
            got = tiny_model.num_parameters(
                only_trainable=only_trainable, exclude_embeddings=exclude_embeddings
            )
            assert got == expected


def test_second_call_does_not_walk_parameters(tiny_model, patched):
    tiny_model.num_parameters(exclude_embeddings=True)

    with patch.object(
        tiny_model, "named_parameters", side_effect=AssertionError("walked")
    ):
        assert tiny_model.num_parameters(exclude_embeddings=True) > 0


def test_cache_is_per_argument_set(tiny_model, patched):
    tiny_model.lm_head.weight.requires_grad_(False)

    all_params = tiny_model.num_parameters()
    trainable = tiny_model.num_parameters(only_trainable=True)

    assert trainable == all_params - tiny_model.lm_head.weight.numel()


def test_patch_is_idempotent(patched):
    first = ModuleUtilsMixin.num_parameters
    patch_num_parameters()
    assert ModuleUtilsMixin.num_parameters is first


def test_trainable_count_follows_requires_grad(tiny_model, patched):
    before = tiny_model.num_parameters(only_trainable=True)
    tiny_model.lm_head.weight.requires_grad_(False)

    after = tiny_model.num_parameters(only_trainable=True)

    assert before - after == tiny_model.lm_head.weight.numel()


def test_resize_token_embeddings_invalidates(tiny_model, patched):
    before = tiny_model.num_parameters()

    tiny_model.resize_token_embeddings(80)

    assert tiny_model.num_parameters() == patched(tiny_model)
    assert tiny_model.num_parameters() != before
