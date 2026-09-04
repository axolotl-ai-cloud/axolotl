"""Unit tests for the shared checkpoint weight-conversion registration helper."""

import pytest
from transformers.conversion_mapping import (
    get_checkpoint_conversion_mapping,
    register_checkpoint_conversion_mapping,
)
from transformers.core_model_loading import WeightRenaming

import axolotl.utils.weight_conversions as wc
from axolotl.utils.weight_conversions import register_weight_conversions


def _sources(mapping):
    return [tuple(entry.source_patterns) for entry in mapping]


@pytest.fixture
def key(request):
    """A unique model_type per test, cleaned out of both registries afterwards."""
    model_type = f"wc_test_{request.node.name}"
    yield model_type
    register_checkpoint_conversion_mapping(model_type, [], overwrite=True)
    wc._MERGE_REGISTERED_MODEL_TYPES.discard(model_type)


def _seed_builtin(key):
    builtin = WeightRenaming(["stock.weight"], ["stock.renamed"])
    register_checkpoint_conversion_mapping(key, [builtin], overwrite=True)
    return builtin


def test_merge_keeps_builtins_and_orders_new_first(key):
    _seed_builtin(key)
    new = WeightRenaming(["experts.new.weight"], ["experts.new.renamed"])

    register_weight_conversions(key, [new])

    # new entry leads (wins first-match), the stock built-in is preserved as a fallback
    assert _sources(get_checkpoint_conversion_mapping(key)) == [
        ("experts.new.weight",),
        ("stock.weight",),
    ]


def test_merge_is_idempotent(key):
    _seed_builtin(key)
    new = WeightRenaming(["experts.new.weight"], ["experts.new.renamed"])

    register_weight_conversions(key, [new])
    register_weight_conversions(key, [new])

    assert _sources(get_checkpoint_conversion_mapping(key)) == [
        ("experts.new.weight",),
        ("stock.weight",),
    ]


def test_merge_same_source_overrides_builtin(key):
    _seed_builtin(key)
    override = WeightRenaming(["stock.weight"], ["stock.override"])

    register_weight_conversions(key, [override])

    mapping = get_checkpoint_conversion_mapping(key)
    assert _sources(mapping) == [("stock.weight",)]
    assert mapping[0].target_patterns == ["stock.override"]


def test_replace_drops_existing(key):
    _seed_builtin(key)
    new = WeightRenaming(["experts.new.weight"], ["experts.new.renamed"])

    register_weight_conversions(key, [new], replace_existing=True)

    assert _sources(get_checkpoint_conversion_mapping(key)) == [("experts.new.weight",)]


def test_replace_without_prior_merge_is_allowed(key):
    new = WeightRenaming(["experts.new.weight"], ["experts.new.renamed"])

    register_weight_conversions(key, [new], replace_existing=True)

    assert _sources(get_checkpoint_conversion_mapping(key)) == [("experts.new.weight",)]


def test_replace_after_merge_raises_not_implemented(key):
    register_weight_conversions(key, [WeightRenaming(["a.weight"], ["a.renamed"])])

    with pytest.raises(NotImplementedError, match="not supported yet"):
        register_weight_conversions(
            key, [WeightRenaming(["b.weight"], ["b.renamed"])], replace_existing=True
        )
