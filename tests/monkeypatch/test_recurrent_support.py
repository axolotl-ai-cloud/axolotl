"""Tests for the recurrent-model packing / context-parallel support table."""

import pytest

from axolotl.loaders.patch_manager import PatchManager
from axolotl.monkeypatch.models.recurrent_support import (
    PACKING_PATCHED,
    PACKING_UNSUPPORTED,
    validate_recurrent_model_config,
)
from axolotl.utils.dict import DictDefault


def _cfg(model_type, **overrides):
    return DictDefault(
        {
            "model_config_type": model_type,
            "sample_packing": False,
            "batch_flattening": False,
            "context_parallel_size": 1,
            **overrides,
        }
    )


def test_tables_are_disjoint():
    assert not PACKING_PATCHED & PACKING_UNSUPPORTED


def test_seq_idx_injected_models_are_listed_as_patched():
    assert set(PatchManager._SEQ_IDX_INJECTED_MODELS) <= PACKING_PATCHED


@pytest.mark.parametrize("model_type", sorted(PACKING_UNSUPPORTED))
@pytest.mark.parametrize("mode", ["sample_packing", "batch_flattening"])
def test_packing_rejected_for_unpatched_recurrent_models(model_type, mode):
    with pytest.raises(ValueError, match="state would leak"):
        validate_recurrent_model_config(_cfg(model_type, **{mode: True}))


@pytest.mark.parametrize("model_type", sorted(PACKING_PATCHED))
def test_packing_allowed_for_patched_recurrent_models(model_type):
    validate_recurrent_model_config(_cfg(model_type, sample_packing=True))


@pytest.mark.parametrize("model_type", sorted(PACKING_UNSUPPORTED))
def test_unpacked_training_allowed(model_type):
    validate_recurrent_model_config(_cfg(model_type))


@pytest.mark.parametrize("model_type", sorted(PACKING_PATCHED | PACKING_UNSUPPORTED))
def test_context_parallel_defers_to_runtime_mixer_validation(model_type):
    validate_recurrent_model_config(_cfg(model_type, context_parallel_size=4))


def test_plain_attention_models_are_ignored():
    validate_recurrent_model_config(
        _cfg("llama", sample_packing=True, context_parallel_size=4)
    )
