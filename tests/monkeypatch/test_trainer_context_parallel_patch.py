"""Tests for the HF Trainer context parallel patch."""

import pytest
from transformers import Trainer

from axolotl.monkeypatch.transformers.trainer_context_parallel import (
    GUARD_PATTERN,
    PATCHED_GUARD,
    patch_prepare_context_parallel_inputs,
)


@pytest.fixture
def restore_trainer_prepare_method():
    """Ensure Trainer._prepare_context_parallel_inputs is restored after a test."""
    original_method = getattr(
        Trainer,
        "_original_prepare_context_parallel_inputs",
        Trainer._prepare_context_parallel_inputs,
    )
    patched_attr_present = hasattr(
        Trainer, "_axolotl_prepare_context_parallel_inputs_patched"
    )

    yield

    Trainer._prepare_context_parallel_inputs = original_method
    if patched_attr_present:
        delattr(Trainer, "_axolotl_prepare_context_parallel_inputs_patched")
    if hasattr(Trainer, "_original_prepare_context_parallel_inputs"):
        delattr(Trainer, "_original_prepare_context_parallel_inputs")
    if hasattr(Trainer, "_axolotl_prepare_context_parallel_inputs_source"):
        delattr(Trainer, "_axolotl_prepare_context_parallel_inputs_source")


def test_patch_attention_guard(restore_trainer_prepare_method):
    """Patch should swap the guard to allow sdpa or flash attention."""
    # Ensure we start from the unpatched method
    if hasattr(Trainer, "_original_prepare_context_parallel_inputs"):
        Trainer._prepare_context_parallel_inputs = (
            Trainer._original_prepare_context_parallel_inputs
        )
        delattr(Trainer, "_original_prepare_context_parallel_inputs")
    if hasattr(Trainer, "_axolotl_prepare_context_parallel_inputs_patched"):
        delattr(Trainer, "_axolotl_prepare_context_parallel_inputs_patched")

    patch_prepare_context_parallel_inputs()

    patched_method = Trainer._prepare_context_parallel_inputs
    assert patched_method is not None
    assert getattr(Trainer, "_axolotl_prepare_context_parallel_inputs_patched", False)

    source = Trainer._axolotl_prepare_context_parallel_inputs_source
    assert GUARD_PATTERN not in source
    assert PATCHED_GUARD in source


def test_patch_is_idempotent(restore_trainer_prepare_method):
    """Calling the patch twice should leave the same patched function in place."""
    patch_prepare_context_parallel_inputs()
    first_patched = Trainer._prepare_context_parallel_inputs

    patch_prepare_context_parallel_inputs()
    second_patched = Trainer._prepare_context_parallel_inputs

    assert first_patched is second_patched
