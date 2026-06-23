"""CPU-only tests for the scoped quantized-training guard (#4).

Tests run against a throwaway module object (not the global ``transformers.trainer``) so they are
fully isolated from conftest/session state.
"""

import types

from axolotl.integrations.kernels.quant_training_guard import (
    relax_quantized_training_guard,
    restore_quantized_training_guard,
)


class _Model:
    def __init__(self, peft):
        self._hf_peft_config_loaded = peft


class _PeftModel:
    """Mirrors get_peft_model() output: exposes .peft_config, no _hf_peft_config_loaded flag."""

    peft_config = {"default": object()}


class _TrackingGuard:
    def __init__(self):
        self.calls = []

    def __call__(self, model):
        self.calls.append(model)


def _fresh_module():
    tracker = _TrackingGuard()
    mod = types.SimpleNamespace(validate_quantization_for_training=tracker)
    return mod, tracker


def test_peft_model_skips_guard():
    mod, tracker = _fresh_module()
    relax_quantized_training_guard(mod)
    mod.validate_quantization_for_training(_Model(peft=True))
    assert tracker.calls == []  # PEFT/quantized is the supported pattern -> skipped


def test_get_peft_model_skips_guard():
    # axolotl's get_peft_model() path: .peft_config present, _hf_peft_config_loaded absent.
    mod, tracker = _fresh_module()
    relax_quantized_training_guard(mod)
    mod.validate_quantization_for_training(_PeftModel())
    assert (
        tracker.calls == []
    )  # recognized as PEFT -> skipped (FP8 base + adapters is supported)


def test_non_peft_delegates_to_original():
    mod, tracker = _fresh_module()
    relax_quantized_training_guard(mod)
    model = _Model(peft=False)
    mod.validate_quantization_for_training(model)
    assert tracker.calls == [model]  # delegated to the real guard


def test_idempotent():
    mod, _ = _fresh_module()
    relax_quantized_training_guard(mod)
    wrapped = mod.validate_quantization_for_training
    relax_quantized_training_guard(mod)  # second call must not re-wrap
    assert mod.validate_quantization_for_training is wrapped


def test_restore():
    mod, tracker = _fresh_module()
    relax_quantized_training_guard(mod)
    assert mod.validate_quantization_for_training is not tracker  # wrapped
    restore_quantized_training_guard(mod)
    assert (
        mod.validate_quantization_for_training is tracker
    )  # original preserved/restored
