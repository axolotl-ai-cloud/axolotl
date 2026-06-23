"""Scoped relaxation of transformers' quantized-training guard.

``transformers.trainer.validate_quantization_for_training`` rejects pre-quantized FP8/NVFP4
checkpoints for training. Its FP8/NVFP4 branch fires even when LoRA adapters are attached and the
quantized base is frozen — the supported QLoRA-style pattern. Rather than globally no-op'ing the
guard for all runs, we wrap it: for PEFT models (frozen quantized base + trainable adapters) we
skip, and for everything else we delegate to the original guard unchanged. Idempotent; preserves
the original callable for restore.
"""

from __future__ import annotations

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

_ORIG_ATTR = "_axolotl_quant_guard_original"
_FN_NAME = "validate_quantization_for_training"


def _target_module(module):
    if module is not None:
        return module
    import transformers.trainer as hf_trainer

    return hf_trainer


def relax_quantized_training_guard(module=None) -> None:
    """Install the scoped guard wrapper (idempotent).

    ``module`` defaults to ``transformers.trainer``; tests pass a throwaway module to avoid
    touching global state.
    """
    mod = _target_module(module)
    current = getattr(mod, _FN_NAME, None)
    if current is None or getattr(current, _ORIG_ATTR, None) is not None:
        return  # missing or already wrapped

    original = current

    def guarded(model):
        # LoRA/PEFT on a frozen quantized base is the supported (QLoRA-style) pattern; the upstream
        # FP8/NVFP4 branch rejects it anyway. Skip only for PEFT models; delegate the rest.
        # _hf_peft_config_loaded is set by HF-native model.add_adapter(); axolotl applies LoRA via
        # get_peft_model() -> PeftModel, which exposes .peft_config but NOT that flag. Check both, or
        # FP8 checkpoints (e.g. DeepSeek-V4-Flash-NVFP4) get rejected despite frozen-base + adapters.
        if getattr(model, "_hf_peft_config_loaded", False) or getattr(
            model, "peft_config", None
        ):
            return None
        return original(model)

    setattr(guarded, _ORIG_ATTR, original)
    setattr(mod, _FN_NAME, guarded)
    LOG.info(
        "kernels: scoped quantized-training guard (skips only PEFT/quantized; delegates others)"
    )


def restore_quantized_training_guard(module=None) -> None:
    """Restore the original guard (for tests/teardown)."""
    mod = _target_module(module)
    current = getattr(mod, _FN_NAME, None)
    original = getattr(current, _ORIG_ATTR, None)
    if original is not None:
        setattr(mod, _FN_NAME, original)
