"""Startup probes for the PEFT / transformers API surface ProTrain depends on.

PEFT and transformers occasionally rename or remove attributes between
minor releases. When that happens, the failure currently surfaces deep
inside the trainer as a stray ``AttributeError`` on ``LoraLayer`` or
``Trainer``. The probe below fails fast at config time with a single
actionable message that names the missing symbol and the validated
upper bounds.
"""

from __future__ import annotations

import warnings

from packaging.version import Version

VALIDATED_TRANSFORMERS_MAX = "5.9"
VALIDATED_PEFT_MAX = "0.21"


def assert_supported_peft_transformers_surface() -> None:
    """Probe the PEFT + transformers symbols this integration depends on.

    Raises RuntimeError at config time if a supported symbol is missing,
    rather than letting the failure surface deep inside the trainer.
    """
    missing: list[str] = []

    try:
        from peft.tuners.lora import LoraLayer
    except ImportError as exc:
        raise RuntimeError(
            f"protrain requires peft installed; failed to import LoraLayer: {exc}"
        ) from exc
    if not hasattr(LoraLayer, "adapter_layer_names"):
        missing.append("peft.tuners.lora.LoraLayer.adapter_layer_names")

    try:
        from transformers import Trainer
    except ImportError as exc:
        raise RuntimeError(
            f"protrain requires transformers installed; failed to import Trainer: {exc}"
        ) from exc
    if not hasattr(Trainer, "_load_from_checkpoint"):
        missing.append("transformers.Trainer._load_from_checkpoint")

    if missing:
        import peft
        import transformers

        raise RuntimeError(
            f"protrain: required peft/transformers surface missing: {missing}. "
            f"Installed peft={peft.__version__}, transformers={transformers.__version__}. "
            f"Validated upper bounds: transformers<{VALIDATED_TRANSFORMERS_MAX}, "
            f"peft<{VALIDATED_PEFT_MAX}. "
            f"Open an issue if you need support for a newer combination."
        )


def warn_on_unvalidated_versions() -> None:
    """Emit a warning when transformers/peft versions exceed validated bounds.

    The integration's PEFT-LoRA-container handling + checkpoint resume
    machinery were validated against transformers<5.6 and peft<0.21.
    Newer versions may work but have not been verified by the maintainers.
    """
    import peft
    import transformers

    if Version(transformers.__version__) >= Version(VALIDATED_TRANSFORMERS_MAX):
        warnings.warn(
            f"protrain: transformers {transformers.__version__} exceeds validated upper "
            f"bound (<{VALIDATED_TRANSFORMERS_MAX}). PEFT-LoRA-container hook fan-out "
            f"and Trainer._load_from_checkpoint were verified at the lower version; "
            f"newer versions may behave differently. Open an issue if you encounter "
            f"any regressions.",
            stacklevel=2,
        )
    if Version(peft.__version__) >= Version(VALIDATED_PEFT_MAX):
        warnings.warn(
            f"protrain: peft {peft.__version__} exceeds validated upper bound "
            f"(<{VALIDATED_PEFT_MAX}). LoraLayer.adapter_layer_names + ModuleDict "
            f"surface drift between versions has historically caused silent failures. "
            f"Open an issue if you encounter any regressions.",
            stacklevel=2,
        )


__all__ = [
    "VALIDATED_PEFT_MAX",
    "VALIDATED_TRANSFORMERS_MAX",
    "assert_supported_peft_transformers_surface",
    "warn_on_unvalidated_versions",
]
