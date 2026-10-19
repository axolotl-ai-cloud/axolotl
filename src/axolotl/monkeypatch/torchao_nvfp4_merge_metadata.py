"""Quantizer identity for native NVFP4 merge-aware adapter exports."""

from axolotl.monkeypatch.torchao_nvfp4_merge import capture_native_nvfp4_recipe
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def build_native_merge_aware_metadata(weights, start_step=None):
    """Record each original checkpoint weight's native quantization recipe."""
    import torchao

    return {
        "backend": "native_torchao",
        "version": 1,
        "encoder": f"torchao-{torchao.__version__}",
        "targets": {
            name: capture_native_nvfp4_recipe(weight).fingerprint()
            for name, weight in sorted(weights.items())
        },
        "start_step": start_step,
    }


def validate_native_merge_aware_header(metadata):
    """Warn when native quantizer identity cannot establish merge parity."""
    import torchao

    expected_encoder = f"torchao-{torchao.__version__}"
    valid = (
        metadata.get("version") == 1
        and metadata.get("encoder") == expected_encoder
        and isinstance(metadata.get("targets"), dict)
        and bool(metadata.get("targets"))
    )
    if not valid:
        LOG.warning(
            "NVFP4 MERGE WARNING: native adapter quantizer metadata is incomplete "
            "or differs from the current encoder %s. Continuing with the base "
            "checkpoint's native recipe; merged-model parity is not guaranteed.",
            expected_encoder,
        )
    return valid


def validate_native_merge_aware_target(metadata, name, weight):
    """Compare an exported target recipe with the checkpoint actually being merged."""
    targets = metadata.get("targets")
    expected = targets.get(name) if isinstance(targets, dict) else None
    actual = capture_native_nvfp4_recipe(weight).fingerprint()
    if expected != actual:
        LOG.warning(
            "NVFP4 MERGE WARNING: the native quantization recipe for %s differs "
            "from the adapter's training recipe or is missing. Continuing with "
            "the base checkpoint's recipe; merged-model parity is not guaranteed.",
            name,
        )
        return False
    return True
