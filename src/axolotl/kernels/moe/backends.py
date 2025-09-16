import warnings
from enum import Enum


class MOEBackend(str, Enum):
    AUTO = "auto"
    TORCH_GROUPED = "torch_grouped"
    NAIVE = "naive"


def _probe_torch_grouped() -> bool:
    try:
        import torch  # noqa: F401

        # Prefer a simple version check; exact APIs may vary across 2.8+.
        ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        return ver >= (2, 8)
    except Exception:
        return False


def get_moe_backend_name(preferred: str | None = None) -> MOEBackend:
    """
    Resolve the desired MoE backend using, in order of precedence:
    - explicit preferred argument (e.g., from config)
    - auto detection
    """
    choice = (preferred or "auto").lower()
    try:
        selected = MOEBackend(choice)
    except ValueError:
        warnings.warn(f"Unknown moe backend '{choice}', falling back to auto")
        selected = MOEBackend.AUTO

    if selected == MOEBackend.AUTO:
        if _probe_torch_grouped():
            return MOEBackend.TORCH_GROUPED
        return MOEBackend.NAIVE
    if selected == MOEBackend.TORCH_GROUPED and not _probe_torch_grouped():
        warnings.warn(
            "torch_grouped requested but torch>=2.8 not detected; falling back to naive"
        )
        return MOEBackend.NAIVE
    return selected
