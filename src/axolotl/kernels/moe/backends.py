import os
import warnings
from enum import Enum


class MOEBackend(str, Enum):
    AUTO = "auto"
    HF_TRITON = "hf_triton"
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


def _probe_hf_triton() -> bool:
    try:
        # The hub loads kernels lazily; this import is a light probe.
        import importlib

        importlib.import_module("kernels")
        return True
    except Exception:
        return False


def get_moe_backend_name(preferred: str | None = None) -> MOEBackend:
    """
    Resolve the desired MoE backend using, in order of precedence:
    - explicit preferred argument
    - environment variable AXOLOTL_MOE_BACKEND
    - auto detection
    """
    choice = (preferred or os.getenv("AXOLOTL_MOE_BACKEND") or "auto").lower()
    try:
        selected = MOEBackend(choice)
    except ValueError:
        warnings.warn(f"Unknown moe backend '{choice}', falling back to auto")
        selected = MOEBackend.AUTO

    if selected == MOEBackend.AUTO:
        if _probe_torch_grouped():
            return MOEBackend.TORCH_GROUPED
        if _probe_hf_triton():
            return MOEBackend.HF_TRITON
        return MOEBackend.NAIVE
    if selected == MOEBackend.TORCH_GROUPED and not _probe_torch_grouped():
        warnings.warn(
            "torch_grouped requested but torch>=2.8 not detected; falling back to hf_triton/naive"
        )
        return MOEBackend.HF_TRITON if _probe_hf_triton() else MOEBackend.NAIVE
    if selected == MOEBackend.HF_TRITON and not _probe_hf_triton():
        warnings.warn(
            "hf_triton requested but kernels hub not available; falling back to torch_grouped/naive"
        )
        return MOEBackend.TORCH_GROUPED if _probe_torch_grouped() else MOEBackend.NAIVE
    return selected
