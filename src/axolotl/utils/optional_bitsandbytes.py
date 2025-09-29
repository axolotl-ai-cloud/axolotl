"""Utility helpers to make bitsandbytes an optional dependency.

On platforms where bitsandbytes is unavailable (e.g., native Windows), we
want Axolotl to fall back gracefully to standard torch optimizers / dense
weights when quantization or BnB-specific optimizers are requested.
"""
from __future__ import annotations

from typing import Any, Optional

_BNB_AVAILABLE = False
_BNB_IMPORT_ERROR: Optional[Exception] = None
_bnb: Any = None

try:  # pragma: no cover - import side effect
    import bitsandbytes as _bnb  # type: ignore

    _bnb.__dict__  # touch to silence linters
    _BNB_AVAILABLE = True
    _bnb = _bnb
except Exception as e:  # broad to catch DLL load errors too
    _BNB_IMPORT_ERROR = e


def is_available() -> bool:
    return _BNB_AVAILABLE


def get_bnb() -> Any:
    if not _BNB_AVAILABLE:
        raise RuntimeError(
            "bitsandbytes is not available. Original import error: "
            f"{_BNB_IMPORT_ERROR!r}. If you intended to use 4/8-bit quantization, "
            "install bitsandbytes on a supported platform (Linux) or disable quantization."
        )
    return _bnb


def warn_if_unavailable(logger) -> None:
    if not _BNB_AVAILABLE and logger is not None:
        try:
            logger.warning(
                "bitsandbytes not found; continuing without 4/8-bit quantization or BnB optimizers."
            )
        except Exception:
            pass
