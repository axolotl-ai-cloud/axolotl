"""
Utilities for managing the debug log file and providing a file-only stream for logging
handlers.
"""

from __future__ import annotations

import io
import os
import threading
from pathlib import Path
from typing import Optional

_lock = threading.Lock()
_file_handle: Optional[io.TextIOWrapper] = None
_log_path: Optional[str] = None


class _FileOnlyWriter(io.TextIOBase):
    """A stream-like object that writes only to the tee file.

    Before the file is prepared, writes are dropped (no-op).
    """

    def write(self, s: str) -> int:  # type: ignore[override]
        with _lock:
            if _file_handle is not None:
                _file_handle.write(s)
                _file_handle.flush()
                return len(s)
            return len(s)

    def flush(self) -> None:  # type: ignore[override]
        with _lock:
            if _file_handle is not None:
                try:
                    _file_handle.flush()
                except Exception:
                    pass


file_only_stream: io.TextIOBase = _FileOnlyWriter()


def prepare_debug_log(cfg, filename: str = "debug.log") -> str:
    """
    Prepare the debug log.

    Creates the output directory, handles append/truncate logic based on cfg, and opens
    the debug log file for subsequent writes via file-only handlers.
    """
    global _file_handle, _log_path

    with _lock:
        # If already initialized, reuse existing path
        if _log_path is not None:
            return _log_path

        output_dir = cfg.output_dir
        os.makedirs(output_dir, exist_ok=True)

        log_path = Path(output_dir) / filename
        append = bool(
            cfg.get("resume_from_checkpoint") or cfg.get("auto_resume_from_checkpoints")
        )

        if not append and log_path.exists():
            log_path.unlink()

        fh = open(log_path, "a", encoding="utf-8")
        fh.flush()

        _file_handle = fh
        _log_path = str(log_path)

        return _log_path
