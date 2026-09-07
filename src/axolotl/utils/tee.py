"""
Utilities for managing the debug log file and providing a file-only stream for logging
handlers.
"""

from __future__ import annotations

import io
import os
import sys
import threading
from pathlib import Path
from typing import TextIO, cast

_lock = threading.Lock()
_file_handle: io.TextIOWrapper | None = None
_log_path: str | None = None
_tee_installed: bool = False
_orig_stdout: TextIO | None = None
_orig_stderr: TextIO | None = None


class _FileOnlyWriter(io.TextIOBase):
    """A stream-like object that writes only to the tee file.

    Before the file is prepared, writes are dropped (no-op).
    """

    def write(self, s: str) -> int:  # type: ignore[override]
        with _lock:
            if _file_handle is not None:
                _file_handle.write(s)
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


class _StreamTee(io.TextIOBase):
    """A minimal tee that mirrors writes to the debug log file.

    Installed only after the debug log is prepared; no buffering.
    """

    def __init__(self, stream: io.TextIOBase):
        self._stream = stream

    def write(self, s: str) -> int:  # type: ignore[override]
        with _lock:
            n = self._stream.write(s)
            if _file_handle is not None:
                _file_handle.write(s)
            return n

    def flush(self) -> None:  # type: ignore[override]
        with _lock:
            self._stream.flush()
            if _file_handle is not None:
                try:
                    _file_handle.flush()
                except Exception:
                    pass

    @property
    def encoding(self):  # type: ignore[override]
        return getattr(self._stream, "encoding", None)

    @property
    def errors(self):  # type: ignore[override]
        return getattr(self._stream, "errors", None)

    def isatty(self):  # type: ignore[override]
        return getattr(self._stream, "isatty", lambda: False)()

    def fileno(self):  # type: ignore[override]
        if hasattr(self._stream, "fileno"):
            return self._stream.fileno()
        raise OSError("Underlying stream has no fileno")


def prepare_debug_log(cfg, filename: str = "debug.log") -> str:
    """
    Prepare the debug log.

    Creates the output directory, handles append/truncate logic based on cfg, and opens
    the debug log file for subsequent writes via file-only handlers.
    """
    global _file_handle, _log_path, _tee_installed

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

        if not append:
            log_path.unlink(missing_ok=True)

        fh = open(log_path, "a", encoding="utf-8")
        fh.flush()

        _file_handle = fh
        _log_path = str(log_path)

        # Install a tee so stdout/stderr are mirrored to the debug file
        # Allow disabling via env for testing or advanced usage.
        tee_enabled = os.getenv("AXOLOTL_TEE_STDOUT", "1").lower() not in {
            "0",
            "false",
            "no",
        }
        if tee_enabled and not _tee_installed:
            # Save originals so we can restore later (e.g., tests)
            global _orig_stdout, _orig_stderr
            _orig_stdout = sys.stdout
            _orig_stderr = sys.stderr
            sys.stdout = _StreamTee(cast(io.TextIOBase, sys.stdout))
            sys.stderr = _StreamTee(cast(io.TextIOBase, sys.stderr))
            _tee_installed = True

        return _log_path


def close_debug_log() -> None:
    """Flush and close the debug log and uninstall the stdout/stderr tee.

    Safe to call even if not initialized.
    """
    global _file_handle, _log_path, _tee_installed, _orig_stdout, _orig_stderr
    with _lock:
        # Restore original stdout/stderr if we installed a tee
        if _tee_installed:
            if _orig_stdout is not None:
                sys.stdout = _orig_stdout
            if _orig_stderr is not None:
                sys.stderr = _orig_stderr
            _tee_installed = False
            _orig_stdout = None
            _orig_stderr = None

        # Close the file handle if open
        if _file_handle is not None:
            try:
                _file_handle.flush()
                _file_handle.close()
            except Exception:
                pass
            finally:
                _file_handle = None
        _log_path = None
