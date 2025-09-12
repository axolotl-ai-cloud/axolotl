"""
Utilities to tee all terminal output to a file, with buffering until
the final output directory is known.

Usage:
- start_output_buffering(): install tees on stdout/stderr and buffer to a
  spooled temp file.
- finalize_output_logging(output_dir): create `<output_dir>/debug.log`, dump
  buffered content, and continue teeing to that file.
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tempfile
import threading
from datetime import datetime
from typing import Optional, cast

_lock = threading.Lock()
_started = False
_finalized = False
_spool: Optional[tempfile.SpooledTemporaryFile] = None
_file_handle: Optional[io.TextIOWrapper] = None
_log_path: Optional[str] = None
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr


class _StreamTee(io.TextIOBase):
    def __init__(self, stream: io.TextIOBase):
        self._stream = stream

    def write(self, s: str) -> int:
        # Keep behavior consistent with TextIO: return number of characters written
        with _lock:
            n = self._stream.write(s)
            # Write to spool or final file
            if _file_handle is not None:
                _file_handle.write(s)
                _file_handle.flush()
            elif _spool is not None:
                _spool.write(s)
                _spool.flush()
            return n

    def flush(self) -> None:
        with _lock:
            try:
                self._stream.flush()
            except Exception:
                pass
            if _file_handle is not None:
                try:
                    _file_handle.flush()
                except Exception:
                    pass
            elif _spool is not None:
                try:
                    _spool.flush()
                except Exception:
                    pass

    # Compatibility proxies
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


def start_output_buffering() -> None:
    """Install tees on stdout/stderr and start buffering output.

    Safe to call multiple times; only the first call takes effect.
    """
    global _started, _spool
    if _started:
        return
    with _lock:
        if _started:
            return
        # Buffer up to ~8MB in memory, then spill to disk
        _spool = tempfile.SpooledTemporaryFile(
            max_size=8 * 1024 * 1024, mode="w+", encoding="utf-8"
        )
        # mypy: sys.stdout/err are TextIO, cast to TextIOBase for our adapter
        sys.stdout = _StreamTee(cast(io.TextIOBase, sys.stdout))
        sys.stderr = _StreamTee(cast(io.TextIOBase, sys.stderr))
        _started = True


def finalize_output_logging(output_dir: str, filename: str = "debug.log") -> str:
    """Finalize logging to a file under the provided output_dir.

    - Creates the directory if needed
    - Writes a header, dumps buffered output, and continues teeing to the file
    - Subsequent calls are idempotent and return the same path
    """
    global _finalized, _file_handle, _log_path
    if not output_dir:
        return ""

    with _lock:
        if _finalized and _log_path:
            return _log_path

        os.makedirs(output_dir, exist_ok=True)
        _log_path = os.path.join(output_dir, filename)

        # Open in append mode to be safe across multi-invocations/processes
        fh = open(_log_path, "a", encoding="utf-8")

        # Emit a small header for this process
        ts = datetime.utcnow().isoformat(timespec="seconds") + "Z"
        rank = os.getenv("LOCAL_RANK", os.getenv("RANK", "0"))
        pid = os.getpid()
        header = f"\n===== [{ts}] BEGIN PROCESS pid={pid} rank={rank} =====\n"
        fh.write(header)

        # Dump buffered output if any
        if _spool is not None:
            try:
                _spool.seek(0)
                shutil.copyfileobj(_spool, fh)
            finally:
                try:
                    _spool.close()
                except Exception:
                    pass

        fh.flush()

        # Switch to final file
        _file_handle = fh
        _finalized = True

        return _log_path
