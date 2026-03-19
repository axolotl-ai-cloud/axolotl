"""I/O capture: OS-level stdout/stderr redirect, line parser chain, and parser registry."""

from __future__ import annotations

import logging
import os
import queue
import sys
import threading
from abc import ABC, abstractmethod
from datetime import datetime
from typing import IO

# ---------------------------------------------------------------------------
# Parser registry
# ---------------------------------------------------------------------------

_parser_registry: list[type[LineParser]] = []


def register_parser(cls: type[LineParser]) -> type[LineParser]:
    """Decorator to register a LineParser subclass."""
    if cls not in _parser_registry:
        _parser_registry.append(cls)
    return cls


def get_registered_parsers() -> list[type[LineParser]]:
    return list(_parser_registry)


# ---------------------------------------------------------------------------
# Base LineParser
# ---------------------------------------------------------------------------


class LineParser(ABC):
    """Base class for stdout/stderr line parsers."""

    priority: int = 50
    name: str = ""

    @abstractmethod
    def parse(self, line: str, source: str) -> list[dict]:
        """Parse a single captured line.

        Args:
            line:   one line of captured output, trailing newline stripped.
            source: "stdout" or "stderr".

        Returns:
            List of event dicts to push onto the metric queue.
            Return [] if this line is not relevant.
        """
        ...


# ---------------------------------------------------------------------------
# ParserChain
# ---------------------------------------------------------------------------


class ParserChain:
    def __init__(self):
        self._parsers: list[LineParser] = []

    def register(self, parser: LineParser) -> None:
        self._parsers.append(parser)
        self._parsers.sort(key=lambda p: p.priority)

    def parse(self, line: str, source: str = "stdout") -> list[dict]:
        events: list[dict] = []
        for parser in self._parsers:
            events.extend(parser.parse(line, source))
        return events


# ---------------------------------------------------------------------------
# IOCapture — OS-level fd redirect to pipe
# ---------------------------------------------------------------------------


class IOCapture:
    """Redirects fd 1 and fd 2 into an OS pipe, drains via a reader thread,
    passes lines through a ParserChain, and tees to a log file."""

    def __init__(
        self, log_path: str, parser_chain: ParserChain, metric_queue: queue.Queue
    ):
        self._parser_chain = parser_chain
        self._queue = metric_queue
        self._log_path = log_path
        self._log_file: IO[str] | None = None
        self._thread: threading.Thread | None = None
        self._read_fd: int | None = None
        self._write_fd: int | None = None
        self._saved_stdout_fd: int | None = None
        self._saved_stderr_fd: int | None = None

    def start(self) -> None:
        # Write run-start separator
        self._log_file = open(self._log_path, "a", buffering=1)  # noqa: SIM115
        self._log_file.write(
            f"\n=== axolotl run started {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===\n"
        )
        self._log_file.flush()

        # OS-level pipe
        self._read_fd, self._write_fd = os.pipe()

        # Save originals
        self._saved_stdout_fd = os.dup(1)
        self._saved_stderr_fd = os.dup(2)

        # Redirect both stdout and stderr into the write end
        os.dup2(self._write_fd, 1)
        os.dup2(self._write_fd, 2)
        os.close(self._write_fd)  # write end now held by fds 1 and 2

        # Also redirect Python-level handles
        sys.stdout = open(1, "w", buffering=1, closefd=False)  # noqa: SIM115
        sys.stderr = open(2, "w", buffering=1, closefd=False)  # noqa: SIM115

        # Drain thread
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        # Restore fds — closes the write end, causing reader to see EOF
        if self._saved_stdout_fd is not None and self._saved_stderr_fd is not None:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            os.dup2(self._saved_stdout_fd, 1)
            os.dup2(self._saved_stderr_fd, 2)
            os.close(self._saved_stdout_fd)
            os.close(self._saved_stderr_fd)
            self._saved_stdout_fd = None
            self._saved_stderr_fd = None

        if self._thread is not None:
            self._thread.join(timeout=2.0)
            if self._thread.is_alive():
                logging.getLogger(__name__).warning(
                    "IO capture thread did not exit after 2s"
                )
            self._thread = None

        if self._log_file is not None:
            self._log_file.close()
            self._log_file = None

    def _drain(self) -> None:
        # Read raw bytes and split on both \n and \r to handle tqdm progress bars
        # which use \r for in-place updates without \n
        assert self._read_fd is not None, "_drain called before start()"
        with os.fdopen(self._read_fd, "rb") as pipe:
            buf = b""
            while True:
                chunk = pipe.read(4096)
                if not chunk:
                    # EOF — process remaining buffer
                    if buf:
                        self._process_line(buf.decode("utf-8", errors="replace"))
                    break
                buf += chunk
                # Split on \n or \r
                while b"\n" in buf or b"\r" in buf:
                    # Find the earliest delimiter
                    idx_n = buf.find(b"\n")
                    idx_r = buf.find(b"\r")
                    if idx_n == -1:
                        idx = idx_r
                    elif idx_r == -1:
                        idx = idx_n
                    else:
                        idx = min(idx_n, idx_r)
                    line = buf[:idx].decode("utf-8", errors="replace")
                    buf = buf[idx + 1 :]
                    # Handle \r\n as single delimiter
                    if buf.startswith(b"\n"):
                        buf = buf[1:]
                    if line:
                        self._process_line(line)

    def _process_line(self, line: str) -> None:
        line = line.rstrip()
        if not line:
            return
        if self._log_file:
            self._log_file.write(line + "\n")
            self._log_file.flush()
        for event in self._parser_chain.parse(line):
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                pass
