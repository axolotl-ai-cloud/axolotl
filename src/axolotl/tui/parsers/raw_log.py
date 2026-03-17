"""RawLogParser — catches every line as a log_line event."""

from __future__ import annotations

import re

from axolotl.tui.io_capture import LineParser, register_parser


@register_parser
class RawLogParser(LineParser):
    priority = 99
    name = "raw_log"

    _LOG_RE = re.compile(
        r"^(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}[,\.]\d+)"
        r"\s*[-–]\s*(?P<level>DEBUG|INFO|WARNING|ERROR|CRITICAL)"
        r"\s*[-–]\s*(?P<msg>.+)$",
        re.IGNORECASE,
    )

    # Filter out tqdm progress bar lines and other noisy output
    _TQDM_RE = re.compile(r"^\s*\d+%\|.*\|")
    _EMPTY_RE = re.compile(r"^\s*$")

    def parse(self, line: str, source: str) -> list[dict]:
        # Skip empty lines and tqdm progress bar updates
        if self._EMPTY_RE.match(line) or self._TQDM_RE.match(line):
            return []

        m = self._LOG_RE.match(line)
        level = (
            m.group("level").lower()
            if m
            else ("error" if source == "stderr" else "info")
        )
        return [{"type": "log_line", "level": level, "message": line}]
