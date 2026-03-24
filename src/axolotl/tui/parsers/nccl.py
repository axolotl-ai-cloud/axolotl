"""NCCLErrorParser — surfaces NCCL errors as red alert events."""

from __future__ import annotations

import re

from axolotl.tui.io_capture import LineParser, register_parser


@register_parser
class NCCLErrorParser(LineParser):
    priority = 10
    name = "nccl_error"

    _RE = re.compile(r"NCCL error|Unhandled NCCL", re.IGNORECASE)

    def parse(self, line: str, source: str) -> list[dict]:
        if self._RE.search(line):
            return [
                {
                    "type": "log_line",
                    "level": "error",
                    "message": f"⚠ NCCL: {line}",
                },
                {"type": "alert", "severity": "error", "message": line},
            ]
        return []
