"""TorchCompileParser — detects torch.compile graph breaks and recompilations."""

from __future__ import annotations

import re

from axolotl.tui.io_capture import LineParser, register_parser


@register_parser
class TorchCompileParser(LineParser):
    priority = 20
    name = "torch_compile"

    _RE = re.compile(r"Graph break|Recompiling|torch\.compile", re.IGNORECASE)

    def parse(self, line: str, source: str) -> list[dict]:
        if self._RE.search(line):
            return [
                {
                    "type": "log_line",
                    "level": "warning",
                    "message": f"⚡ compile: {line}",
                }
            ]
        return []
