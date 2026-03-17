"""DeepSpeedParser — extracts DeepSpeed stage info and throughput metrics."""

from __future__ import annotations

import re

from axolotl.tui.io_capture import LineParser, register_parser


@register_parser
class DeepSpeedParser(LineParser):
    priority = 20
    name = "deepspeed"

    _SAMPLES_RE = re.compile(r"samples/sec=([0-9.]+)")
    _STAGE_RE = re.compile(r"ZeRO Stage (\d)")

    def parse(self, line: str, source: str) -> list[dict]:
        events: list[dict] = []
        if m := self._SAMPLES_RE.search(line):
            events.append(
                {
                    "type": "metrics",
                    "logs": {"samples_per_second": float(m.group(1))},
                }
            )
        if m := self._STAGE_RE.search(line):
            events.append({"type": "run_info", "zero_stage": int(m.group(1))})
        return events
