"""TqdmParser — captures tqdm progress bar output and surfaces as structured events."""

from __future__ import annotations

import re

from axolotl.tui.io_capture import LineParser, register_parser


@register_parser
class TqdmParser(LineParser):
    priority = 15
    name = "tqdm"

    # Match tqdm-style progress lines, e.g.:
    #   Tokenizing Prompts (num_proc=24):  35%|███▍      | 19008/54568 [00:02<00:02, 17417.65 examples/s]
    #   Loading weights:  53%|█████▎    | 77/146 [00:00<00:00, 396.39it/s]
    #   0%|          | 0/30 [00:00<?, ?it/s]
    _TQDM_RE = re.compile(
        r"(?P<desc>.*?)\s*"
        r"(?P<pct>\d+)%\|[▏▎▍▌▋▊▉█░▓▒# ]*\|\s*"
        r"(?P<current>[\d,]+)/(?P<total>[\d,]+)"
        r"\s*\[(?P<elapsed>[^\]]*)\]"
    )

    # Also match simpler forms like:
    #   Fetching 0 files: 0it [00:00, ?it/s]
    _FETCH_RE = re.compile(r"(?P<desc>[\w\s]+):\s*(?P<current>\d+)(?:it)?\s*\[.*?\]")

    def parse(self, line: str, source: str) -> list[dict]:
        m = self._TQDM_RE.search(line)
        if m:
            desc = m.group("desc").strip().rstrip(":")
            pct = int(m.group("pct"))
            current = int(m.group("current").replace(",", ""))
            total = int(m.group("total").replace(",", ""))

            events: list[dict] = []

            # Surface as a log line with progress info
            if pct == 100 or pct == 0 or pct % 25 == 0:
                msg = (
                    f"[{desc}] {pct}% ({current}/{total})"
                    if desc
                    else f"{pct}% ({current}/{total})"
                )
                events.append(
                    {
                        "type": "log_line",
                        "level": "info",
                        "message": msg,
                    }
                )

            # Also emit as a progress metric
            cleaned_desc = desc.strip().lower().replace(" ", "_")
            if not cleaned_desc:
                cleaned_desc = "progress"
            events.append(
                {
                    "type": "metrics",
                    "logs": {
                        f"progress/{cleaned_desc}": pct / 100.0,
                    },
                }
            )

            return events

        # Fallback: try simpler fetch-style progress lines
        m = self._FETCH_RE.search(line)
        if m:
            desc = m.group("desc").strip().rstrip(":")
            current = int(m.group("current"))
            cleaned_desc = desc.strip().lower().replace(" ", "_")
            if not cleaned_desc:
                cleaned_desc = "fetch"
            return [
                {
                    "type": "log_line",
                    "level": "info",
                    "message": f"[{desc}] {current}" if desc else f"{current}",
                }
            ]

        return []
