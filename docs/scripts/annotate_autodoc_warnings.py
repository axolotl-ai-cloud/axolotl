"""Report the docstring warnings in a `quartodoc build` log as GitHub annotations.

Writes `::warning` workflow commands and a job summary table. Never fails the build.
"""

import argparse
import os
import re
from pathlib import Path

# griffe prints `<module>` when it has no file path, and line 0 when it has no line
WARNING_RE = re.compile(
    r"^(?P<path>\S+\.py|<module(?:: [^>]+)?>):(?P<line>\d+): (?P<message>.+)$"
)
# GitHub shows at most 10 warning annotations per step and drops the rest
MAX_ANNOTATIONS = 10


def _escape_data(text: str) -> str:
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _escape_property(text: str) -> str:
    return _escape_data(text).replace(":", "%3A").replace(",", "%2C")


def parse_warnings(log: str) -> list[tuple[str, int, str]]:
    warnings = []
    for raw in log.splitlines():
        match = WARNING_RE.match(raw.strip())
        if match:
            path = match["path"]
            if os.path.isabs(path):
                path = os.path.relpath(path)
            warnings.append((path, int(match["line"]), match["message"]))
    return warnings


def _location(path: str, line: int) -> str:
    return f"{path}:{line}" if line > 0 else path


def _annotation(path: str, line: int, message: str) -> str:
    if not path.endswith(".py"):
        return f"::warning title=Docstring::{_escape_data(f'{path}: {message}')}"
    props = f"file={_escape_property(path)}"
    if line > 0:
        props += f",line={line}"
    return f"::warning {props},title=Docstring::{_escape_data(message)}"


def workflow_commands(warnings: list[tuple[str, int, str]]) -> list[str]:
    commands = [_annotation(*warning) for warning in warnings[:MAX_ANNOTATIONS]]
    if (extra := len(warnings) - MAX_ANNOTATIONS) > 0:
        commands.append(
            f"::notice title=Docstring::{extra} more docstring warning(s) in the job summary"
        )
    return commands


def summary(warnings: list[tuple[str, int, str]]) -> str:
    if not warnings:
        return "## Autodoc\n\nNo docstring warnings from `quartodoc build`.\n"
    lines = [
        "## Autodoc docstring warnings",
        "",
        f"`quartodoc build` reported {len(warnings)} docstring warning(s). "
        "The full log is in the `autodoc-log` artifact.",
        "",
        "| Location | Warning |",
        "| --- | --- |",
    ]
    for path, line, message in warnings:
        cell = message.replace("|", "\\|")
        lines.append(f"| `{_location(path, line)}` | {cell} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Captured `quartodoc build` output")
    args = parser.parse_args()

    warnings = parse_warnings(args.log.read_text(encoding="utf-8"))
    for command in workflow_commands(warnings):
        print(command)
    print(f"{len(warnings)} docstring warning(s)")

    if summary_path := os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(summary_path, "a", encoding="utf-8") as fh:
            fh.write(summary(warnings))


if __name__ == "__main__":
    main()
