"""Axolotl ASCII logo utils."""

import os
import subprocess  # nosec B404
from pathlib import Path

import axolotl
from axolotl.utils.distributed import is_main_process

GITHUB_URL = "https://github.com/axolotl-ai-cloud/axolotl"

_AXOLOTL_LOGO_HEAD = "\n".join(
    [
        " ┌─┐┌─┐    ┌─┐┌─┐",
        " │ ││ │    │ ││ │   █████   ██   ██   █████   ██        █████   ███████  ██",
        " │ └┘ └────┘ └┘ │  ██   ██   ██ ██   ██   ██  ██       ██   ██    ███    ██",
        " │              │  ███████    ███    ██   ██  ██       ██   ██    ███    ██",
        " │  ◉        ◉  │  ██   ██   ██ ██   ██   ██  ██       ██   ██    ███    ██",
        " └──────────────┘  ██   ██  ██   ██   █████   ███████   █████     ███    ███████",
    ]
)

_BARS_TOP = " ▄▄▄▄▄▄▄▄▄▄  ▄▄▄"
_BARS_BOTTOM = " ▄▄▄  ▄▄▄▄▄▄▄▄▄▄"

_LETTERS_START_COL = len(" └──────────────┘  ")
_LETTERS_END_COL = len(
    " └──────────────┘  ██   ██  ██   ██   █████   ███████   █████     ███    ███████"
)

HAS_PRINTED_LOGO = False


def _get_git_info() -> tuple[str | None, str | None]:
    """Return (short_sha, commit_date) or (None, None) when not a git checkout."""
    repo_root = Path(axolotl.__file__).resolve().parents[2]
    if not (repo_root / ".git").exists():
        return None, None
    try:
        sha = subprocess.check_output(  # nosec B603 B607
            ["git", "-C", str(repo_root), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        date = subprocess.check_output(  # nosec B603 B607
            ["git", "-C", str(repo_root), "log", "-1", "--format=%cs"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return sha or None, date or None
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None, None


def _centered_under_letters(text: str, *, bars: str = "") -> str:
    """Pad `text` so it sits centered under the AXOLOTL letters, optionally after bars."""
    available = _LETTERS_END_COL - _LETTERS_START_COL
    pad = max(0, (available - len(text)) // 2)
    text_col = _LETTERS_START_COL + pad
    gap = max(1, text_col - len(bars))
    return bars + " " * gap + text


def _build_logo() -> str:
    sha, date = _get_git_info()
    parts = [f"v{axolotl.__version__}"]
    if sha:
        parts.append(sha)
    if date:
        parts.append(date)
    build = " · ".join(parts)
    return "\n".join(
        [
            "",
            _AXOLOTL_LOGO_HEAD,
            _BARS_TOP,
            _centered_under_letters(GITHUB_URL, bars=_BARS_BOTTOM),
            _centered_under_letters(build),
            "",
        ]
    )


def print_axolotl_text_art():
    """Prints axolotl ASCII art."""

    global HAS_PRINTED_LOGO
    if HAS_PRINTED_LOGO or os.environ.get("AXOLOTL_BANNER_PRINTED"):
        return
    if is_main_process():
        HAS_PRINTED_LOGO = True
        os.environ["AXOLOTL_BANNER_PRINTED"] = "1"
        print(_build_logo())
