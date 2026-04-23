"""Guard the attn_implementation source-of-truth invariant.

`cfg.attn_implementation` is the single source of truth for the attention
backend on the validated config. Legacy boolean flags (`flash_attention`,
`sdp_attention`, `xformers_attention`, `flex_attention`, `sage_attention`,
`s2_attention`, `eager_attention`) are input-only deprecated aliases — they
are stripped from `data` by `normalize_attn_implementation` and must never be
read downstream.

This test greps `src/` and fails if it finds a `cfg.<legacy>_attention` read.
If you're here because this test failed, migrate the read site to
`cfg.attn_implementation` or one of the `attn_supports_packing /
attn_uses_flash_lib / attn_needs_dtype_cast` computed capability flags.
"""

from __future__ import annotations

import re
from pathlib import Path

LEGACY_FLAGS = (
    "flash_attention",
    "sdp_attention",
    "xformers_attention",
    "flex_attention",
    "sage_attention",
    "s2_attention",
    "eager_attention",
)

# The normalizer is allowed to read the legacy keys (that's its job).
# lm_eval/cli.py is a raw-YAML entry point (bypasses AxolotlInputConfig) that
# honors both forms during the deprecation period — when we remove the legacy
# flags entirely, drop this allowlist entry and the BC branch in that file.
ALLOWED_FILES = {
    Path("src/axolotl/utils/schemas/config.py"),
    Path("src/axolotl/integrations/lm_eval/cli.py"),
}

# `cfg.<flag>`, `self.cfg.<flag>`, `data.get("<flag>")`, `data["<flag>"]`
_PATTERNS = [re.compile(rf"\bcfg\.{flag}\b") for flag in LEGACY_FLAGS] + [
    re.compile(rf'\bdata\.get\("{flag}"\)') for flag in LEGACY_FLAGS
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def test_no_legacy_attn_reads_in_src():
    root = _repo_root()
    src = root / "src"
    offenders: list[str] = []

    for py_file in src.rglob("*.py"):
        rel = py_file.relative_to(root)
        if rel in ALLOWED_FILES:
            continue
        text = py_file.read_text(encoding="utf-8")
        for pattern in _PATTERNS:
            for match in pattern.finditer(text):
                # Line number for the user's convenience.
                line_no = text.count("\n", 0, match.start()) + 1
                offenders.append(f"{rel}:{line_no}  {match.group(0)}")

    assert not offenders, (
        "Found legacy attention-flag reads in src/. Migrate to "
        "`cfg.attn_implementation` / capability flags:\n  "
        + "\n  ".join(sorted(offenders))
    )
