"""Layout checks for agent skills.

Skills are authored once under the vendor-neutral `.agents/skills/` (read natively
by Codex/Gemini) and exposed to Claude Code via a single `.claude/skills` symlink.
These tests enforce the agentskills.io portable core — a `SKILL.md` per skill whose
frontmatter carries a spec-valid `name` (matching the directory) and `description`
— plus that the Claude symlink resolves to the canonical directory, so the two
locations can never drift.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
AGENTS_SKILLS = REPO_ROOT / ".agents" / "skills"
CLAUDE_SKILLS = REPO_ROOT / ".claude" / "skills"

NAME_RE = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")


def _skill_dirs() -> list[str]:
    # every subdir is a skill; a missing SKILL.md must fail a test, not be skipped
    if not AGENTS_SKILLS.is_dir():
        return []
    return sorted(p.name for p in AGENTS_SKILLS.iterdir() if p.is_dir())


def _frontmatter(text: str) -> dict[str, str]:
    assert text.startswith("---\n"), "SKILL.md must start with YAML frontmatter"
    _, block, _ = text.split("---\n", 2)
    fields: dict[str, str] = {}
    for line in block.splitlines():
        if ":" in line and not line.startswith((" ", "\t")):
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip().strip('"').strip("'")
    return fields


def test_at_least_one_skill() -> None:
    assert _skill_dirs(), "no skills found under .agents/skills"


def test_claude_skills_symlink() -> None:
    assert CLAUDE_SKILLS.is_symlink(), ".claude/skills must be a symlink"
    # resolve() does not require the target to exist, so check the link is live too
    assert CLAUDE_SKILLS.exists(), ".claude/skills symlink is dangling"
    assert CLAUDE_SKILLS.resolve() == AGENTS_SKILLS.resolve(), (
        ".claude/skills must point at .agents/skills (do not duplicate skill content)"
    )


@pytest.mark.parametrize("name", _skill_dirs())
def test_skill_frontmatter(name: str) -> None:
    skill_md = AGENTS_SKILLS / name / "SKILL.md"
    assert skill_md.is_file(), f"{name}: missing SKILL.md"
    fields = _frontmatter(skill_md.read_text("utf-8"))

    assert fields.get("name") == name, (
        f"{name}: frontmatter 'name' must match the directory name"
    )
    assert NAME_RE.match(name), (
        f"{name}: name must be lowercase-hyphen (agentskills.io)"
    )
    assert len(name) <= 64, f"{name}: name exceeds 64 chars"

    description = fields.get("description", "")
    assert description, f"{name}: frontmatter missing 'description'"
    assert len(description) <= 1024, f"{name}: description exceeds 1024 chars"


@pytest.mark.parametrize("name", _skill_dirs())
def test_skill_scripts_compile(name: str) -> None:
    # a syntax error in a bundled script must fail the suite, not ship green;
    # compile() needs no third-party imports (e.g. liger-kernel)
    for script in sorted((AGENTS_SKILLS / name).rglob("*.py")):
        compile(script.read_text("utf-8"), str(script), "exec")
