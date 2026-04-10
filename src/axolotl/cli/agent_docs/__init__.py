"""Bundled agent documentation for axolotl.

These docs are optimized for consumption by AI coding agents.
The source of truth is docs/agents/*.md and AGENTS.md in the repo root.
This module resolves those paths at runtime — no files are duplicated
into the package.

For pip-only installs (no repo checkout), run `axolotl fetch docs` first
to download the docs locally.
"""

from pathlib import Path

# Topic name -> (filename in docs/agents/, fallback filename for AGENTS.md)
TOPICS = {
    "overview": "AGENTS.md",
    "sft": "docs/agents/sft.md",
    "grpo": "docs/agents/grpo.md",
    "preference_tuning": "docs/agents/preference_tuning.md",
    "reward_modelling": "docs/agents/reward_modelling.md",
    "pretraining": "docs/agents/pretraining.md",
    "model_architectures": "docs/agents/model_architectures.md",
    "new_model_support": "docs/agents/new_model_support.md",
}


def _find_repo_root() -> Path | None:
    """Walk up from this file to find the repo root (contains AGENTS.md)."""
    # In an editable install or repo checkout, walk up from
    # src/axolotl/cli/agent_docs/ to find the repo root
    current = Path(__file__).resolve().parent
    while current != current.parent:
        if (current / "AGENTS.md").exists() and (current / "docs" / "agents").is_dir():
            return current
        current = current.parent
    return None


def _find_docs_dir() -> Path | None:
    """Find a fetched docs directory (from `axolotl fetch docs`)."""
    # axolotl fetch docs --dest defaults to ./docs/ in cwd
    cwd_docs = Path.cwd() / "docs" / "agents"
    if cwd_docs.is_dir():
        return Path.cwd()
    return None


def _resolve_path(topic: str) -> Path:
    """Resolve a topic name to the actual file path."""
    if topic not in TOPICS:
        available = ", ".join(sorted(TOPICS.keys()))
        raise FileNotFoundError(f"Unknown topic: {topic!r}. Available: {available}")

    relative_path = TOPICS[topic]

    # Try repo root first (editable install / repo checkout)
    repo_root = _find_repo_root()
    if repo_root:
        candidate = repo_root / relative_path
        if candidate.exists():
            return candidate

    # Try cwd (fetched docs via `axolotl fetch docs`)
    docs_root = _find_docs_dir()
    if docs_root:
        candidate = docs_root / relative_path
        if candidate.exists():
            return candidate

    # Also check cwd directly for AGENTS.md
    if topic == "overview":
        cwd_agents = Path.cwd() / "AGENTS.md"
        if cwd_agents.exists():
            return cwd_agents

    raise FileNotFoundError(
        f"Could not find {relative_path!r}. "
        f"If you installed axolotl via pip, run `axolotl fetch docs` first "
        f"to download the documentation."
    )


def get_doc(topic: str = "overview") -> str:
    """Return the content of an agent doc by topic name.

    Args:
        topic: One of the keys in TOPICS, or "overview" (default).

    Returns:
        The markdown content of the doc.

    Raises:
        FileNotFoundError: If the topic can't be found.
    """
    return _resolve_path(topic).read_text()


def list_topics() -> dict[str, str]:
    """Return a dict of topic name -> first line (title) of each doc."""
    result = {}
    for topic in sorted(TOPICS.keys()):
        try:
            path = _resolve_path(topic)
            first_line = path.read_text().split("\n", 1)[0].lstrip("# ").strip()
            result[topic] = first_line
        except FileNotFoundError:
            result[topic] = "(not found — run `axolotl fetch docs`)"
    return result
