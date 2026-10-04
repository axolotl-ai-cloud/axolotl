#!/usr/bin/env python
"""Hash the inputs that change the CI Modal image, so edits elsewhere reuse it.

Only the dependency-bearing parts of pyproject.toml count: a pytest marker or a ruff
rule must not force a rebuild of a 150 s image.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCKERFILE = "Dockerfile-uv.jinja"
INSTALL_SCRIPTS = ("scripts/cutcrossentropy_install.py",)


def dependency_sections(pyproject: dict) -> dict:
    project = pyproject.get("project", {})
    return {
        "build-system": pyproject.get("build-system"),
        "requires-python": project.get("requires-python"),
        "dependencies": project.get("dependencies"),
        "optional-dependencies": project.get("optional-dependencies"),
        "dependency-groups": pyproject.get("dependency-groups"),
        "tool.uv": pyproject.get("tool", {}).get("uv"),
        "tool.setuptools": pyproject.get("tool", {}).get("setuptools"),
    }


def compute(repo_root: Path, env: dict[str, str], today: dt.date | None = None) -> str:
    digest = hashlib.sha256()
    with open(repo_root / "pyproject.toml", "rb") as fh:
        sections = dependency_sections(tomllib.load(fh))
    digest.update(json.dumps(sections, sort_keys=True).encode())
    dockerfiles = [DEFAULT_DOCKERFILE]
    if env.get("E2E_DOCKERFILE") and env["E2E_DOCKERFILE"] != DEFAULT_DOCKERFILE:
        dockerfiles.append(env["E2E_DOCKERFILE"])
    for rel in [*INSTALL_SCRIPTS, *(f"cicd/{name}" for name in dockerfiles)]:
        digest.update(rel.encode())
        digest.update((repo_root / rel).read_bytes())
    digest.update(
        f"extras={env.get('AXOLOTL_EXTRAS', '')} args={env.get('AXOLOTL_ARGS', '')} "
        f"base={env.get('BASE_TAG', '')}".encode()
    )
    if env.get("NIGHTLY_BUILD") == "true":
        digest.update((today or dt.datetime.now(dt.UTC).date()).isoformat().encode())
    return digest.hexdigest()[:16]


def main() -> int:
    deps_hash = compute(REPO_ROOT, dict(os.environ))
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as fh:
            fh.write(f"DEPS_HASH={deps_hash}\n")
    print(f"DEPS_HASH={deps_hash}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
