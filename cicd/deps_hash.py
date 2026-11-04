#!/usr/bin/env python
"""Hash the inputs that change the CI Modal image, so edits elsewhere reuse it.

Only the dependency-bearing parts of pyproject.toml count: a pytest marker or a ruff
rule must not force a rebuild of a 150 s image. The hash also picks IMAGE_REF, the git
ref the image installs dependencies from: main whenever the branch's dependency inputs
match main's, so every such PR shares one image, and the branch's own ref otherwise.
The container checks out the exact commit under test at start-up either way. BASE_IMAGE
pins the base image tag to the digest it points at now.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import http.client
import json
import os
import re
import subprocess  # nosec B404
import sys
import tomllib
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DOCKERFILE = "Dockerfile-uv.jinja"
INSTALL_SCRIPTS = ("scripts/cutcrossentropy_install.py",)
MAIN_REF = "refs/heads/main"
BASE_REPO = "axolotlai/axolotl-base-uv"
MANIFEST_TYPES = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


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


def _read(repo_root: Path, rel: str, rev: str | None) -> bytes:
    if rev is None:
        return (repo_root / rel).read_bytes()
    return subprocess.check_output(  # nosec B603 B607
        ["git", "show", f"{rev}:{rel}"], cwd=repo_root, stderr=subprocess.DEVNULL
    )


def compute(
    repo_root: Path,
    env: dict[str, str],
    today: dt.date | None = None,
    rev: str | None = None,
) -> str:
    """Hash of the image inputs at the working tree, or at ``rev`` when given."""
    digest = hashlib.sha256()
    sections = dependency_sections(
        tomllib.loads(_read(repo_root, "pyproject.toml", rev).decode())
    )
    digest.update(json.dumps(sections, sort_keys=True).encode())
    dockerfiles = [DEFAULT_DOCKERFILE]
    if env.get("E2E_DOCKERFILE") and env["E2E_DOCKERFILE"] != DEFAULT_DOCKERFILE:
        dockerfiles.append(env["E2E_DOCKERFILE"])
    for rel in [*INSTALL_SCRIPTS, *(f"cicd/{name}" for name in dockerfiles)]:
        digest.update(rel.encode())
        digest.update(_read(repo_root, rel, rev))
    digest.update(
        f"extras={env.get('AXOLOTL_EXTRAS', '')} args={env.get('AXOLOTL_ARGS', '')} "
        f"base={env.get('BASE_TAG', '')}".encode()
    )
    day = today or dt.datetime.now(dt.UTC).date()
    if env.get("NIGHTLY_BUILD") == "true":
        digest.update(day.isoformat().encode())
    else:
        # unpinned ranges and the moving base tag drift; bound a shared image's age to a week
        digest.update("{0}-W{1:02d}".format(*day.isocalendar()[:2]).encode())
    return digest.hexdigest()[:16]


def image_ref(
    repo_root: Path, env: dict[str, str], deps_hash: str, today: dt.date | None = None
) -> str:
    """main when this tree's dependency inputs match main's, else the ref under test."""
    github_ref = env.get("GITHUB_REF", MAIN_REF)
    if github_ref == MAIN_REF:
        return MAIN_REF
    subprocess.run(  # nosec B603 B607
        [
            "git",
            "fetch",
            "-q",
            "--depth=1",
            "origin",
            f"+{MAIN_REF}:refs/remotes/origin/main",
        ],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    try:
        main_hash = compute(repo_root, env, today, rev="origin/main")
    except (
        subprocess.CalledProcessError,
        OSError,
        tomllib.TOMLDecodeError,
        UnicodeDecodeError,
    ):
        return github_ref
    return MAIN_REF if main_hash == deps_hash else github_ref


def base_image(tag: str) -> str:
    """The base image at the digest its tag points to now, or the tag if the lookup fails.

    Builder 2026.10 caches a Dockerfile's FROM line as its own image, so a tag that base.yml
    re-pushes would otherwise never be pulled again.
    """
    reference = f"{BASE_REPO}:{tag}"
    try:
        with urllib.request.urlopen(  # nosec B310
            "https://auth.docker.io/token?service=registry.docker.io"
            f"&scope=repository:{BASE_REPO}:pull",
            timeout=30,
        ) as resp:
            token = json.load(resp)["token"]
        manifest = urllib.request.Request(
            f"https://registry-1.docker.io/v2/{BASE_REPO}/manifests/"
            + urllib.parse.quote(tag, safe=""),
            method="HEAD",
            headers={"Authorization": f"Bearer {token}", "Accept": MANIFEST_TYPES},
        )
        with urllib.request.urlopen(manifest, timeout=30) as resp:  # nosec B310
            digest = resp.headers.get("Docker-Content-Digest", "")
    except (OSError, http.client.HTTPException, KeyError, TypeError, ValueError) as exc:
        print(f"WARNING: {reference} digest lookup failed: {exc}", file=sys.stderr)
        return reference
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
        print(f"WARNING: {reference} has no usable digest: {digest!r}", file=sys.stderr)
        return reference
    return f"{BASE_REPO}@{digest}"


def main() -> int:
    env = dict(os.environ)
    deps_hash = compute(REPO_ROOT, env)
    outputs = {
        "DEPS_HASH": deps_hash,
        "IMAGE_REF": image_ref(REPO_ROOT, env, deps_hash),
    }
    if env.get("BASE_TAG"):
        outputs["BASE_IMAGE"] = base_image(env["BASE_TAG"])
    lines = "".join(f"{key}={value}\n" for key, value in outputs.items())
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as fh:
            fh.write(lines)
    print(lines, end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
