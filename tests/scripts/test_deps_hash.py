"""Unit tests for cicd/deps_hash.py."""

import datetime as dt
import email.message
import importlib.util
import io
import shutil
import subprocess  # nosec B404
import sys
import urllib.error
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "cicd" / "deps_hash.py"
_spec = importlib.util.spec_from_file_location("deps_hash", SCRIPT)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
sys.modules["deps_hash"] = mod
_spec.loader.exec_module(mod)

REPO = SCRIPT.parents[1]


def _copy_inputs(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    for rel in (
        "pyproject.toml",
        "scripts/cutcrossentropy_install.py",
        "cicd/Dockerfile-uv.jinja",
    ):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(REPO / rel, root / rel)
    return root


TODAY = dt.date(2026, 10, 4)


def test_marker_or_ruff_edits_keep_the_hash(tmp_path):
    root = _copy_inputs(tmp_path)
    before = mod.compute(root, {}, today=TODAY)
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            "[tool.pytest.ini_options]",
            "[tool.pytest.ini_options]\nxfail_strict = true",
        )
    )
    assert mod.compute(root, {}, today=TODAY) == before


def test_dependency_edits_change_the_hash(tmp_path):
    root = _copy_inputs(tmp_path)
    before = mod.compute(root, {}, today=TODAY)
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            "dependencies = [", 'dependencies = [\n    "left-pad==1.0",', 1
        )
    )
    assert mod.compute(root, {}, today=TODAY) != before


def test_dockerfile_extras_and_nightly_date_participate(tmp_path):
    root = _copy_inputs(tmp_path)
    base = mod.compute(root, {}, today=TODAY)
    assert mod.compute(root, {"AXOLOTL_EXTRAS": "vllm"}, today=TODAY) != base
    (root / "cicd/Dockerfile-uv.jinja").write_text("FROM scratch\n")
    assert mod.compute(root, {}, today=TODAY) != base
    day1 = mod.compute(root, {"NIGHTLY_BUILD": "true"}, today=dt.date(2026, 1, 1))
    day2 = mod.compute(root, {"NIGHTLY_BUILD": "true"}, today=dt.date(2026, 1, 2))
    assert day1 != day2


def test_shared_image_expires_weekly(tmp_path):
    root = _copy_inputs(tmp_path)
    same_week = mod.compute(root, {}, today=dt.date(2026, 10, 5)) == mod.compute(
        root, {}, today=dt.date(2026, 10, 9)
    )
    next_week = mod.compute(root, {}, today=dt.date(2026, 10, 5)) == mod.compute(
        root, {}, today=dt.date(2026, 10, 12)
    )
    assert same_week and not next_week


def _git(repo: Path, *args: str) -> None:
    subprocess.run(  # nosec B603 B607
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
            "HOME": str(repo),
            "PATH": "/usr/bin:/bin",
        },
    )


def _git_repo_with_main(tmp_path: Path) -> Path:
    root = _copy_inputs(tmp_path)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "base")
    # origin/main is what image_ref compares against; point it at the local main
    _git(root, "update-ref", "refs/remotes/origin/main", "HEAD")
    _git(root, "checkout", "-qb", "feature")
    return root


def test_image_ref_is_main_when_dependencies_match(tmp_path):
    root = _git_repo_with_main(tmp_path)
    env = {"GITHUB_REF": "refs/pull/1/merge"}
    pyproject = root / "pyproject.toml"
    pyproject.write_text(pyproject.read_text() + "\n# docs only\n")
    deps_hash = mod.compute(root, env, today=TODAY)
    assert mod.image_ref(root, env, deps_hash, today=TODAY) == "refs/heads/main"


def test_image_ref_is_the_branch_when_dependencies_differ(tmp_path):
    root = _git_repo_with_main(tmp_path)
    env = {"GITHUB_REF": "refs/pull/1/merge"}
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            "dependencies = [", 'dependencies = [\n    "left-pad==1.0",', 1
        )
    )
    deps_hash = mod.compute(root, env, today=TODAY)
    assert mod.image_ref(root, env, deps_hash, today=TODAY) == "refs/pull/1/merge"


def test_image_ref_falls_back_to_the_branch_without_main(tmp_path):
    root = _copy_inputs(tmp_path)
    env = {"GITHUB_REF": "refs/pull/1/merge"}
    assert mod.image_ref(root, env, "abc", today=TODAY) == "refs/pull/1/merge"


DIGEST = "sha256:" + "ab" * 32


class _Response(io.BytesIO):
    def __init__(self, body=b"", headers=None):
        super().__init__(body)
        self.headers = email.message.Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value


def _fake_registry(monkeypatch, digest):
    requests = []

    def urlopen(request, timeout):
        requests.append(request)
        if len(requests) == 1:
            return _Response(b'{"token": "t"}')
        return _Response(headers={"Docker-Content-Digest": digest} if digest else {})

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen)
    return requests


def test_base_image_is_pinned_to_the_tag_digest(monkeypatch):
    requests = _fake_registry(monkeypatch, DIGEST)
    tag = "main-base-py3.12-cu130-2.13.0"
    assert mod.base_image(tag) == f"axolotlai/axolotl-base-uv@{DIGEST}"
    manifest = requests[1]
    assert manifest.get_method() == "HEAD"
    assert manifest.full_url.endswith(f"/axolotl-base-uv/manifests/{tag}")
    assert manifest.get_header("Authorization") == "Bearer t"


@pytest.mark.parametrize("digest", [None, "sha256:not-a-digest"])
def test_base_image_without_a_usable_digest_keeps_the_tag(monkeypatch, digest):
    _fake_registry(monkeypatch, digest)
    assert mod.base_image("tag1") == "axolotlai/axolotl-base-uv:tag1"


def test_base_image_keeps_the_tag_when_the_registry_is_unreachable(monkeypatch):
    def urlopen(request, timeout):
        raise urllib.error.URLError("offline")

    monkeypatch.setattr(mod.urllib.request, "urlopen", urlopen)
    assert mod.base_image("tag1") == "axolotlai/axolotl-base-uv:tag1"
