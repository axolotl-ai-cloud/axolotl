"""Unit tests for cicd/deps_hash.py."""

import datetime as dt
import importlib.util
import shutil
import sys
from pathlib import Path

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


def test_marker_or_ruff_edits_keep_the_hash(tmp_path):
    root = _copy_inputs(tmp_path)
    before = mod.compute(root, {})
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            "[tool.pytest.ini_options]",
            "[tool.pytest.ini_options]\nxfail_strict = true",
        )
    )
    assert mod.compute(root, {}) == before


def test_dependency_edits_change_the_hash(tmp_path):
    root = _copy_inputs(tmp_path)
    before = mod.compute(root, {})
    pyproject = root / "pyproject.toml"
    pyproject.write_text(
        pyproject.read_text().replace(
            "dependencies = [", 'dependencies = [\n    "left-pad==1.0",', 1
        )
    )
    assert mod.compute(root, {}) != before


def test_dockerfile_extras_and_nightly_date_participate(tmp_path):
    root = _copy_inputs(tmp_path)
    base = mod.compute(root, {})
    assert mod.compute(root, {"AXOLOTL_EXTRAS": "vllm"}) != base
    (root / "cicd/Dockerfile-uv.jinja").write_text("FROM scratch\n")
    assert mod.compute(root, {}) != base
    day1 = mod.compute(root, {"NIGHTLY_BUILD": "true"}, today=dt.date(2026, 1, 1))
    day2 = mod.compute(root, {"NIGHTLY_BUILD": "true"}, today=dt.date(2026, 1, 2))
    assert day1 != day2
