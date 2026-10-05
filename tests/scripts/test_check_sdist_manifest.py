"""Unit tests for scripts/check_sdist_manifest.py."""

import importlib.util
import io
import tarfile
import zipfile
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_sdist_manifest.py"
_spec = importlib.util.spec_from_file_location("check_sdist_manifest", SCRIPT)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mod)


def test_compare_reports_missing_and_extra():
    assert mod.compare({"a.py", "b/c.yaml"}, {"a.py", "d.py"}) == (
        ["b/c.yaml"],
        ["d.py"],
    )


def test_compare_clean():
    assert mod.compare({"a.py"}, {"a.py"}) == ([], [])


def test_wheel_package_files_strips_prefix(tmp_path):
    wheel = tmp_path / "axolotl-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as zf:
        zf.writestr("axolotl/x.py", "")
        zf.writestr("axolotl/sub/y.yaml", "")
        zf.writestr("axolotl-1.0.dist-info/RECORD", "")
    assert mod.wheel_package_files(wheel) == {"x.py", "sub/y.yaml"}


def test_tracked_package_files_filters(monkeypatch):
    out = (
        "src/axolotl/a.py\n"
        "src/axolotl/__pycache__/a.pyc\n"
        "src/axolotl/x/tests/t.py\n"
        "src/axolotl/b.yaml\n"
    )
    monkeypatch.setattr(mod.subprocess, "check_output", lambda *a, **k: out)
    assert mod.tracked_package_files(Path(".")) == {"a.py", "b.yaml"}


def test_report_fails_only_on_missing(capsys):
    assert mod.report(["gone.yaml"], [], 2, 1) == 1
    assert "missing=1" in capsys.readouterr().out
    assert mod.report([], ["untracked.py"], 1, 2) == 0


def test_check_sdist_contents(tmp_path):
    sdist = tmp_path / "pkg-1.tar.gz"
    with tarfile.open(sdist, "w:gz") as tf:
        info = tarfile.TarInfo("pkg-1/AGENTS.md")
        tf.addfile(info, io.BytesIO(b""))
    problems = mod.check_sdist_contents(sdist)
    assert problems == [
        "pkg-1/README.md",
        "pkg-1/LICENSE",
        "pkg-1/VERSION",
        "pkg-1/docs/agents/*.md",
    ]
