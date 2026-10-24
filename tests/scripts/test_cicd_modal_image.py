"""The CI Modal functions must import the runner's cicd/, not the image's checkout."""

import importlib
import sys
from pathlib import Path

import pytest

modal = pytest.importorskip("modal")
pytest.importorskip("jinja2")

CICD = Path(__file__).resolve().parents[2] / "cicd"


@pytest.mark.parametrize("name", ["single_gpu", "multigpu"])
def test_functions_run_from_the_mount_root(monkeypatch, name):
    workdirs = []
    workdir = modal.Image.workdir

    def spy(self, path):
        workdirs.append(str(path))
        return workdir(self, path)

    monkeypatch.setattr(modal.Image, "workdir", spy)
    # import as a package module (relative imports) and fresh, so the image is rebuilt under the spy
    monkeypatch.syspath_prepend(str(CICD.parent))
    for mod in [m for m in sys.modules if m == "cicd" or m.startswith("cicd.")]:
        monkeypatch.delitem(sys.modules, mod)
    importlib.import_module(f"cicd.{name}")

    assert workdirs == [modal.mount.ROOT_DIR.as_posix()]
