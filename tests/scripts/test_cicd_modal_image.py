"""The CI Modal functions must import the runner's cicd/, not the image's checkout."""

import importlib.util
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
    spec = importlib.util.spec_from_file_location(f"_cicd_{name}", CICD / f"{name}.py")
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(importlib.util.module_from_spec(spec))

    assert workdirs == [modal.mount.ROOT_DIR.as_posix()]
