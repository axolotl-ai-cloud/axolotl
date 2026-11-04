"""The CI Modal functions must import the runner's cicd/, not the image's checkout, and the
single- and multi-GPU apps must build one shared image."""

import importlib
import sys
from pathlib import Path

import pytest

modal = pytest.importorskip("modal")
pytest.importorskip("jinja2")

CICD = Path(__file__).resolve().parents[2] / "cicd"


def _import_fresh(monkeypatch, name):
    # import as a package module (relative imports) and fresh, so the image is rebuilt under spies
    monkeypatch.syspath_prepend(str(CICD.parent))
    for mod in [m for m in sys.modules if m == "cicd" or m.startswith("cicd.")]:
        monkeypatch.delitem(sys.modules, mod)
    return importlib.import_module(f"cicd.{name}")


@pytest.mark.parametrize("name", ["single_gpu", "multigpu", "image_build"])
def test_functions_run_from_the_mount_root(monkeypatch, name):
    workdirs = []
    workdir = modal.Image.workdir

    def spy(self, path):
        workdirs.append(str(path))
        return workdir(self, path)

    monkeypatch.setattr(modal.Image, "workdir", spy)
    _import_fresh(monkeypatch, name)

    assert workdirs == [modal.mount.ROOT_DIR.as_posix()]


def test_single_and_multi_gpu_apps_build_the_same_image(monkeypatch):
    builds = {}
    from_dockerfile = modal.Image.from_dockerfile

    def spy(path, **kwargs):
        builds[current] = (Path(path).read_text(encoding="utf-8"), kwargs)
        return from_dockerfile(path, **kwargs)

    monkeypatch.setattr(modal.Image, "from_dockerfile", staticmethod(spy))
    for current in ("single_gpu", "multigpu"):
        _import_fresh(monkeypatch, current)

    assert set(builds) == {"single_gpu", "multigpu"}
    assert builds["single_gpu"] == builds["multigpu"]
    assert builds["single_gpu"][1] == {}


PINNED = "axolotlai/axolotl-base-uv@sha256:" + "0" * 64


@pytest.mark.parametrize(
    ("base_image", "from_line"),
    [(PINNED, f"FROM {PINNED}"), (None, "FROM axolotlai/axolotl-base-uv:tag1")],
)
def test_image_builds_from_the_pinned_base(monkeypatch, base_image, from_line):
    # a lone first-line FROM is what lets builder 2026.10 cache the base pull as its own layer
    first_lines = []
    from_dockerfile = modal.Image.from_dockerfile

    def spy(path, **kwargs):
        first_lines.append(Path(path).read_text(encoding="utf-8").split("\n", 1)[0])
        return from_dockerfile(path, **kwargs)

    monkeypatch.setattr(modal.Image, "from_dockerfile", staticmethod(spy))
    monkeypatch.setenv("BASE_TAG", "tag1")
    if base_image:
        monkeypatch.setenv("BASE_IMAGE", base_image)
    else:
        monkeypatch.delenv("BASE_IMAGE", raising=False)
    _import_fresh(monkeypatch, "single_gpu")

    assert first_lines == [from_line]
