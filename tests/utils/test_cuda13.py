"""Tests for CUDA 13 LD_LIBRARY_PATH helpers."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

from axolotl.utils.cuda13 import cu13_library_path, prepend_cu13_ld_library_path


@pytest.fixture(autouse=True)
def clear_nvidia_cu13_modules():
    for module_name in ("nvidia.cu13", "nvidia"):
        sys.modules.pop(module_name, None)
    yield
    for module_name in ("nvidia.cu13", "nvidia"):
        sys.modules.pop(module_name, None)


def _make_fake_cu13_package(root: Path) -> Path:
    package_root = root / "nvidia" / "cu13"
    package_root.mkdir(parents=True)
    (root / "nvidia" / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "lib").mkdir()
    return package_root


def test_prepend_cu13_ld_library_path_when_package_exists(monkeypatch, tmp_path):
    package_root = _make_fake_cu13_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    expected_lib = str(package_root / "lib")

    assert cu13_library_path() == expected_lib
    assert (
        prepend_cu13_ld_library_path("/opt/custom/lib:/usr/lib")
        == f"{expected_lib}:/opt/custom/lib:/usr/lib"
    )


def test_prepend_cu13_ld_library_path_is_idempotent(monkeypatch, tmp_path):
    package_root = _make_fake_cu13_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    expected_lib = str(package_root / "lib")

    assert (
        prepend_cu13_ld_library_path(f"{expected_lib}:/opt/custom/lib:{expected_lib}")
        == f"{expected_lib}:/opt/custom/lib"
    )


def test_prepend_cu13_ld_library_path_is_noop_without_package(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "nvidia.cu13":
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    assert cu13_library_path() is None
    assert prepend_cu13_ld_library_path("/opt/custom/lib") == "/opt/custom/lib"
    assert prepend_cu13_ld_library_path(None) == ""


def test_cuda13_env_sh_prepend_path_when_package_exists(tmp_path):
    package_root = _make_fake_cu13_package(tmp_path)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{tmp_path}:src"
    env["LD_LIBRARY_PATH"] = ""

    result = subprocess.run(
        [
            "bash",
            "-lc",
            'source scripts/cuda13_env.sh; printf %s "$LD_LIBRARY_PATH"',
        ],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )

    path_parts = result.stdout.split(":")
    assert path_parts[0] == str(package_root / "lib")
    assert str(package_root / "lib") in path_parts
