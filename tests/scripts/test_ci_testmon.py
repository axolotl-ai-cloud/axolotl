"""CPU checks for CI test selection and Modal state transfer."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "cicd" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load("pytest_testmon")
transfer = load("modal_testmon")


def test_selection_and_coverage(monkeypatch, tmp_path):
    monkeypatch.setenv("TESTMON_SELECT", "true")
    monkeypatch.delenv("TESTMON_FULL", raising=False)
    monkeypatch.delenv("TESTMON_SUBPROCESSES", raising=False)
    args = ["tests/", "--cov=axolotl", "--cov-append", "--cov-report", "xml:out.xml"]
    selected = runner.testmon_args(args, tmp_path)
    assert selected == ["--testmon-forceselect", "tests/"]
    first = os.environ["TESTMON_DATAFILE"]
    runner.testmon_args(["other/"], tmp_path)
    assert os.environ["TESTMON_DATAFILE"] != first
    monkeypatch.setenv("TESTMON_FULL", "true")
    assert runner.testmon_args(args, tmp_path) == ["--no-testmon", *args]


def test_main_collects_all(monkeypatch, tmp_path):
    monkeypatch.delenv("TESTMON_SELECT", raising=False)
    monkeypatch.delenv("TESTMON_FULL", raising=False)
    monkeypatch.delenv("TESTMON_SUBPROCESSES", raising=False)
    assert runner.testmon_args(["tests/"], tmp_path)[0] == "--testmon-noselect"


def test_worker_changes_invalidate_environment(monkeypatch, tmp_path):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "src" / "worker.py"
    source.parent.mkdir()
    source.write_text("value = 1\n")
    subprocess.run(["git", "add", "src"], cwd=tmp_path, check=True)
    monkeypatch.setenv("TESTMON_SUBPROCESSES", "true")
    monkeypatch.delenv("TESTMON_FULL", raising=False)
    before = runner.testmon_args(["tests/"], tmp_path)
    source.write_text("value = 2\n")
    after = runner.testmon_args(["tests/"], tmp_path)
    assert before[1].startswith("--testmon-env=")
    assert before[1] != after[1]


def test_transfer_round_trip_and_failure(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    state = {
        "core.testmondata": b"sqlite",
        "core.testmondata-wal": b"wal",
        "core.scope.json": b"{}",
    }
    transfer.write_state(tmp_path / ".testmon", state)

    class Remote:
        def remote(self, baseline, full, select):
            assert baseline == state
            return 1, {"core.testmondata": b"updated"}

    with pytest.raises(RuntimeError, match="exit code 1"):
        transfer.run_local(Remote())
    assert (tmp_path / ".testmon/core.testmondata").read_bytes() == b"updated"
    assert not (tmp_path / ".testmon/core.testmondata-wal").exists()


@pytest.mark.parametrize("name", ["../escape", "/tmp/escape", "sub/file.testmondata"])
def test_transfer_rejects_paths(tmp_path, name):
    with pytest.raises(ValueError, match="Invalid testmon filename"):
        transfer.write_state(tmp_path, {name: b"data"})


def test_real_selection_and_full_override(tmp_path):
    pytest.importorskip("testmon")
    (tmp_path / "test_sample.py").write_text("def test_ok():\n    assert True\n")
    command = [
        sys.executable,
        str(ROOT / "cicd/pytest_testmon.py"),
        "-v",
        "test_sample.py",
    ]
    env = {
        **os.environ,
        "TESTMON_SELECT": "true",
        "TESTMON_FULL": "false",
        "TESTMON_SUBPROCESSES": "false",
    }
    first = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert first.returncode == 0, first.stdout + first.stderr
    assert "1 passed" in first.stdout
    second = subprocess.run(
        command, cwd=tmp_path, env=env, capture_output=True, text=True
    )
    assert second.returncode == 0, second.stdout + second.stderr
    assert "1 deselected" in second.stdout
    full = subprocess.run(
        command,
        cwd=tmp_path,
        env={**env, "TESTMON_FULL": "true"},
        capture_output=True,
        text=True,
    )
    assert full.returncode == 0, full.stdout + full.stderr
    assert "1 passed" in full.stdout
