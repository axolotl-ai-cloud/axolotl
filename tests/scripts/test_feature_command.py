"""Exercise whole-suite selection with real worker processes on CPU."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def suite(tmp_path):
    for name in ("cicd", "tests", "src/liger", "src/cce"):
        (tmp_path / name).mkdir(parents=True)
    (tmp_path / "tests/test_worker.py").write_text(
        "import subprocess, sys\n"
        "def test_worker():\n"
        '    subprocess.run([sys.executable, "src/worker.py"], check=True)\n'
        "def test_other():\n"
        "    assert True\n"
    )
    (tmp_path / "src/worker.py").write_text("assert True\n")
    (tmp_path / "src/liger/plugin.py").write_text("x = 1\n")
    (tmp_path / "src/cce/plugin.py").write_text("x = 1\n")
    manifest = {
        "features": {"liger": ["src/liger/"], "cce": ["src/cce/"]},
        "distributed": {
            "demo": {
                "paths": ["tests/"],
                "exclude": [],
                "features": ["cce"],
                "audited_tests": ["tests/test_worker.py"],
            }
        },
    }
    (tmp_path / "cicd/feature_suites.json").write_text(json.dumps(manifest))
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    env = {**os.environ, "TESTMON_SELECT": "true", "TESTMON_FULL": "false"}

    def run(**changes):
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "cicd/feature_command.py"),
                "demo",
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "tests/",
            ],
            cwd=tmp_path,
            env=env | changes,
            text=True,
            capture_output=True,
        )

    return tmp_path, run


def test_worker_change_runs_entire_suite_and_failure_retries(suite):
    root, run = suite
    first = run()
    assert first.returncode == 0, first.stdout + first.stderr
    assert "2 passed" in first.stdout
    assert not list((root / ".testmon").glob("*.testmondata"))
    assert "skip:" in run().stdout
    (root / "src/worker.py").write_text("assert False\n")
    failure = run()
    assert failure.returncode == 1, failure.stdout + failure.stderr
    assert "1 failed, 1 passed" in failure.stdout
    assert "previous suite did not pass" in run().stdout
    (root / "src/worker.py").write_text("assert True\n")
    assert "2 passed" in run().stdout


def test_feature_skip_full_override_and_new_tests(suite):
    root, run = suite
    assert run().returncode == 0
    (root / "src/liger/plugin.py").write_text("x = 2\n")
    assert "only other explicitly isolated features changed" in run().stdout
    assert "2 passed" in run(TESTMON_FULL="true").stdout
    assert "2 passed" in run(TESTMON_SELECT="false").stdout
    (root / "tests/test_added.py").write_text("def test_new(): assert True\n")
    subprocess.run(["git", "add", "tests/test_added.py"], cwd=root, check=True)
    assert "3 passed" in run().stdout
    (root / "src/liger/plugin.py").write_text("x = 3\n")
    assert "3 passed" in run().stdout


@pytest.mark.parametrize(
    "change", ["metadata", "environment", "shared", "config", "cce"]
)
def test_invalid_or_relevant_baseline_runs_full(suite, change):
    root, run = suite
    assert run().returncode == 0
    overrides = {}
    if change == "metadata":
        next((root / ".testmon").glob("*.scope.json")).write_text("broken")
    elif change == "environment":
        overrides["N_GPUS"] = "4"
    else:
        path = {
            "shared": "src/worker.py",
            "config": "cicd/feature_suites.json",
            "cce": "src/cce/plugin.py",
        }[change]
        with (root / path).open("a") as stream:
            stream.write("\n")
    result = run(**overrides)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 passed" in result.stdout


def test_workflows_publish_separate_baselines_and_refresh_explicitly():
    def workflow(name):
        return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())

    keys = []
    for file, job in [
        ("multi-gpu-e2e-jobs.yml", "test-axolotl-multigpu"),
        ("tests-nightly.yml", "docker-e2e-multigpu-tests"),
    ]:
        steps = workflow(file)["jobs"][job]["steps"]
        restore = next(
            s for s in steps if s.get("uses") == "./.github/actions/testmon-restore"
        )
        save = next(
            s for s in steps if s.get("uses") == "./.github/actions/testmon-save"
        )
        assert restore["with"]["suite"] == save["with"]["suite"]
        keys.append(restore["with"]["suite"])
        assert save["if"] == "${{ !cancelled() }}"
    assert keys[0].endswith("-pinned")
    assert keys[1].endswith("-nightly")
    refresh = workflow("testmon-refresh.yml")
    assert refresh["jobs"]["multigpu"]["if"] == "inputs.multigpu"
    assert (
        refresh["jobs"]["multigpu"]["uses"]
        == "./.github/workflows/multi-gpu-e2e-jobs.yml"
    )


def test_distributed_manifest_audit_and_partition():
    manifest = json.loads((ROOT / "cicd/feature_suites.json").read_text())
    suites = manifest["distributed"]
    for spec in suites.values():
        actual = set()
        for path in spec["paths"]:
            entry = ROOT / path
            candidates = entry.rglob("test_*.py") if entry.is_dir() else [entry]
            actual.update(
                p.relative_to(ROOT).as_posix()
                for p in candidates
                if not any(
                    p.relative_to(ROOT).as_posix().startswith(exclude)
                    for exclude in spec["exclude"]
                )
            )
        assert sorted(actual) == spec["audited_tests"]
        assert set(spec["features"]) <= manifest["features"].keys()
    groups = ["multigpu-cce", "multigpu-general", "multigpu-solo", "multigpu-patched"]
    assigned = [path for group in groups for path in suites[group]["audited_tests"]]
    expected = {
        p.relative_to(ROOT).as_posix()
        for p in (ROOT / "tests/e2e/multigpu").rglob("test_*.py")
    }
    assert set(assigned) == expected
    assert len(assigned) == len(expected)
