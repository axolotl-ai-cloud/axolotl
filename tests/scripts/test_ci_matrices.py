"""Keep scheduled CI environments aligned with their PR counterparts."""

import json
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
WORKFLOWS = Path(__file__).resolve().parents[2] / ".github/workflows"
EXPECTED = {("3.12", "2.13.0"), ("3.14", "2.14.1"), ("3.14", "2.15.0")}


def workflow(name):
    return yaml.safe_load((WORKFLOWS / name).read_text())


def cells(job, suite=None):
    entries = job["strategy"]["matrix"]["include"]
    return {
        (
            ".".join(str(entry["python_version"]).split(".")[:2]),
            str(entry.get("pytorch_version", entry.get("pytorch"))),
        )
        for entry in entries
        if suite is None or entry.get("test_suite") == suite
    }


@pytest.mark.parametrize(
    "name,job,suite",
    [
        ("testmon-cpu.yml", "pytest-testmon", None),
        ("testmon-e2e-baseline.yml", "single-gpu", None),
        ("tests-nightly.yml", "pytest", None),
        ("tests-nightly.yml", "docker-e2e-tests", None),
        ("tests-nightly.yml", "docker-e2e-multigpu-tests", "multigpu"),
        ("testmon-gpu-kernels.yml", "docker-e2e-kernel-tests", None),
        ("multi-gpu-e2e-jobs.yml", "test-axolotl-multigpu", "multigpu"),
        ("nightlies.yml", "build-axolotl", None),
        ("nightlies.yml", "build-axolotl-cloud", None),
    ],
)
def test_general_matrix(name, job, suite):
    assert cells(workflow(name)["jobs"][job], suite) == EXPECTED


def test_docker_e2e_matrix():
    jobs = workflow("docker-e2e-jobs.yml")["jobs"]
    gate = cells(jobs["docker-e2e-tests-testmon"])
    followers = cells(jobs["docker-e2e-tests"])
    assert gate.isdisjoint(followers)
    assert gate | followers == EXPECTED


def test_distributed_matrix_and_slow_refresh_match():
    config = workflow("distributed-cpu.yml")
    # PyYAML's YAML 1.1 loader interprets the workflow's "on" key as True.
    events = config.get("on", config.get(True))
    inputs = events["workflow_call"]["inputs"]
    entries = json.loads(inputs["python_pytorch_matrix"]["default"])
    assert cells({"strategy": {"matrix": {"include": entries}}}) == EXPECTED
    scheduled = workflow("cpu-slow.yml")["jobs"]["cpu-slow"]["with"]
    manual = workflow("testmon-refresh.yml")["jobs"]["slow-cpu"]["with"]
    for key in ("python_pytorch_matrix", "slow", "nightly"):
        default = inputs[key]["default"]
        scheduled_value = scheduled.get(key, default)
        manual_value = manual.get(key, default)
        if key == "python_pytorch_matrix":
            scheduled_value = json.loads(scheduled_value)
            manual_value = json.loads(manual_value)
        assert scheduled_value == manual_value, key


def test_nightly_and_manual_reuse_kernel_matrix():
    nightly = workflow("tests-nightly.yml")["jobs"]["docker-e2e-kernel-tests"]
    manual = workflow("testmon-refresh.yml")["jobs"]["gpu-kernels"]
    assert (
        nightly["uses"]
        == manual["uses"]
        == "./.github/workflows/testmon-gpu-kernels.yml"
    )
    assert nightly["with"]["nightly"] is True


def test_expensive_baseline_refresh_is_explicit():
    config = workflow("testmon-refresh.yml")
    events = config.get("on", config.get(True))
    assert set(events) == {"workflow_dispatch"}
    assert all(
        not option["default"]
        for option in events["workflow_dispatch"]["inputs"].values()
    )
    assert "single-gpu" not in workflow("testmon-baseline.yml")["jobs"]
