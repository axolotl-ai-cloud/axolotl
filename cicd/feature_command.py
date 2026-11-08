"""Select a distributed suite, then execute its entire command without testmon."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path

from feature_testmon import VERSION, choose, environment, partition, snapshot


def run(root: Path, name: str, command: list[str]) -> int:
    manifest = json.loads((root / "cicd/feature_suites.json").read_text())
    spec = manifest["distributed"][name]
    if set(spec["features"]) - manifest["features"].keys():
        raise ValueError("Suite names an unknown feature")
    suites = partition(root, {**spec, "suites": {}})
    if not suites:
        raise ValueError(f"No tests in distributed suite {name}")
    tests = suites[0][1]
    # New test files need a feature audit before they can inherit exclusions.
    features = spec["features"] if tests == spec["audited_tests"] else None
    identity = hashlib.sha256(json.dumps([name, command]).encode()).hexdigest()[:16]
    metadata = root / ".testmon" / f"distributed-{identity}.scope.json"
    previous = {}
    try:
        previous = json.loads(metadata.read_text())
        if not isinstance(previous, dict):
            previous = {}
    except (OSError, ValueError):
        pass
    current = {"version": VERSION, "environment": {}, "files": {}}
    inspected = False
    try:
        current.update(files=snapshot(root), environment=environment())
        current["environment"]["settings"]["PYTEST_ADDOPTS"] = os.environ.get(
            "PYTEST_ADDOPTS", ""
        )
        inspected = True
    except (OSError, subprocess.CalledProcessError):
        pass
    mode, reason = choose(previous, current, features, manifest["features"], inspected)
    if not inspected:
        mode, reason = "full", "cannot inspect baseline inputs"
    if (
        os.environ.get("TESTMON_FULL") == "true"
        or os.environ.get("TESTMON_SELECT") != "true"
    ):
        mode, reason = "full", "full-run override or baseline refresh"
    if mode == "testmon":
        mode = "full"
    print(f"distributed-suite {name}: {mode}: {reason}", flush=True)
    if mode == "skip":
        return 0
    metadata.parent.mkdir(exist_ok=True)
    metadata.write_text(json.dumps({**current, "success": False}, sort_keys=True))
    # Worker execution cannot be attributed to individual parent pytest tests.
    env = {
        **os.environ,
        "PYTEST_ADDOPTS": os.environ.get("PYTEST_ADDOPTS", "") + " --no-testmon",
    }
    result = subprocess.run(command, cwd=root, env=env, check=False)  # nosec B603
    metadata.write_text(
        json.dumps(
            {**current, "success": inspected and result.returncode == 0}, sort_keys=True
        )
    )
    return result.returncode


def main() -> int:
    name, *command = sys.argv[1:]
    if not command:
        raise ValueError("A suite command is required")
    return run(Path.cwd(), name, command)


if __name__ == "__main__":
    sys.exit(main())
