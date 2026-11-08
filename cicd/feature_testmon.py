"""Run explicitly bounded e2e suites with testmon and baseline provenance."""

from __future__ import annotations

import fnmatch
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess  # nosec B404
import sys
import xml.etree.ElementTree as ET  # nosec B405
from collections import Counter
from pathlib import Path

VERSION = 1


def matches(path: str, pattern: str) -> bool:
    return (
        path.startswith(pattern)
        if pattern.endswith("/")
        else fnmatch.fnmatchcase(path, pattern)
    )


def partition(root: Path, group: dict) -> list[tuple[str, list[str], list[str] | None]]:
    files: set[str] = set()
    for entry in group["paths"]:
        path = root / entry
        if not path.exists():
            raise FileNotFoundError(f"Missing suite path: {entry}")
        candidates = path.rglob("test_*.py") if path.is_dir() else [path]
        files.update(
            candidate.relative_to(root).as_posix()
            for candidate in candidates
            if candidate.is_file()
        )
    files = {
        path
        for path in files
        if not any(matches(path, pattern) for pattern in group["exclude"])
    }
    assigned: set[str] = set()
    suites = []
    for name, spec in group["suites"].items():
        selected = {
            path
            for path in files
            if any(matches(path, pattern) for pattern in spec["tests"])
        }
        if assigned & selected:
            raise ValueError(
                f"Overlapping suite membership: {sorted(assigned & selected)}"
            )
        assigned |= selected
        if selected:
            suites.append((name, sorted(selected), spec["features"]))
    if remaining := files - assigned:
        suites.append(("general", sorted(remaining), None))
    return suites


def snapshot(root: Path) -> dict[str, str]:
    paths = (
        subprocess.check_output(  # nosec B603 B607
            ["git", "ls-files", "-z"], cwd=root
        )
        .decode()
        .split("\0")
    )
    return {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in paths
        if name and (root / name).is_file()
    }


def environment() -> dict:
    return {
        "python": platform.python_version(),
        "packages": sorted(
            [dist.metadata["Name"], dist.version]
            for dist in importlib.metadata.distributions()
            if dist.metadata["Name"]
        ),
        "settings": {
            name: os.environ.get(name, "")
            for name in (
                "BASE_TAG",
                "GPU_TYPE",
                "N_GPUS",
                "NIGHTLY_BUILD",
                "CUDA_VISIBLE_DEVICES",
                "TESTMON_SUBPROCESSES",
                "MULTIGPU_TEST_SUITE",
                "NF4_TEST_STACK",
            )
        },
    }


def choose(
    previous: dict,
    current: dict,
    features: list[str] | None,
    routes: dict,
    database: bool,
) -> tuple[str, str]:
    if (
        not database
        or previous.get("version") != VERSION
        or not previous.get("success")
    ):
        return "full", "missing baseline or previous suite did not pass"
    if previous.get("environment") != current["environment"]:
        return "full", "environment changed"
    old = previous.get("files")
    if not isinstance(old, dict):
        return "full", "missing source snapshot"
    changed = sorted(
        path
        for path in old.keys() | current["files"].keys()
        if old.get(path) != current["files"].get(path)
    )
    if not changed:
        return "skip", "source and environment match the successful baseline"
    relevant = []
    for path in changed:
        owners = {
            name
            for name, patterns in routes.items()
            if any(matches(path, pattern) for pattern in patterns)
        }
        if features is None or not owners or owners & set(features):
            relevant.append(path)
    if not relevant:
        return "skip", "only other explicitly isolated features changed"
    if any(
        not path.endswith(".py")
        or not path.startswith(("src/", "tests/"))
        or Path(path).name == "conftest.py"
        for path in relevant
    ):
        return "full", "shared setup, configuration, or non-Python inputs changed"
    return "testmon", f"potentially relevant changes: {', '.join(relevant[:5])}"


def run_suite(
    root: Path,
    name: str,
    files: list[str],
    features: list[str] | None,
    routes: dict,
    current: dict,
    args: list[str],
    require_executed: bool = False,
) -> int:
    state = root / ".testmon"
    state.mkdir(exist_ok=True)
    identity = hashlib.sha256(json.dumps([name, files, args]).encode()).hexdigest()[:16]
    metadata = state / f"{identity}.scope.json"
    previous = {}
    try:
        previous = json.loads(metadata.read_text())
        if not isinstance(previous, dict):
            previous = {}
    except (OSError, ValueError):
        pass
    report = root / f"{name}-junit.xml"
    pytest_args = [*args, *files]
    if require_executed:
        pytest_args.append(f"--junitxml={report}")
    full_override = os.environ.get("TESTMON_FULL") == "true"
    select = os.environ.get("TESTMON_SELECT") == "true"
    block = hashlib.sha256(json.dumps(pytest_args).encode()).hexdigest()[:16]
    database = state / f"{block}.testmondata"
    valid_database = database.is_file() and bool(previous.get("database"))
    if valid_database:
        valid_database = (
            hashlib.sha256(database.read_bytes()).hexdigest() == previous["database"]
        )
    if not valid_database:
        for suffix in ("", "-wal", "-shm"):
            Path(str(database) + suffix).unlink(missing_ok=True)
    mode, reason = choose(previous, current, features, routes, valid_database)
    if full_override or not select:
        mode, reason = "full", "full-run override or baseline refresh"
    print(f"feature-suite {name}: {mode}: {reason}", flush=True)
    if mode == "skip":
        return 0
    env = {**os.environ, "TESTMON_SELECT": str(mode == "testmon").lower()}
    if not full_override:
        metadata.write_text(json.dumps({**current, "success": False}, sort_keys=True))
    command = [
        sys.executable,
        str(Path(__file__).with_name("pytest_testmon.py")),
        *pytest_args,
    ]
    while True:
        if require_executed:
            report.unlink(missing_ok=True)
        result = subprocess.run(command, cwd=root, env=env, check=False)  # nosec B603
        code = result.returncode
        if not require_executed or code not in (0, 5):
            break
        try:
            cases = list(ET.parse(report).iter("testcase"))  # nosec B314
        except (OSError, ET.ParseError) as exc:
            print(f"{name}: cannot validate test execution: {exc}", flush=True)
            code = 1
            break
        skips = Counter(
            skipped.get("message", "<no reason>")
            for case in cases
            if (skipped := case.find("skipped")) is not None
        )
        executed = len(cases) - sum(skips.values())
        print(
            f"{name}: {executed} ran, {sum(skips.values())} skipped: {dict(skips)}",
            flush=True,
        )
        if executed or (not cases and env["TESTMON_SELECT"] == "true"):
            code = 0
            break
        if env["TESTMON_SELECT"] == "true":
            # A changed SM120-only test can be the only selected test on B200.
            print(
                f"{name}: validating skip-only selection with a full suite", flush=True
            )
            env["TESTMON_SELECT"] = "false"
            continue
        print(f"{name}: no tests executed in the full suite", flush=True)
        code = 1
        break
    if not full_override:
        metadata.write_text(
            json.dumps(
                {
                    **current,
                    "success": code == 0,
                    "database": hashlib.sha256(database.read_bytes()).hexdigest()
                    if database.is_file()
                    else None,
                },
                sort_keys=True,
            )
        )
    return code


def main() -> int:
    group_name, *args = sys.argv[1:]
    root = Path.cwd()
    manifest = json.loads((root / "cicd/feature_suites.json").read_text())
    group = manifest["groups"][group_name]
    for spec in group["suites"].values():
        if set(spec["features"]) - manifest["features"].keys():
            raise ValueError("Suite names an unknown feature")
    suites = partition(root, group)
    suites.sort(key=lambda suite: suite[0] in group.get("run_last", []))
    if not suites:
        raise ValueError(f"No test files in feature group {group_name}")
    try:
        files = snapshot(root)
        env = environment()
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()  # nosec B603 B607
        current = {
            "version": VERSION,
            "revision": revision,
            "files": files,
            "environment": env,
        }
    except (OSError, subprocess.CalledProcessError):
        print(
            "feature-suite: cannot inspect baseline inputs; running every eligible test",
            flush=True,
        )
        os.environ["TESTMON_SELECT"] = "false"
        current = {"version": VERSION, "files": {}, "environment": {}}
    failed = 0
    for name, tests, features in suites:
        failed |= run_suite(
            root,
            f"{group_name}-{name}",
            tests,
            features,
            manifest["features"],
            current,
            args,
            require_executed=group.get("require_executed", False),
        )
    return int(bool(failed))


if __name__ == "__main__":
    sys.exit(main())
