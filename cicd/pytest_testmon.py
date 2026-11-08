"""Run one CI pytest block with isolated testmon state."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess  # nosec B404
import sys
from pathlib import Path


def source_fingerprint(root: Path) -> str:
    """Invalidate child-process suites when code or configuration changes."""
    paths = (
        subprocess.check_output(  # nosec
            [
                "git",
                "ls-files",
                "-z",
                "--",
                "src",
                "tests",
                "examples",
                "cicd",
                "scripts",
                "pyproject.toml",
                "uv.lock",
                "requirements*",
            ],
            cwd=root,
        )
        .decode()
        .split("\0")
    )
    digest = hashlib.sha256()
    for name in sorted(filter(None, paths)):
        path = root / name
        digest.update(name.encode())
        digest.update(path.read_bytes() if path.is_file() else b"<deleted>")
    return digest.hexdigest()


def testmon_args(args: list[str], root: Path) -> list[str]:
    if os.environ.get("TESTMON_FULL") == "true":
        return ["--no-testmon", *args]
    block = hashlib.sha256(json.dumps(args).encode()).hexdigest()[:16]
    state = root / ".testmon"
    state.mkdir(exist_ok=True)
    os.environ["TESTMON_DATAFILE"] = str(state / f"{block}.testmondata")
    # pytest-cov disables testmon collection.
    filtered = []
    skip_value = False
    for arg in args:
        if skip_value:
            skip_value = False
            continue
        if arg in {"--cov", "--cov-report", "--cov-config", "--cov-fail-under"}:
            skip_value = True
        elif not arg.startswith("--cov"):
            filtered.append(arg)
    option = (
        "--testmon-forceselect"
        if os.environ.get("TESTMON_SELECT") == "true"
        else "--testmon-noselect"
    )
    options = [option]
    if os.environ.get("TESTMON_SUBPROCESSES") == "true":
        options.append(f"--testmon-env={source_fingerprint(root)}")
    return [*options, *filtered]


def main() -> int:
    import pytest

    return int(pytest.main(testmon_args(sys.argv[1:], Path.cwd())))


if __name__ == "__main__":
    sys.exit(main())
