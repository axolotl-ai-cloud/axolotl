"""Transfer testmon databases between the Actions runner and Modal calls."""

from __future__ import annotations

import os
import re
import subprocess  # nosec B404
from pathlib import Path

DATAFILE = re.compile(r"[a-zA-Z0-9_-]+(?:\.testmondata(?:-wal|-shm)?|\.scope\.json)\Z")


def read_state(directory: Path) -> dict[str, bytes]:
    if not directory.exists():
        return {}
    return {
        path.name: path.read_bytes()
        for path in directory.iterdir()
        if path.is_file() and DATAFILE.fullmatch(path.name)
    }


def write_state(directory: Path, state: dict[str, bytes]) -> None:
    for name in state:
        if not DATAFILE.fullmatch(name):
            raise ValueError(f"Invalid testmon filename: {name!r}")
    directory.mkdir(parents=True, exist_ok=True)
    for path in directory.iterdir():
        if path.is_file() and DATAFILE.fullmatch(path.name):
            path.unlink()
    for name, data in state.items():
        (directory / name).write_bytes(data)


def run_remote(
    script: str,
    state: dict[str, bytes],
    full: bool,
    select: bool,
    subprocesses: bool,
) -> tuple[int, dict[str, bytes]]:
    root = Path("/workspace/axolotl")
    directory = root / ".testmon"
    write_state(directory, state)
    env = {
        **os.environ,
        "TESTMON_FULL": str(full).lower(),
        "TESTMON_SELECT": str(select).lower(),
        "TESTMON_SUBPROCESSES": str(subprocesses).lower(),
    }
    result = subprocess.run(["bash", script], cwd=root, env=env, check=False)  # nosec B603 B607
    return result.returncode, read_state(directory)


def run_local(remote) -> None:
    directory = Path(".testmon")
    code, state = remote.remote(
        read_state(directory),
        os.environ.get("TESTMON_FULL") == "true",
        os.environ.get("TESTMON_SELECT") == "true",
    )
    write_state(directory, state)
    if code:
        raise RuntimeError(f"Modal pytest failed with exit code {code}")
