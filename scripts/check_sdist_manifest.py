#!/usr/bin/env python
"""Check that a wheel built from the sdist ships every git-tracked file under src/axolotl."""

import argparse
import fnmatch
import shutil
import subprocess  # nosec B404
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_DIR = "src/axolotl"
WHEEL_PREFIX = "axolotl/"
SDIST_REQUIRED = ("AGENTS.md", "README.md", "LICENSE", "VERSION")
SDIST_REQUIRED_GLOBS = ("docs/agents/*.md",)


def tracked_package_files(repo_root: Path) -> set[str]:
    out = subprocess.check_output(  # nosec B603 B607
        ["git", "ls-files", PACKAGE_DIR], cwd=repo_root, text=True
    )
    files = set()
    for line in out.splitlines():
        rel = line[len(PACKAGE_DIR) + 1 :]
        parts = rel.split("/")
        if "__pycache__" in parts or "tests" in parts[:-1] or rel.endswith(".pyc"):
            continue
        files.add(rel)
    return files


def wheel_package_files(wheel: Path) -> set[str]:
    with zipfile.ZipFile(wheel) as zf:
        return {
            n[len(WHEEL_PREFIX) :] for n in zf.namelist() if n.startswith(WHEEL_PREFIX)
        }


def compare(tracked: set[str], shipped: set[str]) -> tuple[list[str], list[str]]:
    return sorted(tracked - shipped), sorted(shipped - tracked)


def check_sdist_contents(sdist: Path) -> list[str]:
    """Return the required sdist entries that are absent."""
    with tarfile.open(sdist) as tf:
        names = tf.getnames()
    top = names[0].split("/")[0]
    problems = [
        f"{top}/{name}" for name in SDIST_REQUIRED if f"{top}/{name}" not in names
    ]
    for pattern in SDIST_REQUIRED_GLOBS:
        if not fnmatch.filter(names, f"{top}/{pattern}"):
            problems.append(f"{top}/{pattern}")
    return problems


def report(missing: list[str], extra: list[str], tracked: int, shipped: int) -> int:
    for path in missing:
        print(f"MISSING (tracked, not in wheel): {path}")
    for path in extra:
        print(f"extra (in wheel, not git-tracked): {path}")
    print(
        f"sdist-manifest: tracked={tracked} wheel={shipped} "
        f"missing={len(missing)} extra={len(extra)}"
    )
    return 1 if missing else 0


def build(kind: str, source: Path, out_dir: Path) -> Path:
    if shutil.which("uv"):
        cmd = ["uv", "build", f"--{kind}", "--out-dir", str(out_dir), str(source)]
    else:
        cmd = [
            sys.executable,
            "-m",
            "build",
            f"--{kind}",
            "--outdir",
            str(out_dir),
            str(source),
        ]
    subprocess.run(cmd, check=True)  # nosec B603
    pattern = "*.tar.gz" if kind == "sdist" else "*.whl"
    produced = list(out_dir.glob(pattern))
    assert len(produced) == 1, f"expected one {pattern} in {out_dir}, got {produced}"
    return produced[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sdist", type=Path, help="check this prebuilt sdist")
    parser.add_argument("--keep", action="store_true", help="keep the temp dir")
    parser.add_argument("--out-dir", type=Path, help="copy sdist and wheel here")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    work = Path(tempfile.mkdtemp(prefix="sdist-check-"))
    rc = 0
    try:
        sdist = args.sdist or build("sdist", args.repo_root, work / "dist")
        problems = check_sdist_contents(sdist)
        for problem in problems:
            print(f"MISSING from sdist: {problem}")
        with tarfile.open(sdist) as tf:
            tf.extractall(work / "src", filter="data")
        (sdist_dir,) = [p for p in (work / "src").iterdir() if p.is_dir()]
        wheel = build("wheel", sdist_dir, work / "wheel")
        missing, extra = compare(
            tracked_package_files(args.repo_root), wheel_package_files(wheel)
        )
        rc = report(
            missing,
            extra,
            len(tracked_package_files(args.repo_root)),
            len(wheel_package_files(wheel)),
        )
        if problems:
            rc = 1
        if args.out_dir:
            args.out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(sdist, args.out_dir)
            shutil.copy2(wheel, args.out_dir)
    finally:
        if args.keep:
            print(f"kept: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    sys.exit(rc)


if __name__ == "__main__":
    main()
