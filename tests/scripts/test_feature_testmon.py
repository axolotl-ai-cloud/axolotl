"""CPU checks for explicit e2e suite boundaries and baseline provenance."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "feature_testmon", ROOT / "cicd/feature_testmon.py"
)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)
ROUTES = {"liger": ["src/liger/"], "cce": ["src/cce/"]}


def baseline(files=None):
    return {
        "version": mod.VERSION,
        "success": True,
        "files": files or {},
        "environment": {"python": "pinned"},
    }


@pytest.mark.parametrize(
    "features,expected",
    [(["cce"], "skip"), (["liger"], "testmon"), ([], "skip"), (None, "testmon")],
)
def test_explicit_feature_boundary(features, expected):
    old = baseline({"src/liger/impl.py": "old"})
    new = baseline({"src/liger/impl.py": "new"})
    assert mod.choose(old, new, features, ROUTES, True)[0] == expected


@pytest.mark.parametrize(
    "path,expected",
    [
        ("src/shared.py", "testmon"),
        ("fixtures/data.json", "full"),
        ("tests/conftest.py", "full"),
        ("cicd/run.py", "full"),
        ("tests/test_new.py", "testmon"),
    ],
)
def test_shared_unknown_and_added_inputs(path, expected):
    assert (
        mod.choose(baseline(), baseline({path: "new"}), [], ROUTES, True)[0] == expected
    )


def test_intervening_main_change_is_not_hidden():
    old = baseline({"src/shared.py": "old", "src/liger/impl.py": "old"})
    current = baseline(
        {"src/shared.py": "new-on-main", "src/liger/impl.py": "pr-change"}
    )
    assert mod.choose(old, current, ["cce"], ROUTES, True)[0] == "testmon"


@pytest.mark.parametrize(
    "change",
    [{"success": False}, {"environment": {}}, {"version": -1}, {"files": None}],
)
def test_invalid_baseline_runs_full(change):
    assert mod.choose(baseline() | change, baseline(), [], ROUTES, True)[0] == "full"


def test_missing_database_runs_full():
    assert mod.choose(baseline(), baseline(), [], ROUTES, False)[0] == "full"


def test_partition_retains_unclassified_tests(tmp_path):
    directory = tmp_path / "tests"
    directory.mkdir()
    for name in ["test_a.py", "test_new.py", "test_separate.py"]:
        (directory / name).touch()
    group = {
        "paths": ["tests/"],
        "exclude": ["tests/test_separate.py"],
        "suites": {"a": {"tests": ["tests/test_a.py"], "features": []}},
    }
    assert mod.partition(tmp_path, group) == [
        ("a", ["tests/test_a.py"], []),
        ("general", ["tests/test_new.py"], None),
    ]
    group["suites"]["overlap"] = group["suites"]["a"]
    with pytest.raises(ValueError, match="Overlapping"):
        mod.partition(tmp_path, group)


def test_repository_suites_cover_current_groups():
    manifest = json.loads((ROOT / "cicd/feature_suites.json").read_text())
    for group in manifest["groups"].values():
        suites = mod.partition(ROOT, group)
        assert suites
        names = [path for _, files, _ in suites for path in files]
        assert len(names) == len(set(names))
        assert all(
            not any(mod.matches(path, exclusion) for exclusion in group["exclude"])
            for path in names
        )
    integrations = mod.partition(ROOT, manifest["groups"]["integrations"])
    assert any(
        name == "general" and "tests/e2e/integrations/test_fp8.py" in files
        for name, files, _ in integrations
    )


def test_actual_refresh_skip_input_change_failure_and_override(tmp_path):
    pytest.importorskip("testmon")
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    (tmp_path / "cicd").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "src/liger").mkdir(parents=True)
    plugin = tmp_path / "src/liger/impl.py"
    plugin.write_text("x = 1\n")
    (tmp_path / "input.txt").write_text("ok")
    test = tmp_path / "tests/test_sample.py"
    test.write_text(
        "from pathlib import Path\ndef test_ok():\n    with Path('executions').open('a') as f: f.write('run\\n')\n    assert Path('input.txt').read_text() == 'ok'\n"
    )
    manifest = {
        "features": ROUTES,
        "groups": {
            "demo": {
                "paths": ["tests/"],
                "exclude": [],
                "suites": {
                    "cce": {"tests": ["tests/test_sample.py"], "features": ["cce"]}
                },
            }
        },
    }
    (tmp_path / "cicd/feature_suites.json").write_text(json.dumps(manifest))
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=CI",
            "-c",
            "user.email=ci@example.invalid",
            "commit",
            "-qm",
            "baseline",
        ],
        cwd=tmp_path,
        check=True,
    )
    env = {
        **os.environ,
        "TESTMON_SELECT": "true",
        "TESTMON_FULL": "false",
        "TESTMON_SUBPROCESSES": "false",
    }

    def run(**overrides):
        return subprocess.run(
            [sys.executable, str(ROOT / "cicd/feature_testmon.py"), "demo", "-q"],
            cwd=tmp_path,
            env=env | overrides,
            text=True,
            capture_output=True,
        )

    first = run(TESTMON_SELECT="false")
    assert first.returncode == 0, first.stdout + first.stderr
    plugin.write_text("x = 2\n")
    skipped = run()
    assert skipped.returncode == 0, skipped.stdout + skipped.stderr
    assert "only other explicitly isolated features changed" in skipped.stdout
    assert (tmp_path / "executions").read_text().count("run") == 1
    full = run(TESTMON_FULL="true")
    assert full.returncode == 0, full.stdout + full.stderr
    assert (tmp_path / "executions").read_text().count("run") == 2
    (tmp_path / "input.txt").write_text("bad")
    failed = run()
    assert failed.returncode == 1, failed.stdout + failed.stderr
    assert "non-Python inputs changed" in failed.stdout
    retry = run()
    assert retry.returncode == 1
    assert "previous suite did not pass" in retry.stdout
    (tmp_path / "input.txt").write_text("ok")
    assert run().returncode == 0
    database = next((tmp_path / ".testmon").glob("*.testmondata"))
    database.write_bytes(b"corrupt")
    repaired = run()
    assert repaired.returncode == 0, repaired.stdout + repaired.stderr
    assert "missing baseline" in repaired.stdout
