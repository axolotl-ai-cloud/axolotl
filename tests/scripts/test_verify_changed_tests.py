"""Unit tests for cicd/verify_changed_tests.py."""

import difflib
import importlib.util
import sys
import textwrap
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "cicd" / "verify_changed_tests.py"
_spec = importlib.util.spec_from_file_location("verify_changed_tests", SCRIPT)
assert _spec is not None and _spec.loader is not None
mod = importlib.util.module_from_spec(_spec)
sys.modules["verify_changed_tests"] = mod
_spec.loader.exec_module(mod)

PATH = "tests/test_sample.py"

OLD = '''\
import pytest
from helpers import build, other


@pytest.fixture(name="cfg")
def _cfg_fixture():
    return {"lr": 1}


def _helper(x):
    return x + 1


def _uses_helper(x):
    return _helper(x)


def test_plain():
    assert build() == 1


def test_fixture(cfg):
    assert cfg


def test_helper():
    assert _uses_helper(1) == 2


class _Base:
    def test_inherited(self):
        assert True


class TestThing(_Base):
    """Docs."""

    def test_method(self):
        assert self._make() == 1

    def test_other(self):
        assert other()

    def _make(self):
        return 1
'''


def _changed(old: str, new: str) -> dict[str, str]:
    """Changed test nodeids and why, for a modification from ``old`` to ``new``."""
    changes = mod.parse_diff(_unified_diff(old, new))
    assert len(changes) == 1
    tests = mod.changed_tests_for_file(changes[0], old, new)
    assert tests is not None
    return {t.nodeid.split("::", 1)[1]: t.why for t in tests}


def _unified_diff(old: str, new: str) -> str:
    lines = difflib.unified_diff(
        old.splitlines(), new.splitlines(), f"a/{PATH}", f"b/{PATH}", n=0, lineterm=""
    )
    return f"diff --git a/{PATH} b/{PATH}\n" + "\n".join(lines) + "\n"


def test_parse_log_reads_xdist_serial_and_split_lines():
    log = textwrap.dedent(
        """\
        2026-11-05T19:40:09.3774394Z [gw0] [  0%] PASSED tests/a.py::test_x[p 1]
        2026-11-05T19:40:09.3774394Z \x1b[32m[gw1] [  5%] SKIPPED tests/a.py::T::test_y\x1b[0m
        tests/b.py::test_z PASSED [ 40%]
        tests/b.py::test_w[a] XFAIL (known) [ 41%]
        tests/c.py::TestR::test_rm some captured output
        more output
        PASSED [100%]
        ======== warnings summary ========
        tests/d.py::test_warned
          /x.py:1: DeprecationWarning: old
        tests/e.py::test_late FAILED [ 50%]
        """
    )
    assert mod.parse_log(log) == {
        "tests/a.py::test_x[p 1]": {"PASSED"},
        "tests/a.py::T::test_y": {"SKIPPED"},
        "tests/b.py::test_z": {"PASSED"},
        "tests/b.py::test_w[a]": {"XFAIL"},
        "tests/c.py::TestR::test_rm": {"PASSED"},
        "tests/e.py::test_late": {"FAILED"},
    }


def test_parse_diff_tracks_renames_additions_and_deletions():
    diff = textwrap.dedent(
        """\
        diff --git a/tests/old.py b/tests/new.py
        similarity index 100%
        rename from tests/old.py
        rename to tests/new.py
        diff --git a/tests/added.py b/tests/added.py
        new file mode 100644
        --- /dev/null
        +++ b/tests/added.py
        @@ -0,0 +1,3 @@
        diff --git a/tests/mod.py b/tests/mod.py
        --- a/tests/mod.py
        +++ b/tests/mod.py
        @@ -10,2 +10,0 @@ def f():
        @@ -20 +18,3 @@ def g():
        """
    )
    renamed, added, modified = mod.parse_diff(diff)
    assert (renamed.old_path, renamed.new_path, renamed.renamed) == (
        "tests/old.py",
        "tests/new.py",
        True,
    )
    assert (added.old_path, added.new_path, added.new_lines) == (
        None,
        "tests/added.py",
        {1, 2, 3},
    )
    assert modified.old_lines == {10, 11, 20}
    assert modified.new_lines == {18, 19, 20}


def test_editing_a_test_body_selects_only_that_test():
    new = OLD.replace("assert build() == 1", "assert build() == 2")
    assert _changed(OLD, new) == {"test_plain": "edited"}


def test_adding_a_method_does_not_select_its_neighbours():
    new = OLD.replace(
        "    def _make(self):",
        "    def test_added(self):\n        assert True\n\n    def _make(self):",
    )
    assert _changed(OLD, new) == {"TestThing::test_added": "edited"}


def test_deleting_a_test_selects_nothing():
    new = OLD.replace("    def test_other(self):\n        assert other()\n\n", "")
    assert _changed(OLD, new) == {}


def test_helpers_select_their_callers_transitively():
    new = OLD.replace("return x + 1", "return x + 2")
    assert _changed(OLD, new) == {"test_helper": "uses _uses_helper"}


def test_helper_methods_select_tests_calling_them():
    new = OLD.replace("        return 1\n", "        return 2\n")
    assert _changed(OLD, new) == {"TestThing::test_method": "uses _make"}


def test_fixtures_select_by_requested_name():
    new = OLD.replace('{"lr": 1}', '{"lr": 2}')
    assert _changed(OLD, new) == {"test_fixture": "uses cfg"}


def test_base_class_tests_select_the_collected_subclass():
    new = OLD.replace(
        "    def test_inherited(self):\n        assert True",
        ("    def test_inherited(self):\n        assert not False"),
    )
    assert _changed(OLD, new) == {"TestThing::test_inherited": "edited"}


def test_import_changes_select_users_of_changed_bindings_only():
    new = OLD.replace(
        "from helpers import build, other", "from helpers import build, other2 as other"
    )
    assert _changed(OLD, new) == {"TestThing::test_other": "uses other"}


def test_class_docstrings_and_blank_lines_are_inert():
    new = OLD.replace('"""Docs."""', '"""Other docs."""').replace(
        "    def test_other", "\n    def test_other"
    )
    assert _changed(OLD, new) == {}


def test_autouse_fixture_and_pytestmark_select_the_module():
    every = {
        "test_plain",
        "test_fixture",
        "test_helper",
        "TestThing::test_inherited",
        "TestThing::test_method",
        "TestThing::test_other",
    }
    autouse = OLD.replace(
        '@pytest.fixture(name="cfg")', '@pytest.fixture(name="cfg", autouse=True)'
    )
    assert set(_changed(OLD, autouse)) == every
    marked = OLD.replace(
        "import pytest\n", "import pytest\n\npytestmark = pytest.mark.slow\n"
    )
    assert set(_changed(OLD, marked)) == every


def test_new_files_select_every_test():
    change = mod.FileChange(None, PATH, new_lines={1})
    tests = mod.changed_tests_for_file(change, None, OLD)
    assert tests is not None
    assert {t.why for t in tests} == {"new file"}
    lines = {t.nodeid.split("::", 1)[1]: t.line for t in tests}
    assert lines["test_plain"] == OLD.splitlines().index("def test_plain():") + 1
    # an inherited test is annotated where its base class defines it
    assert lines["TestThing::test_inherited"] == (
        OLD.splitlines().index("    def test_inherited(self):") + 1
    )
    assert len(tests) == 6


def test_unparsable_files_are_reported():
    change = mod.FileChange(PATH, PATH, new_lines={1})
    assert mod.changed_tests_for_file(change, OLD, "def test_x(:\n") is None


@pytest.mark.parametrize(
    ("source", "markers"),
    [
        (
            "import pytest\n@pytest.mark.slow\n@pytest.mark.gpu\ndef test_x(): pass\n",
            {"slow", "gpu"},
        ),
        (
            "import pytest, torch\n"
            "requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='x')\n"
            "@requires_cuda\ndef test_x(): pass\n",
            {"gpu"},
        ),
        (
            "import pytest\n"
            "pytestmark = pytest.mark.skipif(True, reason='needs Blackwell (sm100)')\n"
            "def test_x(): pass\n",
            {"gpu_blackwell"},
        ),
        (
            "import pytest\n"
            "@pytest.mark.parametrize('d', ['cpu', pytest.param('cuda', marks=pytest.mark.gpu)])\n"
            "def test_x(d): pass\n",
            {"gpu"},
        ),
        (
            "import pytest\n"
            "class TestA:\n"
            "    pytestmark = pytest.mark.skip(reason='broken')\n"
            "    def test_x(self): pass\n",
            {"skip"},
        ),
    ],
)
def test_static_markers(source, markers):
    index = mod.ModuleIndex(PATH, source)
    (test,) = index.owners
    assert index.markers(test) == markers


@pytest.mark.parametrize(
    ("path", "markers", "label", "lane"),
    [
        ("tests/test_x.py", set(), None, mod.CPU_ROUTE.lane),
        ("tests/e2e/test_x.py", set(), "run-gpu-tests", "single-GPU e2e"),
        ("tests/core/test_x.py", {"gpu"}, "run-gpu-tests", "single-GPU e2e"),
        ("tests/integrations/kernels/test_x.py", set(), "run-gpu-tests", "GPU kernels"),
        ("tests/kernels/test_x.py", {"gpu_blackwell"}, "run-gpu-tests", "B200"),
        ("tests/e2e/multigpu/test_x.py", {"gpu"}, "run-gpu-tests", None),
        ("tests/e2e/kernels/test_x.py", {"slow"}, "run-gpu-slow", "Slow GPU"),
        ("tests/integrations/test_x.py", {"slow"}, "run-cpu-slow", "Slow CPU"),
        ("tests/monkeypatch/test_nf4_loading.py", {"slow"}, "run-gpu-tests", None),
        ("tests/core/test_x.py", {"slow"}, None, None),
    ],
)
def test_route_for(path, markers, label, lane):
    route = mod.route_for(path, markers)
    assert route.label == label
    if lane is not None:
        assert route.lane == lane


def _evidence(outcomes: dict[str, dict[str, set[str]]], gpu_jobs=()) -> object:
    evidence = mod.Evidence()
    for job_label, seen in outcomes.items():
        workflow, name = job_label.split(" › ")
        job = mod.Job(1, name, workflow, 1, "completed", "success")
        assert job.label == job_label
        evidence.add(job, seen)
    assert evidence.gpu_jobs == set(gpu_jobs)
    return evidence


CPU = "Tests › PyTest (3.14, 2.14.1)"
GPU = "Tests › docker-e2e / run-e2e / docker-e2e-tests (130, 3.14, 2.14.1)"


def _test(nodeid: str, markers=()) -> object:
    path, *qualname = nodeid.split("::")
    return mod.ChangedTest(path, tuple(qualname), "edited", set(markers))


def test_judge_statuses_and_routes():
    evidence = _evidence(
        {
            CPU: {
                "tests/a.py::test_ran[x]": {"PASSED"},
                "tests/a.py::test_ran[y]": {"FAILED"},
                "tests/a.py::test_mixed[cpu]": {"PASSED"},
                "tests/a.py::test_mixed[cuda]": {"SKIPPED"},
                "tests/a.py::test_cpu_skip": {"SKIPPED"},
            }
        }
    )
    ran = mod.judge(_test("tests/a.py::test_ran"), evidence)
    assert (ran.status, ran.failed, ran.route) == ("ran", True, None)
    mixed = mod.judge(_test("tests/a.py::test_mixed", {"gpu"}), evidence)
    assert (mixed.status, mixed.route.label) == ("partial", "run-gpu-tests")
    cpu_skip = mod.judge(_test("tests/a.py::test_cpu_skip"), evidence)
    assert (cpu_skip.status, cpu_skip.route) == ("skipped", mod.CUDA_GATE_ROUTE)
    missing = mod.judge(_test("tests/e2e/test_b.py::test_missing"), evidence)
    assert (missing.status, missing.route.label) == ("not_run", "run-gpu-tests")


def test_judge_explains_skips_that_labels_cannot_fix():
    evidence = _evidence(
        {
            CPU: {
                "tests/a.py::test_dep": {"SKIPPED"},
                "tests/a.py::test_off": {"SKIPPED"},
            },
            GPU: {"tests/e2e/test_b.py::test_gpu_skip": {"SKIPPED"}},
        },
        gpu_jobs=[GPU],
    )
    on_gpu = mod.judge(_test("tests/e2e/test_b.py::test_gpu_skip"), evidence)
    assert on_gpu.route == mod.SKIPPED_ON_GPU_ROUTE
    dep = mod.judge(_test("tests/a.py::test_dep"), evidence)
    assert dep.route == mod.CPU_SKIP_ROUTE
    off = mod.judge(_test("tests/a.py::test_off", {"skip"}), evidence)
    assert off.route == mod.SKIP_MARK_ROUTE


def test_evidence_merges_runs_that_share_job_labels():
    evidence = mod.Evidence()
    job = mod.Job(1, "PyTest (3.14, 2.14.1)", "Tests", 1, "completed", "success")
    evidence.add(job, {"tests/a.py::test_x": {"FAILED"}})
    evidence.add(job, {"tests/a.py::test_x": {"PASSED"}})
    verdict = mod.judge(_test("tests/a.py::test_x"), evidence)
    assert verdict.variants["tests/a.py::test_x"][CPU] == {"FAILED", "PASSED"}
    assert (verdict.status, verdict.failed) == ("ran", True)


def test_find_pr_skips_heads_no_pull_request_has(monkeypatch):
    monkeypatch.setattr(mod, "_gh_items", lambda path, jq: [{"number": 7}])
    monkeypatch.setattr(mod, "_gh", lambda *args: "b" * 40 + "\n")
    assert mod.find_pr("o/r", "b" * 40) == 7
    assert mod.find_pr("o/r", "a" * 40) is None


def test_gather_evidence_does_not_wait_on_its_own_run(monkeypatch):
    def items(path, jq):
        if "/jobs?" in path:
            return [
                {
                    "id": 9,
                    "name": "Changed tests report",
                    "run_attempt": 1,
                    "status": "in_progress",
                    "conclusion": None,
                }
            ]
        return [
            {"id": 1, "name": "Tests", "status": "in_progress"},
            {"id": 2, "name": "Slow GPU", "status": "in_progress"},
            {"id": 3, "name": "lint", "status": "in_progress"},
        ]

    monkeypatch.setattr(mod, "_gh_items", items)
    evidence = mod.gather_evidence("o/r", "a" * 40, current_run=1)
    assert evidence.pending == ["Slow GPU"]
    assert mod.gather_evidence("o/r", "a" * 40).pending == ["Tests", "Slow GPU"]


@pytest.mark.parametrize(
    ("help_text", "flags"),
    [
        (
            "  --allow-escape-sequences   Allow ...\n  --jq string",
            ("--allow-escape-sequences",),
        ),
        ("  --jq string", ()),
    ],
)
def test_log_downloads_allow_escapes_when_gh_supports_it(monkeypatch, help_text, flags):
    commands = []

    def check_output(argv, **kwargs):
        commands.append(argv)
        return b"tests/a.py::test_x PASSED [100%]\x1b[0m\n"

    mod._gh_log_flags.cache_clear()
    monkeypatch.setattr(mod, "_gh", lambda *args: help_text)
    monkeypatch.setattr(mod.subprocess, "check_output", check_output)
    monkeypatch.setattr(
        mod,
        "_gh_items",
        lambda path, jq: (
            [{"id": 1, "name": "Tests", "status": "completed"}]
            if "runs?" in path
            else [
                {
                    "id": 2,
                    "name": "PyTest (3.14, 2.14.1)",
                    "run_attempt": 1,
                    "status": "completed",
                    "conclusion": "success",
                }
            ]
        ),
    )
    evidence = mod.gather_evidence("o/r", "a" * 40)
    mod._gh_log_flags.cache_clear()
    assert commands == [["gh", "api", *flags, "repos/o/r/actions/jobs/2/logs"]]
    assert evidence.by_base["tests/a.py::test_x"]["tests/a.py::test_x"] == {
        CPU: {"PASSED"}
    }


def test_status_payload():
    evidence = _evidence({CPU: {"tests/a.py::test_ran": {"PASSED"}}})
    ran = mod.judge(_test("tests/a.py::test_ran"), evidence)
    missing = mod.judge(_test("tests/e2e/test_b.py::test_missing"), evidence)
    assert mod.status_payload([], evidence, set()) == ("success", "No changed tests")
    assert mod.status_payload([ran], evidence, set()) == (
        "success",
        "All 1 changed tests ran",
    )
    assert mod.status_payload([ran, missing], evidence, set()) == (
        "failure",
        "1/2 changed tests ran; run-gpu-tests 1",
    )
    assert mod.status_payload([ran, missing], evidence, {"run-gpu-tests"}) == (
        "failure",
        "1/2 changed tests ran; not run 1",
    )
    evidence.unreadable.append("Tests › PyTest (gh: Not Found (HTTP 404))")
    assert mod.status_payload([ran, missing], evidence, set()) == (
        "error",
        "1 job logs unreadable; 1/2 changed tests ran; run-gpu-tests 1",
    )
    evidence.pending.append("Tests")
    state, text = mod.status_payload([ran, missing], evidence, set())
    assert state == "pending" and text.startswith("Waiting on Tests; ")


def test_render_lists_actions_and_support_files():
    evidence = _evidence({CPU: {"tests/a.py::test_ran": {"PASSED"}}})
    verdicts = [
        mod.judge(_test("tests/a.py::test_ran"), evidence),
        mod.judge(_test("tests/e2e/test_b.py::test_missing"), evidence),
    ]
    changes = mod.ChangeSet(support_files=["tests/conftest.py"])
    report = mod.render(1, "a" * 40, changes, verdicts, evidence, set())
    assert report.startswith("## Changed tests: 1/2 ran\n")
    assert "**Add `run-gpu-tests` (single-GPU e2e)** · 1" in report
    assert "- `tests/conftest.py`" in report


def _missing_verdicts(n: int) -> list:
    evidence = _evidence({})
    return [
        mod.judge(
            mod.ChangedTest(
                "tests/e2e/t.py", ("TestA", f"test_{i}"), "edited", set(), i
            ),
            evidence,
        )
        for i in range(1, n + 1)
    ]


def test_annotations_name_the_action_at_the_def_line():
    (note,) = mod.annotations_for(_missing_verdicts(1), set())
    assert (note.path, note.line, note.title) == (
        "tests/e2e/t.py",
        1,
        "Not run: TestA::test_1",
    )
    assert note.message == "Add run-gpu-tests (single-GPU e2e). Changed: edited."


def test_workflow_commands_escape_and_cap_per_test_warnings():
    notes = mod.annotations_for(_missing_verdicts(12), set())
    lines = mod.workflow_commands(notes, "failure", "0/12 ran, 50%")
    assert len(lines) == mod.MAX_COMMAND_ANNOTATIONS + 1
    assert lines[0] == (
        "::warning file=tests/e2e/t.py,line=1,title=Not run%3A TestA%3A%3Atest_1"
        "::Add run-gpu-tests (single-GPU e2e). Changed: edited."
    )
    assert lines[-1] == (
        "::warning title=Changed tests executed::0/12 ran, 50%25 "
        "(3 more tests in the run summary)"
    )
    assert mod.workflow_commands([], "success", "All 2 ran") == [
        "::notice title=Changed tests executed::All 2 ran"
    ]


def test_post_check_run_supersedes_stale_checks_and_batches_annotations(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(mod, "_gh_items", lambda path, jq: [{"id": 5}])

    def gh_json(method, path, body):
        calls.append((method, path, body))
        return {"id": 9}

    monkeypatch.setattr(mod, "_gh_json", gh_json)
    notes = mod.annotations_for(_missing_verdicts(120), set())
    mod.post_check_run("o/r", "a" * 40, "failure", "title", "x" * 70000, notes)

    stale, create, *appends = calls
    assert stale[:2] == ("PATCH", "repos/o/r/check-runs/5")
    assert stale[2]["conclusion"] == "neutral"
    method, path, body = create
    assert (method, path) == ("POST", "repos/o/r/check-runs")
    assert (body["status"], body["conclusion"], body["head_sha"]) == (
        "completed",
        "failure",
        "a" * 40,
    )
    assert len(body["output"]["summary"]) == mod.CHECK_SUMMARY_LIMIT
    assert [len(c[2]["output"]["annotations"]) for c in [create, *appends]] == [
        50,
        50,
        20,
    ]
    assert {c[1] for c in appends} == {"repos/o/r/check-runs/9"}


@pytest.mark.parametrize(
    ("state", "fields"),
    [
        ("pending", {"status": "in_progress"}),
        ("error", {"status": "completed", "conclusion": "neutral"}),
        ("success", {"status": "completed", "conclusion": "success"}),
    ],
)
def test_post_check_run_states(monkeypatch, state, fields):
    calls = []
    monkeypatch.setattr(mod, "_gh_items", lambda path, jq: [])
    monkeypatch.setattr(
        mod, "_gh_json", lambda method, path, body: calls.append(body) or {"id": 1}
    )
    mod.post_check_run("o/r", "a" * 40, state, "title", "summary", [])
    (body,) = calls
    assert {k: body[k] for k in ("status", "conclusion") if k in body} == fields
    assert body["output"]["annotations"] == []
