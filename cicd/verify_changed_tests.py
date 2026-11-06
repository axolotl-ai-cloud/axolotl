#!/usr/bin/env python
"""Check that every test a PR adds or edits ran in some CI job for the PR head.

Changed tests come from the PR diff, per test function:

* A test whose lines (decorators included) the diff touches, on either side.
* A test that uses a helper, fixture, constant or import the diff touches, found by
  name within the same file. A touched autouse fixture, setup method, class body or
  other module-level statement counts every test in its scope.
* Every test in a new or renamed file.

Test files that no longer parse, conftest.py and other support modules under ``tests/``
are listed, but the tests that depend on support modules are not traced.

Evidence is the ``pytest -v`` outcome line each job prints per test, read from the
logs of every job in the Tests, Slow CPU and Slow GPU runs for the PR head, all
attempts included. A test executed when some job reports it passed, failed, errored,
xfailed or xpassed. When a test never executed, or some of its parametrizations
only skipped, the report names the label and lane that would run it.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import subprocess  # nosec B404
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

REPO_ROOT = Path(__file__).resolve().parents[1]
TEST_WORKFLOWS = ("Tests", "Slow CPU", "Slow GPU")
CHECK_NAME = "Changed tests executed"
CHECK_CONCLUSIONS = {"success": "success", "failure": "failure", "error": "neutral"}
CHECK_SUMMARY_LIMIT = 65535
# the Checks API takes 50 annotations per request
ANNOTATION_BATCH = 50
MAX_ANNOTATIONS = 250
# GitHub keeps 10 warning commands per step; the verdict takes one
MAX_COMMAND_ANNOTATIONS = 9
TEST_FILE_GLOBS = ("test_*.py", "*_test.py")
# Jobs that never run pytest; their logs are not downloaded
NON_TEST_JOB_RE = re.compile(
    r"Modal image build|pre-commit|Prefetch S3|path filter|Select e2e tests|"
    r"check-gpu-label|gate-skip-e2e|cleanup|Changed tests report"
)
GPU_JOB_PREFIXES = ("docker-e2e /", "multi-gpu-e2e /", "gpu-slow /")
MAX_LISTED = 150

OUTCOMES = ("PASSED", "FAILED", "SKIPPED", "XFAIL", "XPASS", "ERROR")
EXECUTED = frozenset(OUTCOMES) - {"SKIPPED"}
_OUTCOME = "|".join(OUTCOMES)
_TIMESTAMP_RE = re.compile(r"^\d{4}-\d\d-\d\dT[\d:.]+Z ")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# pytest-xdist: "[gw0] [ 50%] PASSED tests/x.py::test_y"
_XDIST_RE = re.compile(
    rf"^\[gw\d+\] \[\s*\d+%\] ({_OUTCOME}) (tests/\S*?\.py::.+?)\s*$"
)
# serial: "tests/x.py::test_y PASSED [ 50%]"; under -s the test's output sits in between
_SERIAL_RE = re.compile(
    rf"^(tests/\S*?\.py::.+?) ({_OUTCOME})(?: \(.*\))?\s+\[\s*\d+%\]\s*$"
)
_SERIAL_START_RE = re.compile(r"^(tests/\S*?\.py::\S+)")
_SERIAL_END_RE = re.compile(rf"({_OUTCOME})(?: \(.*\))?\s+\[\s*\d+%\]\s*$")
_SECTION_RE = re.compile(r"^(=|_){3,} ")

# "skip" is the unconditional mark, kept to explain tests that skip in every lane
LANE_MARKERS = frozenset(
    {
        "slow",
        "gpu",
        "gpu_blackwell",
        "nvfp4",
        "distributed_cpu",
        "nf4_distributed",
        "skip",
    }
)
# Mirror tests/conftest.py, which auto-applies gpu / gpu_blackwell from skip reasons
GPU_REASON_RE = re.compile(r"\b(?:cuda|gpu)s?\b", re.IGNORECASE)
BLACKWELL_REASON_RE = re.compile(
    r"\bblackwell\b|\bsm[_ ]?(?:>=\s*)?1[0-2]\d\b", re.IGNORECASE
)
PRE_BLACKWELL_ARCH_RE = re.compile(
    r"\bsm[_ ]?[5-9]\d\b|\b(?:ampere|ada|hopper|[ah]100|l40s?)\b", re.IGNORECASE
)
SETUP_NAMES = frozenset(
    {
        "setup_method",
        "teardown_method",
        "setup_class",
        "teardown_class",
        "setup_module",
        "teardown_module",
        "setup_function",
        "teardown_function",
        "setUp",
        "tearDown",
        "setUpClass",
        "tearDownClass",
        "asyncSetUp",
        "asyncTearDown",
    }
)

LABEL_GPU = "run-gpu-tests"
LABEL_GPU_SLOW = "run-gpu-slow"
LABEL_CPU_SLOW = "run-cpu-slow"
# cicd/gpu_slow.sh suites
GPU_SLOW_PATHS = (
    "tests/e2e/kernels/",
    "tests/utils/lora/test_merge_lora.py",
    "tests/e2e/multigpu/test_ringmaster.py",
    "tests/e2e/multigpu/solo/test_twotower_expert_parallel.py",
)
# slow tests that cicd/cicd_cuda_kernels.sh and cicd/nf4.sh select by path
SLOW_IN_GPU_LANES = (
    "tests/monkeypatch/test_fla_mamba.py",
    "tests/monkeypatch/test_nf4_loading.py",
)


def _git(*args: str, cwd: Path = REPO_ROOT) -> str:
    return subprocess.check_output(  # nosec B603 B607
        ["git", *args], cwd=cwd, text=True, stderr=subprocess.PIPE
    )


def _gh(*args: str) -> str:
    return subprocess.check_output(  # nosec B603 B607
        ["gh", *args], text=True, stderr=subprocess.PIPE
    )


def _gh_items(path: str, jq: str) -> list[dict]:
    out = _gh("api", "--paginate", path, "--jq", jq)
    return [json.loads(line) for line in out.splitlines() if line.strip()]


# --- log evidence -----------------------------------------------------------


def parse_log(text: str) -> dict[str, set[str]]:
    """Map each nodeid with a ``pytest -v`` outcome line in a job log to its outcomes."""
    outcomes: dict[str, set[str]] = defaultdict(set)
    pending = None
    for raw in text.splitlines():
        line = _ANSI_RE.sub("", _TIMESTAMP_RE.sub("", raw, count=1)).rstrip()
        if match := _XDIST_RE.match(line):
            outcomes[match.group(2)].add(match.group(1))
        elif match := _SERIAL_RE.match(line):
            outcomes[match.group(1)].add(match.group(2))
            pending = None
        elif match := _SERIAL_START_RE.match(line):
            pending = match.group(1)
        elif _SECTION_RE.match(line):
            pending = None
        elif pending and (match := _SERIAL_END_RE.search(line)):
            outcomes[pending].add(match.group(1))
            pending = None
    return dict(outcomes)


@dataclass
class Job:
    id: int
    name: str
    workflow: str
    attempt: int
    status: str
    conclusion: str | None

    @property
    def label(self) -> str:
        suffix = f" (attempt {self.attempt})" if self.attempt > 1 else ""
        return f"{self.workflow} › {self.name}{suffix}"

    @property
    def gpu(self) -> bool:
        return self.workflow == "Slow GPU" or self.name.startswith(GPU_JOB_PREFIXES)

    @property
    def has_tests(self) -> bool:
        return (
            self.status == "completed"
            and self.conclusion not in ("skipped", None)
            and not NON_TEST_JOB_RE.search(self.name)
        )


@dataclass
class Evidence:
    jobs: list[Job] = field(default_factory=list)
    pending: list[str] = field(default_factory=list)
    unreadable: list[str] = field(default_factory=list)
    gpu_jobs: set[str] = field(default_factory=set)
    # nodeid without parameters -> nodeid -> job label -> outcomes
    by_base: dict[str, dict[str, dict[str, set[str]]]] = field(
        default_factory=lambda: defaultdict(lambda: defaultdict(dict))
    )

    def add(self, job: Job, outcomes: dict[str, set[str]]) -> None:
        if job.gpu:
            self.gpu_jobs.add(job.label)
        # two runs of a workflow for one head share job labels
        for nodeid, seen in outcomes.items():
            jobs = self.by_base[nodeid.split("[", 1)[0]][nodeid]
            jobs.setdefault(job.label, set()).update(seen)


def gather_evidence(
    repo: str,
    head_sha: str,
    log_cache: Path | None = None,
    current_run: int | None = None,
) -> Evidence:
    """Outcomes from every test job for ``head_sha``; ``current_run`` is not pending."""
    evidence = Evidence()
    runs = _gh_items(
        f"repos/{repo}/actions/runs?head_sha={head_sha}&per_page=100",
        ".workflow_runs[] | {id, name, status}",
    )
    for run in runs:
        if run["name"] not in TEST_WORKFLOWS:
            continue
        if run["status"] != "completed" and run["id"] != current_run:
            evidence.pending.append(run["name"])
        for raw in _gh_items(
            f"repos/{repo}/actions/runs/{run['id']}/jobs?filter=all&per_page=100",
            ".jobs[] | {id, name, run_attempt, status, conclusion}",
        ):
            evidence.jobs.append(
                Job(
                    id=raw["id"],
                    name=raw["name"],
                    workflow=run["name"],
                    attempt=raw["run_attempt"],
                    status=raw["status"],
                    conclusion=raw["conclusion"],
                )
            )

    def fetch(job: Job) -> tuple[Job, str | None, str]:
        cached = log_cache / f"{job.id}.log" if log_cache else None
        if cached and cached.exists():
            return job, cached.read_text(encoding="utf-8", errors="replace"), ""
        error = ""
        for attempt in range(3):
            try:
                raw = subprocess.check_output(  # nosec B603 B607
                    ["gh", "api", f"repos/{repo}/actions/jobs/{job.id}/logs"],
                    stderr=subprocess.PIPE,
                )
                break
            except subprocess.CalledProcessError as exc:
                lines = (
                    exc.stderr.decode("utf-8", errors="replace").strip().splitlines()
                )
                error = lines[-1] if lines else f"gh exited {exc.returncode}"
                time.sleep(2 * (attempt + 1))
        else:
            return job, None, error
        text = raw.decode("utf-8", errors="replace")
        if cached:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(text, encoding="utf-8")
        return job, text, ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        for job, text, error in pool.map(
            fetch, [j for j in evidence.jobs if j.has_tests]
        ):
            if text is not None:
                evidence.add(job, parse_log(text))
            # a job cancelled before it started has no log
            elif job.conclusion != "cancelled":
                print(f"cannot read the log of {job.label}: {error}", file=sys.stderr)
                evidence.unreadable.append(f"{job.label} ({error})")
    return evidence


# --- changed tests ------------------------------------------------------------

_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class FileChange:
    old_path: str | None
    new_path: str | None
    renamed: bool = False
    old_lines: set[int] = field(default_factory=set)
    new_lines: set[int] = field(default_factory=set)


def parse_diff(text: str) -> list[FileChange]:
    """Parse ``git diff -U0`` output into per-file changed line numbers."""
    changes: list[FileChange] = []
    current: FileChange | None = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            current = FileChange(None, None)
            changes.append(current)
        elif current is None:
            continue
        elif line.startswith("rename from "):
            current.old_path, current.renamed = line[len("rename from ") :], True
        elif line.startswith("rename to "):
            current.new_path = line[len("rename to ") :]
        elif line.startswith("--- "):
            current.old_path = None if line == "--- /dev/null" else line[6:]
        elif line.startswith("+++ "):
            current.new_path = None if line == "+++ /dev/null" else line[6:]
        elif match := _HUNK_RE.match(line):
            old_start, old_len, new_start, new_len = match.groups()
            old_n = 1 if old_len is None else int(old_len)
            new_n = 1 if new_len is None else int(new_len)
            current.old_lines.update(range(int(old_start), int(old_start) + old_n))
            current.new_lines.update(range(int(new_start), int(new_start) + new_n))
    return changes


def is_test_file(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return path.startswith("tests/") and any(
        fnmatch.fnmatch(name, glob) for glob in TEST_FILE_GLOBS
    )


_FuncDef = ast.FunctionDef | ast.AsyncFunctionDef
_Def = _FuncDef | ast.ClassDef
_DEF_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _dotted(node: ast.AST) -> str:
    if isinstance(node, ast.Attribute):
        return f"{_dotted(node.value)}.{node.attr}"
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Call):
        return _dotted(node.func)
    return ""


def _span(node: _Def) -> tuple[int, int]:
    start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
    return start, node.end_lineno or node.lineno


def _class_line_is_inert(node: ast.ClassDef, line: int) -> bool:
    """A blank line, comment or docstring in a class body, outside its statements."""
    if line < node.body[0].lineno:
        return False
    for stmt in node.body:
        if stmt.lineno <= line <= (stmt.end_lineno or stmt.lineno):
            return isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant)
    return True


def _references(node: ast.AST) -> set[str]:
    """Names a definition may use: identifiers, attributes, fixture args, usefixtures."""
    names: set[str] = set()
    for sub in ast.walk(node):
        if isinstance(sub, ast.Name):
            names.add(sub.id)
        elif isinstance(sub, ast.Attribute):
            names.add(sub.attr)
        elif isinstance(sub, ast.arg):
            names.add(sub.arg)
        elif isinstance(sub, ast.Call) and _dotted(sub.func).endswith("usefixtures"):
            names.update(
                arg.value
                for arg in sub.args
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
            )
    return names


def _fixture(node: _Def) -> tuple[str | None, bool]:
    """(requested name, autouse) when ``node`` is a pytest fixture."""
    for dec in node.decorator_list:
        if not _dotted(dec).endswith("fixture"):
            continue
        name, autouse = node.name, False
        for kw in dec.keywords if isinstance(dec, ast.Call) else []:
            if kw.arg == "name" and isinstance(kw.value, ast.Constant):
                name = str(kw.value.value)
            elif kw.arg == "autouse" and isinstance(kw.value, ast.Constant):
                autouse = bool(kw.value.value)
        return name, autouse
    return None, False


class ModuleIndex:
    """The tests pytest would collect from one file, and what each one uses."""

    def __init__(self, path: str, source: str):
        self.path = path
        self.tree = ast.parse(source)
        self.defs: dict[tuple[str, ...], _Def] = {}
        self._index(self.tree.body, ())
        self.assigns = {
            target.id: stmt.value
            for stmt in self.tree.body
            if isinstance(stmt, ast.Assign)
            for target in stmt.targets
            if isinstance(target, ast.Name)
        }
        # collected test qualname -> qualname of the def that implements it
        self.owners: dict[tuple[str, ...], tuple[str, ...]] = {}
        for qualname, node in self.defs.items():
            if len(qualname) == 1 and not isinstance(node, ast.ClassDef):
                if node.name.startswith("test"):
                    self.owners[qualname] = qualname
            elif isinstance(node, ast.ClassDef) and self._collected_chain(qualname):
                for name, owner in self._methods(qualname).items():
                    if name.startswith("test"):
                        self.owners[(*qualname, name)] = owner

    def _index(self, body: list[ast.stmt], prefix: tuple[str, ...]) -> None:
        for node in body:
            if isinstance(node, _DEF_TYPES):
                self.defs[(*prefix, node.name)] = node
                if isinstance(node, ast.ClassDef):
                    self._index(node.body, (*prefix, node.name))

    def _collected_chain(self, qualname: tuple[str, ...]) -> bool:
        for i in range(1, len(qualname) + 1):
            node = self.defs.get(qualname[:i])
            if not isinstance(node, ast.ClassDef) or not node.name.startswith("Test"):
                return False
            if any(
                isinstance(n, ast.FunctionDef) and n.name == "__init__"
                for n in node.body
            ):
                return False
        return True

    def _methods(
        self, qualname: tuple[str, ...], seen: frozenset = frozenset()
    ) -> dict[str, tuple[str, ...]]:
        """Method name -> defining qualname, with bases defined in this module."""
        node = self.defs[qualname]
        assert isinstance(node, ast.ClassDef)
        methods: dict[str, tuple[str, ...]] = {}
        for base in node.bases:
            base_q = (_dotted(base).rsplit(".", 1)[-1],)
            if isinstance(self.defs.get(base_q), ast.ClassDef) and base_q not in seen:
                for name, owner in self._methods(base_q, seen | {qualname}).items():
                    methods.setdefault(name, owner)
        for child in node.body:
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                methods[child.name] = (*qualname, child.name)
        return methods

    def _innermost(self, line: int) -> tuple[str, ...] | None:
        best = None
        for qualname, node in self.defs.items():
            start, end = _span(node)
            if start <= line <= end and (best is None or len(qualname) > len(best)):
                best = qualname
        return best

    def _scope_tests(self, scope: tuple[str, ...]) -> set[tuple[str, ...]]:
        """Tests defined in or inherited from ``scope`` (empty scope = the module)."""
        n = len(scope)
        return {
            test
            for test, owner in self.owners.items()
            if test[:n] == scope or owner[:n] == scope
        }

    def _test_references(self, test: tuple[str, ...]) -> set[str]:
        names = _references(self.defs[self.owners[test]])
        for i in range(1, len(test)):
            cls = self.defs[test[:i]]
            assert isinstance(cls, ast.ClassDef)
            for expr in [*cls.decorator_list, *cls.bases]:
                names |= _references(expr)
        return names

    def touched(self, lines: set[int]) -> tuple[set[tuple[str, ...]], set[str], bool]:
        """(tests edited directly, symbols edited, whole module affected)."""
        tests: set[tuple[str, ...]] = set()
        symbols: set[str] = set()
        for line in lines:
            qualname = self._innermost(line)
            if qualname is None:
                stmt_symbols = self._module_statement_symbols(line)
                if stmt_symbols is None:
                    return tests, symbols, True
                symbols |= stmt_symbols
                continue
            node = self.defs[qualname]
            implemented = {t for t, owner in self.owners.items() if owner == qualname}
            if implemented:
                tests |= implemented
                continue
            if isinstance(node, ast.ClassDef):
                # the class header or a class-level statement, not one of its methods
                if not _class_line_is_inert(node, line):
                    tests |= self._scope_tests(qualname)
                    symbols.add(node.name)
                continue
            fixture_name, autouse = _fixture(node)
            if autouse or node.name in SETUP_NAMES:
                tests |= self._scope_tests(qualname[:-1])
                if not qualname[:-1]:
                    return tests, symbols, True
            # a helper method also changes the classes that carry it
            symbols.update(qualname)
            if fixture_name:
                symbols.add(fixture_name)
        return tests, symbols, False

    def _module_statement_symbols(self, line: int) -> set[str] | None:
        for stmt in self.tree.body:
            if not stmt.lineno <= line <= (stmt.end_lineno or stmt.lineno):
                continue
            if isinstance(stmt, (ast.Import, ast.ImportFrom)):
                # changed_tests_for_file diffs the bindings instead
                return None if any(a.name == "*" for a in stmt.names) else set()
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    stmt.targets if isinstance(stmt, ast.Assign) else [stmt.target]
                )
                names = {
                    n.id
                    for t in targets
                    for n in ast.walk(t)
                    if isinstance(n, ast.Name)
                }
                return None if "pytestmark" in names else names
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                return set()
            if isinstance(stmt, ast.If) and "__main__" in ast.unparse(stmt.test):
                return set()
            return None
        return set()

    def imports(self) -> set[tuple[str, str, str]]:
        """(bound name, module, imported name) for each module-level import."""
        bindings = set()
        for stmt in self.tree.body:
            if isinstance(stmt, ast.Import):
                for alias in stmt.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    bindings.add((bound, alias.name, ""))
            elif isinstance(stmt, ast.ImportFrom):
                module = "." * stmt.level + (stmt.module or "")
                for alias in stmt.names:
                    bindings.add((alias.asname or alias.name, module, alias.name))
        return bindings

    def users_of(self, symbols: set[str]) -> dict[tuple[str, ...], set[str]]:
        """Tests that use ``symbols`` directly or through helpers in this module."""
        symbols = set(symbols)
        helpers = []
        for qualname, node in self.defs.items():
            if qualname in self.owners.values():
                continue
            if isinstance(node, ast.ClassDef):
                if self._collected_chain(qualname):
                    continue
                # methods are helpers of their own; a class uses its header and attributes
                parts: list[ast.AST] = [*node.decorator_list, *node.bases]
                parts += [n for n in node.body if not isinstance(n, _DEF_TYPES)]
                used = set().union(*map(_references, parts)) if parts else set()
            else:
                used = _references(node)
            helpers.append(({node.name, _fixture(node)[0] or node.name}, used))
        grew = True
        while grew:
            grew = False
            for provided, used in helpers:
                if provided - symbols and used & symbols:
                    symbols |= provided
                    grew = True
        users = {}
        for test in self.owners:
            if used := self._test_references(test) & symbols:
                users[test] = used
        return users

    def markers(self, test: tuple[str, ...]) -> set[str]:
        """Lane markers applied statically to ``test``, plus ``gpu`` for CUDA skip gates."""
        exprs: list[ast.AST] = list(self.defs[self.owners[test]].decorator_list)
        scopes: list[list[ast.stmt]] = [self.tree.body]
        for i in range(1, len(test)):
            cls = self.defs[test[:i]]
            assert isinstance(cls, ast.ClassDef)
            exprs.extend(cls.decorator_list)
            scopes.append(cls.body)
        for body in scopes:
            for stmt in body:
                if isinstance(stmt, ast.Assign) and any(
                    isinstance(t, ast.Name) and t.id == "pytestmark"
                    for t in stmt.targets
                ):
                    exprs.append(stmt.value)
        return self._marks(exprs, set())

    def _marks(self, exprs: list[ast.AST], seen: set[str]) -> set[str]:
        found: set[str] = set()
        for expr in exprs:
            for sub in ast.walk(expr):
                if isinstance(sub, ast.Attribute) and _dotted(sub.value).endswith(
                    "mark"
                ):
                    found.add(sub.attr)
                elif isinstance(sub, ast.Call) and _dotted(sub.func).rsplit(".")[
                    -1
                ] in ("skip", "skipif"):
                    for arg in [*sub.args, *(kw.value for kw in sub.keywords)]:
                        found |= _skip_gate_marks(arg)
                elif (
                    isinstance(sub, ast.Name)
                    and sub.id in self.assigns
                    and sub.id not in seen
                ):
                    seen.add(sub.id)
                    found |= self._marks([self.assigns[sub.id]], seen)
        return found & LANE_MARKERS


def _skip_gate_marks(expr: ast.AST) -> set[str]:
    found = set()
    for leaf in ast.walk(expr):
        if isinstance(leaf, ast.Attribute) and leaf.attr == "cuda":
            found.add("gpu")
        elif isinstance(leaf, ast.Constant) and isinstance(leaf.value, str):
            if BLACKWELL_REASON_RE.search(leaf.value) and not (
                PRE_BLACKWELL_ARCH_RE.search(leaf.value)
            ):
                found.add("gpu_blackwell")
            elif GPU_REASON_RE.search(leaf.value):
                found.add("gpu")
    return found


@dataclass
class ChangedTest:
    path: str
    qualname: tuple[str, ...]
    why: str
    markers: set[str]
    line: int = 1

    @property
    def nodeid(self) -> str:
        return "::".join((self.path, *self.qualname))


@dataclass
class ChangeSet:
    tests: list[ChangedTest] = field(default_factory=list)
    support_files: list[str] = field(default_factory=list)
    unparsable: list[str] = field(default_factory=list)


def _show(rev: str, path: str) -> str:
    return _git("show", f"{rev}:{path}")


def _parse(path: str, source: str) -> ModuleIndex | None:
    try:
        return ModuleIndex(path, source)
    except (SyntaxError, ValueError):
        return None


def changed_tests_for_file(
    change: FileChange, old_source: str | None, new_source: str
) -> list[ChangedTest] | None:
    """Tests a file change affects; None when the new file does not parse."""
    assert change.new_path is not None
    new = _parse(change.new_path, new_source)
    if new is None:
        return None
    old = (
        _parse(change.old_path, old_source) if old_source and change.old_path else None
    )

    whys: dict[tuple[str, ...], str] = {}
    if old is None or change.renamed:
        reason = "renamed file" if change.renamed else "new file"
        whys = dict.fromkeys(new.owners, reason)
    else:
        edited, symbols, whole = new.touched(change.new_lines)
        old_edited, old_symbols, old_whole = old.touched(change.old_lines)
        if whole or old_whole:
            whys = dict.fromkeys(new.owners, "module-level change")
        edited |= old_edited & new.owners.keys()
        symbols |= {bound for bound, *_ in new.imports() ^ old.imports()}
        for test in edited:
            whys.setdefault(test, "edited")
        for test, used in new.users_of(symbols | old_symbols).items():
            whys.setdefault(test, "uses " + ", ".join(sorted(used)[:3]))
    return [
        ChangedTest(
            change.new_path,
            test,
            why,
            new.markers(test),
            new.defs[new.owners[test]].lineno,
        )
        for test, why in sorted(whys.items())
    ]


def collect_changes(base: str, head: str) -> ChangeSet:
    diff = _git(
        "diff",
        "-U0",
        "-M",
        "--no-color",
        "--no-ext-diff",
        "--no-textconv",
        "--src-prefix=a/",
        "--dst-prefix=b/",
        base,
        head,
        "--",
        "tests/",
    )
    result = ChangeSet()
    for change in parse_diff(diff):
        path = change.new_path
        if path is None or not path.endswith(".py"):
            continue
        if not is_test_file(path):
            result.support_files.append(path)
            continue
        old_source = _show(base, change.old_path) if change.old_path else None
        tests = changed_tests_for_file(change, old_source, _show(head, path))
        if tests is None:
            result.unparsable.append(path)
        else:
            result.tests.extend(tests)
    return result


# --- verdicts -------------------------------------------------------------------


@dataclass(frozen=True)
class Route:
    label: str | None
    lane: str


CPU_ROUTE = Route(
    None, "Expected in CPU PyTest without a label; check its selection in tests.yml"
)
SKIP_MARK_ROUTE = Route(None, "Skipped by an unconditional skip mark")
SKIPPED_ON_GPU_ROUTE = Route(
    None, "Skipped on every runner that collected them, GPU jobs included"
)
CUDA_GATE_ROUTE = Route(LABEL_GPU, "GPU, if the CPU skip is a CUDA gate")
CPU_SKIP_ROUTE = Route(
    None,
    "Skipped in every CPU job, and no GPU job collects them; check the skip reasons",
)
GPU_PATHS = ("tests/e2e/", "tests/integrations/kernels/")


def route_for(path: str, markers: set[str]) -> Route:
    """The PR label and lane that would run a test, from its path and static markers."""
    gpu = bool(markers & {"gpu", "gpu_blackwell", "nvfp4"}) or path.startswith(
        GPU_PATHS
    )
    if "slow" in markers:
        if path.startswith(SLOW_IN_GPU_LANES):
            return Route(LABEL_GPU, "GPU kernels / NF4")
        if gpu and path.startswith(GPU_SLOW_PATHS):
            return Route(LABEL_GPU_SLOW, "Slow GPU")
        if not gpu and path.startswith("tests/integrations/"):
            return Route(LABEL_CPU_SLOW, "Slow CPU")
        return Route(None, "No PR lane runs slow tests at this path")
    if "nvfp4" in markers or path.startswith("tests/e2e/multigpu/"):
        return Route(LABEL_GPU, "multi-GPU e2e; the PR must also touch its path filter")
    if "gpu_blackwell" in markers:
        return Route(LABEL_GPU, "B200")
    if path.startswith(("tests/e2e/kernels/", "tests/integrations/kernels/")):
        return Route(LABEL_GPU, "GPU kernels")
    if gpu:
        return Route(LABEL_GPU, "single-GPU e2e")
    if markers & {"distributed_cpu", "nf4_distributed"}:
        return Route(
            None, "Expected in Distributed CPU, which runs only the files it lists"
        )
    return CPU_ROUTE


@dataclass
class Verdict:
    test: ChangedTest
    status: str  # ran | partial | skipped | not_run
    variants: dict[str, dict[str, set[str]]]
    failed: bool
    route: Route | None


def judge(test: ChangedTest, evidence: Evidence) -> Verdict:
    variants = evidence.by_base.get(test.nodeid, {})
    executed = {
        nodeid
        for nodeid, jobs in variants.items()
        if any(seen & EXECUTED for seen in jobs.values())
    }
    if not variants:
        status = "not_run"
    elif len(executed) == len(variants):
        status = "ran"
    else:
        status = "partial" if executed else "skipped"
    failed = any(
        seen & {"FAILED", "ERROR"}
        for jobs in variants.values()
        for seen in jobs.values()
    )
    route = None if status == "ran" else route_for(test.path, test.markers)
    if status in ("partial", "skipped"):
        skipped_in = {
            job
            for nodeid, jobs in variants.items()
            if nodeid not in executed
            for job in jobs
        }
        if "skip" in test.markers:
            route = SKIP_MARK_ROUTE
        elif skipped_in & evidence.gpu_jobs:
            route = SKIPPED_ON_GPU_ROUTE
        elif route == CPU_ROUTE:
            route = CPU_SKIP_ROUTE if evidence.gpu_jobs else CUDA_GATE_ROUTE
    return Verdict(test, status, dict(variants), failed, route)


STATUS_TITLES = {
    "ran": "✅ ran",
    "partial": "◐ ran, but some parametrizations only skipped",
    "skipped": "⏭️ only skipped",
    "not_run": "❌ not run",
}


def _action(route: Route, labels: set[str]) -> str:
    if route.label is None:
        return route.lane
    if route.label in labels:
        return f"`{route.label}` is set, but no matching job ran these ({route.lane})"
    return f"Add `{route.label}` ({route.lane})"


def _variant_summary(verdict: Verdict) -> str:
    counts: dict[str, int] = defaultdict(int)
    for jobs in verdict.variants.values():
        best = set().union(*jobs.values())
        counts["executed" if best & EXECUTED else "skipped"] += 1
    return ", ".join(f"{n} {kind}" for kind, n in sorted(counts.items()))


def _md(text: str) -> str:
    return text.replace("`", "'").replace("|", "\\|")


def render(
    pr: int | None,
    head_sha: str,
    changes: ChangeSet,
    verdicts: list[Verdict],
    evidence: Evidence,
    labels: set[str],
) -> str:
    counts = {key: sum(v.status == key for v in verdicts) for key in STATUS_TITLES}
    out = [f"## Changed tests: {counts['ran']}/{len(verdicts)} ran", ""]
    where = f"PR #{pr} · " if pr else ""
    ran_jobs = [job for job in evidence.jobs if job.has_tests]
    out.append(
        f"{where}head `{head_sha[:10]}` · {len(ran_jobs)} test jobs read · "
        f"labels: {', '.join(f'`{label}`' for label in sorted(labels)) or 'none'}"
    )
    if evidence.pending:
        out += ["", f"⏳ Still running: {', '.join(sorted(set(evidence.pending)))}"]
    if evidence.unreadable:
        out += ["", f"⚠️ Logs unavailable: {', '.join(evidence.unreadable)}"]
    out += ["", "| Status | Tests |", "| --- | ---: |"]
    out += [f"| {title} | {counts[key]} |" for key, title in STATUS_TITLES.items()]

    if failing := [v for v in verdicts if v.failed]:
        out += ["", f"### Failed or errored in some job ({len(failing)})", ""]
        out += [f"- `{_md(v.test.nodeid)}`" for v in failing[:MAX_LISTED]]

    missing = [v for v in verdicts if v.status != "ran"]
    by_route: dict[Route, list[Verdict]] = defaultdict(list)
    for verdict in missing:
        assert verdict.route is not None
        by_route[verdict.route].append(verdict)
    if missing:
        out += ["", "### Needs attention", ""]
    for route, group in sorted(by_route.items(), key=lambda kv: -len(kv[1])):
        out += [f"**{_action(route, labels)}** · {len(group)}", ""]
        for verdict in group[:MAX_LISTED]:
            detail = STATUS_TITLES[verdict.status]
            if verdict.variants:
                detail += f": {_variant_summary(verdict)}"
            out.append(
                f"- `{_md(verdict.test.nodeid)}` · {detail} · _{_md(verdict.test.why)}_"
            )
        if len(group) > MAX_LISTED:
            out.append(f"- … and {len(group) - MAX_LISTED} more")
        out.append("")

    if changes.unparsable:
        out += ["", "### Test files that do not parse", ""]
        out += [f"- `{_md(path)}`" for path in changes.unparsable]
    if changes.support_files:
        out += ["", "### Support files changed (dependent tests not traced)", ""]
        out += [f"- `{_md(path)}`" for path in changes.support_files]

    ran = [v for v in verdicts if v.status == "ran"]
    if ran:
        out += ["", f"<details><summary>Ran ({len(ran)})</summary>", ""]
        for verdict in ran[:MAX_LISTED]:
            jobs = sorted({job for jobs in verdict.variants.values() for job in jobs})
            out.append(
                f"- `{_md(verdict.test.nodeid)}` · {len(verdict.variants)} variant(s) · "
                f"{_md(', '.join(jobs[:3]))}{' …' if len(jobs) > 3 else ''}"
            )
        if len(ran) > MAX_LISTED:
            out.append(f"- … and {len(ran) - MAX_LISTED} more")
        out += ["", "</details>"]
    return "\n".join(out) + "\n"


PLAIN_STATUS = {
    "partial": "Some parametrizations only skipped",
    "skipped": "Only skipped",
    "not_run": "Not run",
}


@dataclass
class Annotation:
    path: str
    line: int
    title: str
    message: str


def annotations_for(verdicts: list[Verdict], labels: set[str]) -> list[Annotation]:
    """One warning per changed test that did not fully run, at its ``def`` line."""
    return [
        Annotation(
            v.test.path,
            v.test.line,
            f"{PLAIN_STATUS[v.status]}: {'::'.join(v.test.qualname)}",
            f"{_action(v.route, labels).replace('`', '')}. Changed: {v.test.why}.",
        )
        for v in verdicts
        if v.route is not None
    ]


def _command_value(text: str, prop: bool = False) -> str:
    text = text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    return text.replace(":", "%3A").replace(",", "%2C") if prop else text


def workflow_commands(
    annotations: list[Annotation], state: str, description: str
) -> list[str]:
    """``::warning`` lines that annotate the diff, then the verdict."""
    lines = [
        f"::warning file={_command_value(a.path, True)},line={a.line},"
        f"title={_command_value(a.title, True)}::{_command_value(a.message)}"
        for a in annotations[:MAX_COMMAND_ANNOTATIONS]
    ]
    if (extra := len(annotations) - MAX_COMMAND_ANNOTATIONS) > 0:
        description += f" ({extra} more tests in the run summary)"
    level = "notice" if state == "success" else "warning"
    title = _command_value(CHECK_NAME, True)
    lines.append(f"::{level} title={title}::{_command_value(description)}")
    return lines


def status_payload(
    verdicts: list[Verdict], evidence: Evidence, labels: set[str]
) -> tuple[str, str]:
    ran = sum(v.status == "ran" for v in verdicts)
    if not verdicts:
        state, text = "success", "No changed tests"
    elif ran == len(verdicts):
        state, text = "success", f"All {ran} changed tests ran"
    else:
        state = "failure"
        needs: dict[str, int] = defaultdict(int)
        for verdict in verdicts:
            if verdict.route is None:
                continue
            label = verdict.route.label
            key = label if label and label not in labels else verdict.status
            needs[key.replace("_", " ")] += 1
        text = f"{ran}/{len(verdicts)} changed tests ran; " + ", ".join(
            f"{key} {n}" for key, n in sorted(needs.items(), key=lambda kv: -kv[1])
        )
    if evidence.unreadable and state == "failure":
        # missing evidence, not missing tests
        state = "error"
        text = f"{len(evidence.unreadable)} job logs unreadable; " + text
    if evidence.pending:
        state = "pending"
        text = f"Waiting on {', '.join(sorted(set(evidence.pending)))}; " + text
    return state, text[:140]


def to_json(
    pr: int | None, head_sha: str, changes: ChangeSet, verdicts: list[Verdict]
) -> dict:
    return {
        "pr": pr,
        "head_sha": head_sha,
        "support_files": changes.support_files,
        "unparsable": changes.unparsable,
        "tests": [
            {
                "nodeid": v.test.nodeid,
                "why": v.test.why,
                "markers": sorted(v.test.markers),
                "status": v.status,
                "failed": v.failed,
                "route": None if v.route is None else vars(v.route),
                "variants": {
                    nodeid: {job: sorted(seen) for job, seen in jobs.items()}
                    for nodeid, jobs in v.variants.items()
                },
            }
            for v in verdicts
        ],
    }


# --- GitHub ---------------------------------------------------------------------


def find_pr(repo: str, head_sha: str) -> int | None:
    try:
        numbers = [
            item["number"]
            for item in _gh_items(
                f"repos/{repo}/commits/{head_sha}/pulls", ".[] | {number}"
            )
        ]
    except subprocess.CalledProcessError:
        numbers = []
    if not numbers:
        # commit-to-PR lookups miss fork heads; search covers them
        numbers = [
            item["number"]
            for item in _gh_items(
                f"search/issues?q=repo:{repo}+is:pr+{head_sha}", ".items[] | {number}"
            )
        ]
    for number in numbers:
        if _gh("api", f"repos/{repo}/pulls/{number}", "--jq", ".head.sha").strip() == (
            head_sha
        ):
            return number
    return None


def ensure_commits(repo: str, pr: int, shas: list[str]) -> None:
    missing = []
    for sha in shas:
        try:
            _git("cat-file", "-e", f"{sha}^{{commit}}")
        except subprocess.CalledProcessError:
            missing.append(sha)
    if missing:
        # --depth on a full clone would make it shallow
        shallow = _git("rev-parse", "--is-shallow-repository").strip() == "true"
        _git(
            "fetch",
            "--no-tags",
            *(["--depth=1"] if shallow else []),
            f"https://github.com/{repo}.git",
            f"refs/pull/{pr}/head",
            *missing,
        )


def _gh_json(method: str, path: str, body: dict) -> dict:
    out = subprocess.run(  # nosec B603 B607
        ["gh", "api", "-X", method, path, "--input", "-"],
        input=json.dumps(body),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return json.loads(out) if out.strip() else {}


def _check_output(title: str, summary: str, batch: list[Annotation]) -> dict:
    if len(summary) > CHECK_SUMMARY_LIMIT:
        note = "\n\n… truncated; the job summary has the full report\n"
        summary = summary[: CHECK_SUMMARY_LIMIT - len(note)] + note
    return {
        "title": title,
        "summary": summary,
        "annotations": [
            {
                "path": a.path,
                "start_line": a.line,
                "end_line": a.line,
                "annotation_level": "warning",
                "title": a.title,
                "message": a.message,
            }
            for a in batch
        ],
    }


def post_check_run(
    repo: str,
    sha: str,
    state: str,
    title: str,
    summary: str,
    annotations: list[Annotation],
) -> None:
    """Publish the report as a check run on ``sha``, annotating changed tests."""
    # an in-progress check from an earlier report would otherwise spin forever
    for stale in _gh_items(
        f"repos/{repo}/commits/{sha}/check-runs?check_name={quote(CHECK_NAME)}"
        "&filter=all&per_page=100",
        '.check_runs[] | select(.status != "completed"'
        ' and .app.slug == "github-actions") | {id}',
    ):
        _gh_json(
            "PATCH",
            f"repos/{repo}/check-runs/{stale['id']}",
            {
                "status": "completed",
                "conclusion": "neutral",
                "output": {
                    "title": "Superseded",
                    "summary": "A newer report replaced this check.",
                },
            },
        )
    body: dict = {"name": CHECK_NAME, "head_sha": sha}
    if state == "pending":
        body["status"] = "in_progress"
    else:
        body |= {"status": "completed", "conclusion": CHECK_CONCLUSIONS[state]}
    kept = annotations[:MAX_ANNOTATIONS]
    batches = [
        kept[i : i + ANNOTATION_BATCH] for i in range(0, len(kept), ANNOTATION_BATCH)
    ] or [[]]
    run = _gh_json(
        "POST",
        f"repos/{repo}/check-runs",
        body | {"output": _check_output(title, summary, batches[0])},
    )
    # later batches append to the annotations already on the check run
    for batch in batches[1:]:
        _gh_json(
            "PATCH",
            f"repos/{repo}/check-runs/{run['id']}",
            {"output": _check_output(title, summary, batch)},
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--repo", default="axolotl-ai-cloud/axolotl")
    parser.add_argument("--pr", type=int, help="pull request number")
    parser.add_argument("--head-sha", help="PR head commit; resolves --pr when omitted")
    parser.add_argument("--summary", type=Path, help="append the markdown report here")
    parser.add_argument("--json", type=Path, help="write the full report as JSON")
    parser.add_argument(
        "--check-run",
        action="store_true",
        help=f"publish the report as the '{CHECK_NAME}' check run",
    )
    parser.add_argument("--log-cache", type=Path, help="reuse job logs saved here")
    parser.add_argument(
        "--strict", action="store_true", help="exit 1 unless every changed test ran"
    )
    parser.add_argument(
        "--annotate",
        action="store_true",
        help="emit GitHub Actions annotations for the verdict and missing tests",
    )
    args = parser.parse_args()
    if not args.pr and not args.head_sha:
        parser.error("pass --pr or --head-sha")

    pr = args.pr or find_pr(args.repo, args.head_sha)
    if pr is None:
        print(
            f"No pull request has head {args.head_sha}; it moved on or closed, "
            "and the newer head's runs report instead",
            file=sys.stderr,
        )
        return 0
    info = json.loads(_gh("api", f"repos/{args.repo}/pulls/{pr}"))
    head_sha = args.head_sha or info["head"]["sha"]
    if head_sha != info["head"]["sha"]:
        print(f"PR #{pr} head moved past {head_sha}; reporting anyway", file=sys.stderr)
    labels = {label["name"] for label in info["labels"]}
    merge_base = _gh(
        "api",
        f"repos/{args.repo}/compare/{info['base']['sha']}...{head_sha}",
        "--jq",
        ".merge_base_commit.sha",
    ).strip()
    ensure_commits(args.repo, pr, [merge_base, head_sha])

    changes = collect_changes(merge_base, head_sha)
    # a report job inside a Tests run sees that run as in progress
    run_id = os.environ.get("GITHUB_RUN_ID")
    evidence = gather_evidence(
        args.repo, head_sha, args.log_cache, int(run_id) if run_id else None
    )
    verdicts = [judge(test, evidence) for test in changes.tests]
    report = render(pr, head_sha, changes, verdicts, evidence, labels)
    print(report)
    if args.summary:
        with args.summary.open("a", encoding="utf-8") as fh:
            fh.write(report)
    if args.json:
        args.json.write_text(
            json.dumps(to_json(pr, head_sha, changes, verdicts), indent=2) + "\n",
            encoding="utf-8",
        )
    state, description = status_payload(verdicts, evidence, labels)
    print(f"{CHECK_NAME}: {state} - {description}", file=sys.stderr)
    # while runs are pending or logs are missing, "not run" is not yet a finding
    notes = annotations_for(verdicts, labels) if state == "failure" else []
    if args.check_run:
        post_check_run(args.repo, head_sha, state, description, report, notes)
    if args.annotate:
        print("\n".join(workflow_commands(notes, state, description)))
    return 1 if args.strict and state != "success" else 0


if __name__ == "__main__":
    sys.exit(main())
