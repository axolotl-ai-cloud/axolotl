#!/usr/bin/env python
"""Select the e2e test files a diff can affect, for the experimental selected-e2e CI arms.

Every edge is derived from the tree, so new features need no registration:

* The config-key universe is every class-body annotated field under ``src/axolotl``
  that some in-scope e2e test sets. A module "reads" a key when its AST references
  the attribute, subscript or ``.get`` by that name.
* Keys set by most in-scope tests carry no selection signal. A changed module that
  reads any such key is core and runs the whole scope.
* Otherwise a changed module selects the tests that import it directly, that name
  its module path, file stem or package in a config string, or that set a key it
  reads. A module with no edges at all runs the whole scope.
* A module may declare ``__ci_config_keys__ = ("activation_offloading",)`` when the
  graph cannot see what enables it; the selector then picks exactly the tests that set
  those keys. A declared key that is not a config field runs the whole scope.
* A dependency bump in ``pyproject.toml`` selects through the importers of the
  distribution's import roots, when ``[tool.axolotl.ci.deps]`` maps the distribution
  to them. An unmapped distribution, a map entry no module imports, or any other
  pyproject change runs the whole scope.
* Changed test files select themselves. Any other change under ``tests/``, a
  deleted module, a build or CI file, or an exception in the selector itself runs
  the whole scope. Nothing here can select fewer tests than the graph supports.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import os
import re
import subprocess  # nosec B404
import sys
import tomllib
import traceback
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = "src/axolotl"
UBIQUITY_THRESHOLD = 0.5
SOURCE_UBIQUITY_THRESHOLD = 0.2
PYPROJECT = "pyproject.toml"
DEP_MAP_TABLE = ("tool", "axolotl", "ci", "deps")
DEP_SECTIONS = ("dependencies", "optional-dependencies")
RUN_ALL_GLOBS = (
    "setup.py",
    "setup.cfg",
    "requirements*.txt",
    "cicd/*",
    "cicd/**/*",
    ".github/workflows/*",
    "scripts/*install*",
    "tests/conftest.py",
    f"{SRC_ROOT}/__init__.py",
)
FORCE_ALL_TOKENS = ("[test all]", "[no filter]")
# files that cannot reach a test at runtime; anything else unmodeled runs the whole scope
INERT_GLOBS = (
    "docs/*",
    "docs/**/*",
    "examples/*",
    "examples/**/*",
    "*.md",
    "**/*.md",
    "*.qmd",
    "**/*.qmd",
    "**/LICENSE",
    "LICENSE",
    ".gitignore",
    ".pre-commit-config.yaml",
    "*.png",
    "**/*.png",
    "*.svg",
    "**/*.svg",
    "*.jpg",
    "**/*.jpg",
)
# Stems and package names too generic to identify a feature from a config string.
DECLARED_KEYS_NAME = "__ci_config_keys__"
REQUIREMENT_NAME_RE = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
GENERIC_NAMES = frozenset(
    {
        "__init__",
        "args",
        "base",
        "cli",
        "common",
        "config",
        "constants",
        "core",
        "helpers",
        "loss",
        "main",
        "model",
        "models",
        "plugin",
        "train",
        "trainer",
        "types",
        "utils",
    }
)
MIN_STRING_LEN = 3
FUZZY_MIN_LEN = 5
MIN_REGISTRY_ENTRIES = 3
REGISTRY_NAMED_FRACTION = 1 / 3


@dataclass
class Selection:
    mode: str
    tests: list[str] = field(default_factory=list)
    reason: str = ""
    explain: dict[str, set[str]] = field(default_factory=dict)


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(  # nosec B603 B607
        ["git", *args], cwd=repo, text=True, stderr=subprocess.STDOUT
    )


def _safe_parse(path: Path) -> ast.AST | None:
    try:
        return ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None


class Selector:
    def __init__(
        self, repo: Path, scope: str, exclude: tuple[str, ...] = (), head: str = "HEAD"
    ):
        self.repo = repo
        self.head = head
        self._edge_cache: dict[str, tuple[set[str] | None, str] | None] = {}
        self._registry_cache: dict[str, bool] = {}
        self._value_cache: dict[str, set[str]] = {}
        self.scope = scope.rstrip("/")
        self.exclude = tuple(e.rstrip("/") for e in exclude)

    # -- repo state --------------------------------------------------------

    def test_files(self) -> list[str]:
        out = []
        for path in sorted((self.repo / self.scope).rglob("test_*.py")):
            rel = path.relative_to(self.repo).as_posix()
            if any(rel.startswith(f"{e}/") for e in self.exclude):
                continue
            out.append(rel)
        return out

    def source_files(self) -> list[str]:
        return sorted(
            p.relative_to(self.repo).as_posix()
            for p in (self.repo / SRC_ROOT).rglob("*.py")
        )

    @staticmethod
    def module_name(rel: str) -> str:
        parts = Path(rel).with_suffix("").parts
        if parts[0] == "src":
            parts = parts[1:]
        if parts[-1] == "__init__":
            parts = parts[:-1]
        return ".".join(parts)

    def _diff_base(self, base: str | None, merge_commit: bool) -> str:
        if merge_commit:
            return f"{self.head}^1"
        assert base
        return _git(self.repo, "merge-base", base, self.head).strip()

    def changed(
        self, base: str | None, merge_commit: bool
    ) -> tuple[list[str], list[str]]:
        rev = self._diff_base(base, merge_commit)
        out = _git(self.repo, "diff", "--name-status", "--no-renames", rev, self.head)
        changed: list[str] = []
        deleted: list[str] = []
        for line in out.splitlines():
            status, _, path = line.partition("\t")
            (deleted if status.startswith("D") else changed).append(path.strip())
        return changed, deleted

    def commit_messages(self, base: str | None, merge_commit: bool) -> str:
        rev = self._diff_base(base, merge_commit)
        return _git(self.repo, "log", "--format=%B", f"{rev}..{self.head}")

    # -- AST extraction ----------------------------------------------------

    @staticmethod
    def _config_keys_and_values(tree: ast.AST) -> tuple[set[str], set[str]]:
        """Top-level keys of dict literals, and every string constant in the file.

        Configs are assembled from helper arguments and fixtures as well as literals, so
        any string may be a config value; nested dict keys are not config keys.
        """
        keys: set[str] = set()
        values: set[str] = set()
        nested: set[int] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Dict):
                for child in ast.walk(node):
                    if child is not node and isinstance(child, ast.Dict):
                        nested.add(id(child))
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                values.add(node.value)
            elif isinstance(node, ast.Dict) and id(node) not in nested:
                for key in node.keys:
                    if isinstance(key, ast.Constant) and isinstance(key.value, str):
                        keys.add(key.value)
        return keys, values - keys

    @staticmethod
    def _import_targets(node: ast.Import | ast.ImportFrom, package: str) -> set[str]:
        if isinstance(node, ast.Import):
            return {alias.name for alias in node.names}
        if node.level == 0:
            base = node.module or ""
        else:
            parts = package.split(".")
            parts = parts[: len(parts) - (node.level - 1)] if node.level > 1 else parts
            base = ".".join(parts + ([node.module] if node.module else []))
        if not base:
            return set()
        return {base} | {f"{base}.{alias.name}" for alias in node.names}

    def _imports(self, tree: ast.AST, package: str = "") -> set[str]:
        mods: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                mods |= self._import_targets(node, package)
        return mods

    @staticmethod
    def _declared_keys(tree: ast.AST) -> tuple[str, ...] | None:
        for node in getattr(tree, "body", []):
            if not isinstance(node, ast.Assign):
                continue
            if not any(
                isinstance(t, ast.Name) and t.id == DECLARED_KEYS_NAME
                for t in node.targets
            ):
                continue
            keys: list[str] = []
            elts = (
                node.value.elts if isinstance(node.value, (ast.Tuple, ast.List)) else []
            )
            for elt in elts:
                if not (isinstance(elt, ast.Constant) and isinstance(elt.value, str)):
                    return ()
                keys.append(elt.value)
            return tuple(keys)
        return None

    def _scan_source(
        self, tree: ast.AST, package: str
    ) -> tuple[set[str], set[str], set[str]]:
        """One walk: class fields, imports, and referenced names."""
        fields: set[str] = set()
        imports: set[str] = set()
        names: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                for stmt in node.body:
                    if isinstance(stmt, ast.AnnAssign) and isinstance(
                        stmt.target, ast.Name
                    ):
                        fields.add(stmt.target.id)
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports |= self._import_targets(node, package)
            elif isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Subscript) and isinstance(
                node.slice, ast.Constant
            ):
                if isinstance(node.slice.value, str):
                    names.add(node.slice.value)
            elif isinstance(node, ast.Call):
                func = node.func
                is_get = isinstance(func, ast.Attribute) and func.attr in {"get", "pop"}
                is_getattr = isinstance(func, ast.Name) and func.id in {
                    "getattr",
                    "hasattr",
                }
                if is_get and node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        names.add(node.args[0].value)
                if (
                    is_getattr
                    and len(node.args) > 1
                    and isinstance(node.args[1], ast.Constant)
                ):
                    if isinstance(node.args[1].value, str):
                        names.add(node.args[1].value)
        return fields, imports, names

    @staticmethod
    def _referenced_names(tree: ast.AST) -> set[str]:
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                names.add(node.attr)
            elif isinstance(node, ast.Subscript) and isinstance(
                node.slice, ast.Constant
            ):
                if isinstance(node.slice.value, str):
                    names.add(node.slice.value)
            elif isinstance(node, ast.Call):
                func = node.func
                is_get = isinstance(func, ast.Attribute) and func.attr in {"get", "pop"}
                is_getattr = isinstance(func, ast.Name) and func.id in {
                    "getattr",
                    "hasattr",
                }
                if is_get and node.args and isinstance(node.args[0], ast.Constant):
                    if isinstance(node.args[0].value, str):
                        names.add(node.args[0].value)
                if (
                    is_getattr
                    and len(node.args) > 1
                    and isinstance(node.args[1], ast.Constant)
                ):
                    if isinstance(node.args[1].value, str):
                        names.add(node.args[1].value)
        return names

    # -- graph -------------------------------------------------------------

    def build(self) -> None:
        tests = self.test_files()
        sources = self.source_files()
        conftests = {
            p.relative_to(self.repo).as_posix(): _safe_parse(p)
            for p in (self.repo / self.scope).rglob("conftest.py")
        }

        # helper modules next to the tests (worker scripts, parity probes) carry configs too
        helpers: dict[str, tuple[set[str], set[str], set[str]]] = {}
        for path in (self.repo / self.scope).rglob("*.py"):
            rel = path.relative_to(self.repo).as_posix()
            if path.name.startswith("test_") or path.name == "__init__.py":
                continue
            tree = _safe_parse(path)
            if tree is None:
                continue
            hkeys, hstrings = self._config_keys_and_values(tree)
            helpers[rel] = (hkeys, hstrings, self._imports(tree))

        def helper_refs(rel: str, strings: set[str], imports: set[str]) -> set[str]:
            """Helpers this file names by filename or stem, or imports by module."""
            here = Path(rel).parent
            found = set()
            for helper in helpers:
                hpath = Path(helper)
                hmod = self.module_name(helper)
                if (
                    hpath.name in strings
                    or hpath.stem in strings
                    or helper in strings
                    or any(imp == hmod or imp.startswith(hmod + ".") for imp in imports)
                    or (
                        hpath.parent == here
                        and any(imp.endswith("." + hpath.stem) for imp in imports)
                    )
                ):
                    found.add(helper)
            return found

        test_keys: dict[str, set[str]] = {}
        test_strings: dict[str, set[str]] = {}
        test_imports: dict[str, set[str]] = {}
        helper_tests: dict[str, set[str]] = defaultdict(set)
        for rel in tests:
            tree = _safe_parse(self.repo / rel)
            keys: set[str] = set()
            strings: set[str] = set()
            imports: set[str] = set()
            if tree is not None:
                keys, strings = self._config_keys_and_values(tree)
                imports = self._imports(tree)
            # fixtures merge into the config, so a conftest on the path contributes its keys
            for cpath, ctree in conftests.items():
                if ctree is not None and rel.startswith(cpath[: -len("conftest.py")]):
                    ckeys, cstrings = self._config_keys_and_values(ctree)
                    keys |= ckeys
                    strings |= cstrings
            seen: set[str] = set()
            frontier = helper_refs(rel, strings, imports)
            while frontier:
                helper = frontier.pop()
                if helper in seen:
                    continue
                seen.add(helper)
                hkeys, hstrings, himports = helpers[helper]
                keys |= hkeys
                strings |= hstrings
                imports |= himports
                frontier |= helper_refs(helper, hstrings, himports) - seen
            test_keys[rel] = keys
            test_strings[rel] = strings
            test_imports[rel] = imports
            for helper in seen:
                helper_tests[helper].add(rel)

        all_fields: set[str] = set()
        src_trees = {}
        src_names: dict[str, set[str]] = {}
        src_imports: dict[str, set[str]] = {}
        src_declared: dict[str, tuple[str, ...]] = {}
        by_module = {self.module_name(rel): rel for rel in sources}
        importers: dict[str, set[str]] = defaultdict(set)
        for rel in sources:
            tree = _safe_parse(self.repo / rel)
            src_trees[rel] = tree
            if tree is None:
                continue
            module = self.module_name(rel)
            package = (
                module if rel.endswith("__init__.py") else module.rpartition(".")[0]
            )
            fields, imports, names = self._scan_source(tree, package)
            all_fields |= fields
            src_names[rel] = names
            src_imports[rel] = imports
            declared = self._declared_keys(tree)
            if declared is not None:
                src_declared[rel] = declared
            for imp in imports:
                target = by_module.get(imp)
                if target is None and "." in imp:
                    target = by_module.get(imp.rsplit(".", 1)[0])
                if target is not None and target != rel:
                    importers[target].add(rel)
        self.importers = importers
        used_keys = set().union(*test_keys.values()) if test_keys else set()
        universe = all_fields & used_keys
        readers: dict[str, int] = defaultdict(int)
        for names in src_names.values():
            for key in names & universe:
                readers[key] += 1
        universe -= {
            k
            for k, n in readers.items()
            if n / max(len(src_names), 1) >= SOURCE_UBIQUITY_THRESHOLD
        }

        opaque_exempt: set[str] = set()
        configured = [t for t in tests if test_keys[t] & universe]
        counts: dict[str, int] = defaultdict(int)
        for test in configured:
            for key in test_keys[test] & universe:
                counts[key] += 1
        ubiquitous = {
            k
            for k, c in counts.items()
            if configured and c / len(configured) >= UBIQUITY_THRESHOLD
        }
        # a test with no visible config can be reached by anything, so it rides along with every subset
        self.opaque_tests = [
            t for t in tests if t not in configured and t not in opaque_exempt
        ]

        self.entry_point_dirs = self._entry_point_dirs(by_module)
        self.script_modules = self._script_modules(by_module)
        self.tests = tests
        self.helper_tests = helper_tests
        self.test_keys = test_keys
        self.test_strings = test_strings
        self.test_imports = test_imports
        self.src_trees = src_trees
        self.src_names = src_names
        self.src_imports = src_imports
        self.src_declared = src_declared
        self.all_fields = all_fields
        self.universe = universe
        self.ubiquitous = ubiquitous

    @staticmethod
    def _norm(text: str) -> str:
        return re.sub(r"[^a-z0-9]", "", text.lower())

    def _value_edges(self, rel: str) -> set[str]:
        """Tests whose config values name this module: a dotted path, a stem or package name."""
        if rel in self._value_cache:
            return self._value_cache[rel]
        hits = self._compute_value_edges(rel)
        self._value_cache[rel] = hits
        return hits

    def _compute_value_edges(self, rel: str) -> set[str]:
        module = self.module_name(rel)
        path = Path(rel)
        is_package = path.name == "__init__.py"
        own = path.parent.name if is_package else path.stem.lstrip("_")
        exact: set[str] = set()
        fuzzy: set[str] = set()
        for name in (own, path.parent.name):
            if name in GENERIC_NAMES or name == "axolotl" or len(name) < MIN_STRING_LEN:
                continue
            exact.add(name)
            # repo ids like org/Model-Name reach model support packages
            if name == own and len(self._norm(name)) >= FUZZY_MIN_LEN:
                fuzzy.add(self._norm(name))
        hits = set()
        for test, values in self.test_strings.items():
            if (
                values & exact
                or module in values
                # a plugin value is a class path inside its package
                or (is_package and any(v.startswith(module + ".") for v in values))
            ):
                hits.add(test)
                continue
            if fuzzy and any(
                f in self._norm(v) for v in values if "/" in v for f in fuzzy
            ):
                hits.add(test)
        return hits

    def _import_edges(self, rel: str) -> set[str]:
        module = self.module_name(rel)
        hits = set()
        for test, imports in self.test_imports.items():
            if any(imp == module or imp.startswith(module + ".") for imp in imports):
                hits.add(test)
        return hits

    def _pyproject(self) -> dict:
        try:
            with open(self.repo / "pyproject.toml", "rb") as fh:
                return tomllib.load(fh).get("project", {})
        except (OSError, tomllib.TOMLDecodeError):
            return {}

    def _script_modules(self, by_module: dict[str, str]) -> set[str]:
        """Modules behind console scripts: every subprocess-driven test starts there."""
        return {
            by_module[target.split(":")[0]]
            for target in self._pyproject().get("scripts", {}).values()
            if target.split(":")[0] in by_module
        }

    def _entry_point_dirs(self, by_module: dict[str, str]) -> set[str]:
        """Package directories registered through pyproject entry points (plugins, model support)."""
        groups = self._pyproject().get("entry-points", {})
        dirs = set()
        for group, entries in groups.items():
            if not group.startswith("axolotl"):
                continue
            for target in entries.values():
                module = target.split(":")[0]
                while module and module not in by_module:
                    module = module.rpartition(".")[0]
                if module:
                    dirs.add(str(Path(by_module[module]).parent))
        return dirs

    def _is_registry_dir(self, directory: str) -> bool:
        """A directory most of whose entries are named by config values (strategies, plugins, model support)."""
        if directory in self._registry_cache:
            return self._registry_cache[directory]
        if directory == SRC_ROOT:
            return False
        entries: dict[str, bool] = {}
        for rel in self.src_trees:
            if not rel.startswith(directory + "/"):
                continue
            rest = rel[len(directory) + 1 :]
            entry = rest.split("/", 1)[0]
            if entry == "__init__.py":
                continue
            entries[entry] = entries.get(entry, False) or bool(self._value_edges(rel))
        named = sum(entries.values())
        by_value = (
            named >= MIN_REGISTRY_ENTRIES
            and named / len(entries) >= REGISTRY_NAMED_FRACTION
        )
        declared = sum(
            1 for entry in entries if f"{directory}/{entry}" in self.entry_point_dirs
        )
        result = by_value or declared >= MIN_REGISTRY_ENTRIES
        self._registry_cache[directory] = result
        return result

    def _registry_entry(self, rel: str) -> str | None:
        """The registry entry holding this module: the file itself or its package directory."""
        directory = str(Path(rel).parent)
        if self._is_registry_dir(directory):
            return rel
        if self._is_registry_dir(str(Path(directory).parent)):
            return directory
        return None

    def _is_registry_leaf(self, rel: str, entry: str) -> bool:
        """Reached only by config value: nothing outside the registry entry imports any of it."""
        members = [m for m in self.src_trees if m == entry or m.startswith(entry + "/")]
        return not any(
            imp
            for member in members
            for imp in self.importers.get(member, ())
            if imp != entry and not imp.startswith(entry + "/")
        )

    def _own_edges(self, rel: str) -> tuple[set[str] | None, str] | None:
        """Edges this module carries itself; None when it has none and importers decide."""
        if rel in self._edge_cache:
            return self._edge_cache[rel]
        result = self._compute_own_edges(rel)
        self._edge_cache[rel] = result
        return result

    def _compute_own_edges(self, rel: str) -> tuple[set[str] | None, str] | None:
        if rel in self.script_modules:
            return None, "console script entry point"
        declared = self.src_declared.get(rel)
        if declared is not None:
            unknown = [k for k in declared if k not in self.all_fields]
            if not declared or unknown:
                return (
                    None,
                    f"{DECLARED_KEYS_NAME} names no config field: {unknown or declared}",
                )
            hits = {t for t, keys in self.test_keys.items() if keys & set(declared)}
            return hits, f"declared keys {list(declared)}"
        names = self.src_names.get(rel)
        if names is None:
            tree = _safe_parse(self.repo / rel)
            if tree is None:
                return None, "unparseable module"
            names = self._referenced_names(tree)
        read = names & self.universe
        discriminating = read - self.ubiquitous
        key_hits = {t for t, keys in self.test_keys.items() if keys & discriminating}
        imported = self._import_edges(rel)
        named = self._value_edges(rel)
        entry = self._registry_entry(rel)
        if entry is not None:
            if not named:
                # members of a plugin or strategy package are reached through the package name
                init = f"{entry}/__init__.py"
                if init != rel and init in self.src_trees:
                    named = self._value_edges(init)
            if named:
                return (
                    named | imported | key_hits,
                    f"named by config values; keys={sorted(discriminating)[:4]}",
                )
            if self._is_registry_leaf(rel, entry):
                return (
                    imported | key_hits,
                    f"registry leaf no config value names; keys={sorted(discriminating)[:4]}",
                )
        core = read & self.ubiquitous
        if core:
            return None, f"reads ubiquitous key(s) {sorted(core)[:3]}"
        hits = named | imported | key_hits
        if hits or discriminating:
            return hits, f"keys={sorted(discriminating)[:4]}"
        return None

    def impact_of_source(self, rel: str) -> tuple[set[str] | None, str]:
        """Return (tests, reason); tests=None means the whole scope.

        A module without edges of its own is reached through whatever imports it, so
        the walk climbs importers until every branch ends at a module with edges.
        """
        own = self._own_edges(rel)
        if own is not None:
            return own
        hits: set[str] = set()
        seen = {rel}
        frontier = [rel]
        explained = []
        while frontier:
            module = frontier.pop()
            parents = self.importers.get(module, set()) - seen
            if not parents and module == rel:
                return None, "no derivable edges and no importers"
            for parent in parents:
                seen.add(parent)
                edges = self._own_edges(parent)
                if edges is None:
                    frontier.append(parent)
                    continue
                tests, reason = edges
                if tests is None:
                    return None, f"via {parent}: {reason}"
                hits |= tests
                explained.append(parent)
        if not explained:
            return None, "no derivable edges through importers"
        return hits, f"via {sorted(explained)[:3]}"

    # -- pyproject dependencies -------------------------------------------

    @staticmethod
    def _normalize_dist(name: str) -> str:
        return re.sub(r"[-_.]+", "-", name).lower()

    def _requirements(self, pyproject: dict) -> dict[str, set[str]]:
        """Requirement strings per normalized distribution, keyed by the table declaring them.

        The same string can sit in several extras; moving it between them changes what
        each extra installs, so the table name is part of the identity.
        """
        project = pyproject.get("project", {})
        tables: list[tuple[str, list[str]]] = [("", project.get("dependencies", []))]
        tables.extend(project.get("optional-dependencies", {}).items())
        by_dist: dict[str, set[str]] = defaultdict(set)
        for table, reqs in tables:
            for req in reqs:
                match = REQUIREMENT_NAME_RE.match(req)
                if match:
                    by_dist[self._normalize_dist(match.group(1))].add(
                        f"{table}:{req.strip()}"
                    )
        return by_dist

    @staticmethod
    def _without_dependencies(pyproject: dict) -> dict:
        trimmed = dict(pyproject)
        project = dict(trimmed.get("project", {}))
        for section in DEP_SECTIONS:
            project.pop(section, None)
        trimmed["project"] = project
        return trimmed

    def _dep_map(self, pyproject: dict) -> dict[str, list[str]]:
        table: object = pyproject
        for key in DEP_MAP_TABLE:
            table = table.get(key, {}) if isinstance(table, dict) else {}
        if not isinstance(table, dict):
            return {}
        return {
            self._normalize_dist(dist): list(roots) if isinstance(roots, list) else []
            for dist, roots in table.items()
        }

    def _root_importers(self, root: str) -> set[str]:
        return {
            rel
            for rel, imports in self.src_imports.items()
            if any(imp == root or imp.startswith(root + ".") for imp in imports)
        }

    def _root_tests(self, root: str) -> set[str]:
        hits = set()
        for test, imports in self.test_imports.items():
            if any(imp == root or imp.startswith(root + ".") for imp in imports):
                hits.add(test)
        for test, strings in self.test_strings.items():
            if root in strings:
                hits.add(test)
        return hits

    def impact_of_pyproject(self, base_rev: str) -> tuple[set[str] | None, str]:
        """Tests reached by the distributions whose requirements changed; None runs everything."""
        try:
            base = tomllib.loads(_git(self.repo, "show", f"{base_rev}:{PYPROJECT}"))
            head = tomllib.loads((self.repo / PYPROJECT).read_text(encoding="utf-8"))
        except (subprocess.CalledProcessError, OSError, tomllib.TOMLDecodeError):
            return None, f"{PYPROJECT} unreadable at base or head"
        if self._without_dependencies(base) != self._without_dependencies(head):
            return None, f"{PYPROJECT} changed outside the dependency tables"
        base_extras = set(base.get("project", {}).get("optional-dependencies", {}))
        head_extras = set(head.get("project", {}).get("optional-dependencies", {}))
        if base_extras != head_extras:
            return None, f"{PYPROJECT} adds or removes an extra"
        base_reqs, head_reqs = self._requirements(base), self._requirements(head)
        changed = sorted(
            d
            for d in base_reqs.keys() | head_reqs.keys()
            if base_reqs[d] != head_reqs[d]
        )
        if not changed:
            return set(), f"{PYPROJECT} dependency tables unchanged"
        dep_map = self._dep_map(head)
        unmapped = [d for d in changed if d not in dep_map]
        if unmapped:
            return None, f"{PYPROJECT}: no [tool.axolotl.ci.deps] entry for {unmapped}"
        hits: set[str] = set()
        explained: list[str] = []
        for dist in changed:
            roots = dep_map[dist]
            if not roots:
                return None, f"{PYPROJECT}: {dist} maps to no import root"
            for root in roots:
                importers = self._root_importers(root)
                tests = self._root_tests(root)
                if not importers and not tests:
                    return None, f"{PYPROJECT}: nothing imports {root} ({dist})"
                for rel in sorted(importers):
                    reached, reason = self.impact_of_source(rel)
                    if reached is None:
                        return (
                            None,
                            f"{PYPROJECT}: {dist} -> {root} via {rel}: {reason}",
                        )
                    hits |= reached
                hits |= tests
                explained.append(f"{dist} -> {root} ({len(importers)} importer(s))")
        return hits, "; ".join(explained)

    # -- decision ----------------------------------------------------------

    @staticmethod
    def _inert(path: str) -> bool:
        return any(fnmatch.fnmatch(path, g) for g in INERT_GLOBS)

    def select(self, base: str | None, merge_commit: bool) -> Selection:
        if not merge_commit and not base:
            return Selection("all", self.test_files(), "no base ref")
        try:
            changed, deleted = self.changed(base, merge_commit)
        except subprocess.CalledProcessError as err:
            return Selection(
                "all", self.test_files(), f"diff failed: {err.output.strip()}"
            )
        messages = self.commit_messages(base, merge_commit).lower()
        if any(tok in messages for tok in FORCE_ALL_TOKENS):
            return Selection(
                "all", self.test_files(), "commit message forces a full run"
            )

        self.build()
        run_all: list[str] = []
        selected: dict[str, set[str]] = defaultdict(set)
        for path in changed + deleted:
            if any(fnmatch.fnmatch(path, g) for g in RUN_ALL_GLOBS):
                run_all.append(f"{path} matches a run-all pattern")
        if PYPROJECT in deleted:
            run_all.append(f"deleted {PYPROJECT}")
        elif PYPROJECT in changed:
            changed = [p for p in changed if p != PYPROJECT]
            hits, reason = self.impact_of_pyproject(self._diff_base(base, merge_commit))
            if hits is None:
                run_all.append(reason)
            else:
                for test in hits:
                    selected[test].add(f"{PYPROJECT} ({reason})")
        for path in deleted:
            if path.startswith(SRC_ROOT + "/") and path.endswith(".py"):
                run_all.append(f"deleted module {path}")
            elif path.startswith("tests/") and not Path(path).name.startswith("test_"):
                run_all.append(f"deleted test support {path}")
            elif not self._inert(path) and not path.startswith(
                ("tests/", SRC_ROOT + "/")
            ):
                run_all.append(f"deleted unmodeled file {path}")

        in_scope = set(self.tests)
        for path in changed:
            if path in in_scope:
                selected[path].add("changed test file")
                continue
            if path.startswith("tests/"):
                if path.endswith(".py") and Path(path).name.startswith("test_"):
                    continue  # a test outside this scope
                if path in self.helper_tests:
                    for test in self.helper_tests[path]:
                        selected[test].add(f"{path} (helper it references)")
                    continue
                run_all.append(f"{path} is shared test support")
                continue
            if path.startswith(SRC_ROOT + "/") and path.endswith(".py"):
                hits, reason = self.impact_of_source(path)
                if hits is None:
                    run_all.append(f"{path}: {reason}")
                    continue
                for test in hits:
                    selected[test].add(f"{path} ({reason})")
                continue
            if not self._inert(path):
                # package data, deepspeed configs, anything the graph does not model
                run_all.append(f"{path} is not modeled")
        if run_all:
            return Selection("all", self.tests, "; ".join(run_all), dict(selected))
        if selected:
            for test in self.opaque_tests:
                selected[test].add("no visible config; rides along with every subset")
        tests = sorted(selected)
        if not tests:
            return Selection("none", [], "no in-scope test is reachable from the diff")
        if len(tests) == len(self.tests):
            return Selection(
                "all", tests, "every in-scope test is reachable", dict(selected)
            )
        return Selection(
            "subset",
            tests,
            f"{len(tests)} of {len(self.tests)} test files",
            dict(selected),
        )


def _write_outputs(name: str, sel: Selection) -> None:
    output = os.environ.get("GITHUB_OUTPUT")
    if output:
        with open(output, "a", encoding="utf-8") as fh:
            fh.write(f"{name}_mode={sel.mode}\n")
            fh.write(f"{name}_count={len(sel.tests)}\n")
            fh.write(
                f"{name}_tests={','.join(sel.tests) if sel.mode == 'subset' else ''}\n"
            )
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(
                f"### e2e selection `{name}`: **{sel.mode}** ({len(sel.tests)} files)\n\n"
            )
            fh.write(f"{sel.reason}\n\n")
            if sel.mode == "subset":
                fh.writelines(f"- `{t}`\n" for t in sel.tests)
            fh.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scope", default="tests/e2e")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument(
        "--base", help="base ref; the diff is merge-base(base, HEAD)..HEAD"
    )
    parser.add_argument(
        "--merge-commit",
        action="store_true",
        help="HEAD is a PR merge commit; diff HEAD^1..HEAD",
    )
    parser.add_argument(
        "--head",
        default="HEAD",
        help="revision to diff (the working tree is still what gets parsed)",
    )
    parser.add_argument("--name", default="e2e", help="prefix for GitHub outputs")
    parser.add_argument(
        "--output", type=Path, help="write the selected test files here"
    )
    parser.add_argument("--explain", action="store_true")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    args = parser.parse_args()

    selector = Selector(args.repo_root, args.scope, tuple(args.exclude), args.head)
    try:
        sel = selector.select(args.base, args.merge_commit)
    except Exception:  # pylint: disable=broad-except
        traceback.print_exc()
        sel = Selection(
            "all", selector.test_files(), "selector raised; running everything"
        )

    print(f"### MODE: {sel.mode} ({sel.reason}) ###")
    print(f"### {len(sel.tests)} test file(s) in scope {args.scope} ###")
    for test in sel.tests:
        print(f"  {test}")
        if args.explain:
            for why in sorted(sel.explain.get(test, ())):
                print(f"      <- {why}")
    if args.explain and hasattr(selector, "ubiquitous"):
        print(f"### ubiquitous keys (no signal): {sorted(selector.ubiquitous)}")
        registries = sorted(
            d for d, is_reg in selector._registry_cache.items() if is_reg
        )  # pylint: disable=protected-access
        print(f"### registry directories: {registries}")
        print(f"### opaque tests (always in a subset): {selector.opaque_tests}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("".join(f"{t}\n" for t in sel.tests), encoding="utf-8")
        args.output.with_suffix(".mode").write_text(f"{sel.mode}\n", encoding="utf-8")
    _write_outputs(args.name, sel)
    return 0


if __name__ == "__main__":
    sys.exit(main())
