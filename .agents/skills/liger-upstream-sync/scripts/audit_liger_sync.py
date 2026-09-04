#!/usr/bin/env python3
"""Audit axolotl's Liger plugin against the installed liger-kernel.

The plugin dispatches a model type to the upstream *generic* native path when

    model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN
    and model_config_type not in axolotl_override_liger_fn

so a hand-written ``elif`` branch only runs when the type is in the override set
(or absent from the upstream dict). When liger ships native support for a type
axolotl still hand-patches, the generic path silently shadows the hand-patch and
the elif becomes dead code -- a correctness/perf drift that no version check or
import error catches. This script reports those cases plus glu-activation drift in
the generic path's parameter probes.

Exit status: 0 = clean, 1 = findings need review, 2 = could not audit reliably
(liger-kernel not installed, or plugin.py drifted from the shape the parser reads).
So a CI gate trips on any non-zero status.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass, field
from pathlib import Path

PLUGIN_REL = Path("src/axolotl/integrations/liger/plugin.py")
OVERRIDE_SET_NAME = "axolotl_override_liger_fn"


@dataclass
class PluginFacts:
    override_set: set[str] = field(default_factory=set)
    elif_types: set[str] = field(default_factory=set)
    relied_params: set[str] = field(default_factory=set)
    # raised when the plugin shape drifts from what the parser can read statically;
    # results are then untrustworthy and the audit exits 2
    warnings: list[str] = field(default_factory=list)


def find_plugin(start: Path) -> Path:
    for base in (start, *start.parents):
        candidate = base / PLUGIN_REL
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"could not locate {PLUGIN_REL} from {start}")


def _is_model_config_type(node: ast.expr) -> bool:
    return isinstance(node, ast.Attribute) and node.attr == "model_config_type"


def _str_constants(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        out: list[str] = []
        for elt in node.elts:
            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                out.append(elt.value)
        return out
    return []


def _model_type_constants(test: ast.expr) -> set[str]:
    out: set[str] = set()
    for node in ast.walk(test):
        if isinstance(node, ast.Compare) and _is_model_config_type(node.left):
            for op, comparator in zip(node.ops, node.comparators, strict=False):
                if isinstance(op, (ast.Eq, ast.In)):
                    out.update(_str_constants(comparator))
    return out


def _references(node: ast.AST, name: str) -> bool:
    return any(isinstance(n, ast.Name) and n.id == name for n in ast.walk(node))


def _looks_like_dispatch_head(test: ast.expr) -> bool:
    # Match the dispatch *shape* (`model_config_type in MODEL_TYPE_TO_APPLY_LIGER_FN`
    # and `... not in axolotl_override_liger_fn`), not mere symbol presence -- else an
    # unrelated `if` that only mentions the names could win and empty the elif types.
    has_upstream_membership = False
    has_override_exclusion = False
    for node in ast.walk(test):
        if not (isinstance(node, ast.Compare) and _is_model_config_type(node.left)):
            continue
        for op, comparator in zip(node.ops, node.comparators, strict=False):
            if (
                isinstance(op, ast.In)
                and isinstance(comparator, ast.Name)
                and comparator.id == "MODEL_TYPE_TO_APPLY_LIGER_FN"
            ):
                has_upstream_membership = True
            elif (
                isinstance(op, ast.NotIn)
                and isinstance(comparator, ast.Name)
                and comparator.id == OVERRIDE_SET_NAME
            ):
                has_override_exclusion = True
    return has_upstream_membership and has_override_exclusion


def _find_dispatch_head(tree: ast.AST) -> tuple[ast.If | None, list[str]]:
    # The genuine dispatch head tests *both* the upstream table and the override
    # set (`in MODEL_TYPE_TO_APPLY_LIGER_FN and not in axolotl_override_liger_fn`).
    warnings: list[str] = []
    candidates = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.If) and _looks_like_dispatch_head(n.test)
    ]
    candidates.sort(key=lambda n: n.lineno)
    if not candidates:
        warnings.append(
            "dispatch head not found (no `if` tests both MODEL_TYPE_TO_APPLY_LIGER_FN "
            "and the override set); hand-patched elif types could not be read"
        )
        return None, warnings
    if len(candidates) > 1:
        warnings.append(
            f"{len(candidates)} dispatch-head candidates found; using the first by "
            "line number -- the if/elif ladder may have been restructured"
        )
    return candidates[0], warnings


def _elif_types(head: ast.If | None) -> set[str]:
    # Walk only the elif *tests* down the orelse chain. Branch bodies (e.g. the
    # generic path's rope-default tuple) must not be scanned.
    types: set[str] = set()
    node = head
    while (
        node is not None
        and len(node.orelse) == 1
        and isinstance(node.orelse[0], ast.If)
    ):
        node = node.orelse[0]
        types |= _model_type_constants(node.test)
    return types


def _is_unresolved(value: ast.expr) -> bool:
    # Anything not fully readable as a literal set/list/tuple of string constants is
    # unresolved, so a missed entry surfaces as a reliability warning instead of a
    # silent undercount in [1]/[2].
    if isinstance(value, (ast.Set, ast.List, ast.Tuple)):
        return any(
            not (isinstance(elt, ast.Constant) and isinstance(elt.value, str))
            for elt in value.elts
        )
    if isinstance(value, ast.Call):
        # empty set()/frozenset() is fine; it gets populated later via .add()/|=
        func = value.func
        empty_ctor = (
            isinstance(func, ast.Name)
            and func.id in {"set", "frozenset"}
            and not value.args
        )
        return not empty_ctor
    if isinstance(value, ast.Constant) and isinstance(value.value, str):
        return False  # a bare string (e.g. `.add("x")`) is fully resolved
    return True  # Name, Attribute, Subscript, BinOp, Dict, comprehension, ...


def _collect_override(tree: ast.AST) -> tuple[set[str], list[str]]:
    # Capture every static mutation of the override set: `= {...}`, `|= {...}`,
    # `.update({...})`, `.add("x")`. Warn (rather than silently undercount) when a
    # mutation uses a form we cannot read, since a missed entry would surface as a
    # false positive in [1]/[2].
    entries: set[str] = set()
    warnings: list[str] = []

    def _take(value: ast.expr, ctx: str) -> None:
        entries.update(_str_constants(value))
        if _is_unresolved(value):
            warnings.append(
                f"override set built via {ctx} with a non-literal value; entries may "
                "be incomplete"
            )

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            if any(
                isinstance(t, ast.Name) and t.id == OVERRIDE_SET_NAME
                for t in node.targets
            ):
                _take(node.value, "assignment")
        elif isinstance(node, ast.AugAssign):
            if (
                isinstance(node.target, ast.Name)
                and node.target.id == OVERRIDE_SET_NAME
                and isinstance(node.op, ast.BitOr)
            ):
                _take(node.value, "|=")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            func = node.func
            if (
                isinstance(func.value, ast.Name)
                and func.value.id == OVERRIDE_SET_NAME
                and func.attr in {"update", "add"}
                and node.args
            ):
                _take(node.args[0], f".{func.attr}()")
    return entries, warnings


def _relied_params(head: ast.If | None) -> set[str]:
    # The generic path probes `"<param>" in liger_fn_sig.parameters`; liger_fn_sig
    # only exists in that block, so scope the scan to the head body -- an unrelated
    # `.parameters` membership test elsewhere must not mask real generic-path drift.
    params: set[str] = set()
    if head is None:
        return params
    for stmt in head.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Compare) and len(node.ops) == 1:
                (op,) = node.ops
                (right,) = node.comparators
                if (
                    isinstance(op, ast.In)
                    and isinstance(node.left, ast.Constant)
                    and isinstance(node.left.value, str)
                    and isinstance(right, ast.Attribute)
                    and right.attr == "parameters"
                    and isinstance(right.value, ast.Name)
                    and right.value.id == "liger_fn_sig"
                ):
                    params.add(node.left.value)
    return params


def parse_plugin(path: Path) -> PluginFacts:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    facts = PluginFacts()
    head, head_warnings = _find_dispatch_head(tree)
    facts.elif_types = _elif_types(head)
    facts.override_set, override_warnings = _collect_override(tree)
    facts.relied_params = _relied_params(head)
    facts.warnings = head_warnings + override_warnings
    return facts


def load_upstream() -> tuple[dict[str, object] | None, str]:
    try:
        from liger_kernel.transformers.monkey_patch import (
            MODEL_TYPE_TO_APPLY_LIGER_FN,
        )
    except Exception as exc:  # noqa: BLE001 - report, don't crash the audit
        return None, f"unavailable ({exc.__class__.__name__}: {exc})"
    version = ""
    try:
        import liger_kernel

        version = getattr(liger_kernel, "__version__", "") or ""
    except Exception:  # noqa: BLE001
        version = ""
    if not version:
        # builds without __version__ still resolve via package metadata
        from importlib.metadata import PackageNotFoundError, version as _pkg_version

        try:
            version = _pkg_version("liger-kernel")
        except PackageNotFoundError:
            version = "unknown"
    return dict(MODEL_TYPE_TO_APPLY_LIGER_FN), version


MONKEY_PATCH_REL = Path("liger_kernel/transformers/monkey_patch.py")


def dispatch_table_keys(src: str) -> set[str]:
    """Extract the MODEL_TYPE_TO_APPLY_LIGER_FN keys from monkey_patch.py source.

    Lets the audit diff against a liger version that is not installed (and avoids
    importing liger at all), at the cost of real fn signatures -> no [4] drift.
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "MODEL_TYPE_TO_APPLY_LIGER_FN"
            for t in node.targets
        ):
            if isinstance(node.value, ast.Dict):
                keys: set[str] = set()
                for k in node.value.keys:
                    # k is None for `{**base, ...}` unpacking; a non-string-literal
                    # key means the table can't be read in full -> signal unreadable
                    # (empty) rather than return a misleading partial set
                    if not (isinstance(k, ast.Constant) and isinstance(k.value, str)):
                        return set()
                    keys.add(k.value)
                return keys
    return set()


def _resolve_source(path: Path) -> Path | None:
    if path.is_file():
        return path
    for cand in (path / MONKEY_PATCH_REL, path / "src" / MONKEY_PATCH_REL):
        if cand.is_file():
            return cand
    hits = sorted(path.glob(f"**/{MONKEY_PATCH_REL}"))
    return hits[0] if hits else None


def _source_version(monkey_patch: Path) -> str:
    # best-effort: __version__ from the package __init__.py, else a static version
    # in a parent pyproject.toml (src/ checkouts often ship an empty __init__.py)
    init = monkey_patch.parent.parent / "__init__.py"
    if init.is_file():
        for line in init.read_text(encoding="utf-8").splitlines():
            if line.strip().startswith("__version__"):
                return line.split("=", 1)[1].strip().strip("\"'")
    for parent in monkey_patch.parents:
        pyproject = parent / "pyproject.toml"
        if pyproject.is_file():
            for line in pyproject.read_text(encoding="utf-8").splitlines():
                if line.split("=", 1)[0].strip() == "version":
                    return line.split("=", 1)[1].strip().strip("\"'")
            break
    return "source"


def load_source(path: Path) -> tuple[dict[str, object] | None, str]:
    monkey_patch = _resolve_source(path)
    if monkey_patch is None:
        return None, f"no monkey_patch.py under {path}"
    keys = dispatch_table_keys(monkey_patch.read_text(encoding="utf-8"))
    if not keys:
        # unreadable shape (not a Dict literal, built incrementally, ...) -> could
        # not audit, rather than a misleading 0-type table that flags everything
        return None, f"no readable MODEL_TYPE_TO_APPLY_LIGER_FN dict in {monkey_patch}"
    # keys-only table (no fn objects) -> signature drift [4] is skipped
    return dict.fromkeys(keys), f"{_source_version(monkey_patch)} (source, keys-only)"


def _glu_param_drift(fn: object, relied: set[str]) -> str | None:
    # Real drift is a *rename*: upstream exposes a glu-activation toggle under a
    # name the plugin does not check (e.g. swiglu->glu), so liger_glu_activation
    # silently stops being forwarded. A fn with no glu param at all is not drift
    # (that model simply has no liger glu kernel), so it is not flagged.
    import inspect

    try:
        params = set(inspect.signature(fn).parameters)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    glu_params = {p for p in params if "glu" in p.lower()}
    # substring match, so both geglu and swiglu probes count as a live glu check
    checked = {p for p in relied if "glu" in p.lower()}
    if glu_params and not (glu_params & checked):
        return (
            f"exposes glu param(s) {sorted(glu_params)} but the plugin only checks "
            f"{sorted(checked)} -> stale glu check (upstream rename?)"
        )
    return None


@dataclass
class Findings:
    shadowed: list[str] = field(default_factory=list)  # [1]
    override_stale: list[str] = field(default_factory=list)  # [2] not native
    override_unrouted: list[str] = field(default_factory=list)  # [2] no elif
    override_ok: list[str] = field(default_factory=list)  # [2] healthy
    custom: list[str] = field(default_factory=list)  # [3]
    drift: list[tuple[str, str]] = field(default_factory=list)  # [4] (type, message)

    @property
    def review_count(self) -> int:
        return (
            len(self.shadowed)
            + len(self.override_stale)
            + len(self.override_unrouted)
            + len(self.drift)
        )


def classify(
    facts: PluginFacts, upstream: dict[str, object], check_drift: bool = True
) -> Findings:
    keys = set(upstream)
    result = Findings()
    # [1] in the table and hand-patched, but not overridden -> generic path shadows it
    result.shadowed = sorted((facts.elif_types & keys) - facts.override_set)
    # [2] each override entry should be native and routed to an elif
    for t in sorted(facts.override_set):
        if t not in keys:
            result.override_stale.append(t)
        elif t not in facts.elif_types:
            result.override_unrouted.append(t)
        else:
            result.override_ok.append(t)
    # [3] hand-patched and absent from the table -> genuinely custom
    result.custom = sorted(facts.elif_types - keys)
    # [4] generic-path types whose glu toggle the plugin no longer forwards
    # (needs real fn signatures; skipped in keys-only source mode)
    if check_drift:
        for t in sorted(keys - facts.override_set):
            msg = _glu_param_drift(upstream[t], facts.relied_params)
            if msg:
                result.drift.append((t, msg))
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plugin",
        type=Path,
        default=None,
        help="path to liger plugin.py (default: auto-locate from cwd)",
    )
    parser.add_argument(
        "--liger-source",
        type=Path,
        default=None,
        help="diff against a liger monkey_patch.py / package dir instead of the "
        "installed liger-kernel (keys-only: [4] signature drift is skipped). "
        "Use to preview a version before bumping the pin.",
    )
    args = parser.parse_args(argv)

    if args.plugin is not None:
        plugin = args.plugin
        if not plugin.is_file():
            print(f"plugin not found: {plugin}")
            return 2
    else:
        try:
            plugin = find_plugin(Path.cwd())
        except FileNotFoundError as exc:
            print(str(exc))
            return 2
    try:
        facts = parse_plugin(plugin)
    except (OSError, UnicodeError, SyntaxError, ValueError) as exc:
        print(f"could not parse plugin for audit: {exc.__class__.__name__}: {exc}")
        return 2

    check_drift = args.liger_source is None
    if args.liger_source is not None:
        upstream, version = load_source(args.liger_source)
    else:
        upstream, version = load_upstream()

    print("Liger upstream-sync audit")
    print("=========================")
    print(f"plugin : {plugin}")
    if upstream is None:
        print(f"liger  : {version}")
        if args.liger_source is not None:
            print(f"\nCannot read the dispatch table from {args.liger_source}.")
        else:
            print(
                "\nCannot diff against the upstream dispatch table without "
                "liger-kernel installed. Install the training env and re-run."
            )
        return 2
    print(f"liger  : {version}  (MODEL_TYPE_TO_APPLY_LIGER_FN: {len(upstream)} types)")
    print(f"override set : {sorted(facts.override_set) or '(empty)'}")
    print(f"hand-patched (elif) types : {sorted(facts.elif_types)}")
    print(f"generic-path relied params : {sorted(facts.relied_params)}")

    if facts.warnings:
        print("\n!! AUDIT RELIABILITY WARNINGS (plugin.py shape changed; re-check)")
        for warning in facts.warnings:
            print(f"  ! {warning}")

    result = classify(facts, upstream, check_drift=check_drift)

    print("\n[1] SHADOWED HAND-PATCHES (generic native path runs; elif is dead code)")
    if result.shadowed:
        for t in result.shadowed:
            print(
                f"  ! {t}: upstream now dispatches '{t}' natively and it is not in "
                f"{OVERRIDE_SET_NAME}. The elif is bypassed. Add '{t}' to the override "
                "set to keep the hand-patch, or delete the elif if native suffices."
            )
    else:
        print("  none")

    print("\n[2] OVERRIDE-SET HEALTH")
    if facts.override_set:
        stale, unrouted = set(result.override_stale), set(result.override_unrouted)
        for t in sorted(facts.override_set):
            if t in stale:
                print(
                    f"  ! {t}: in {OVERRIDE_SET_NAME} but NOT in upstream dict -> the "
                    "override is stale (generic path would not fire anyway). Remove it."
                )
            elif t in unrouted:
                print(
                    f"  ! {t}: overridden out of the generic path but has no elif branch "
                    "-> liger is silently never applied for this type."
                )
            else:
                print(
                    f"  ok {t}: native + routed to its elif (verify the branch still "
                    "adds kernels native lacks)"
                )
    else:
        print("  (empty)")

    print("\n[3] CUSTOM HAND-PATCHES (not in upstream dict; correct to hand-patch)")
    print(f"  {result.custom or 'none'}")

    print("\n[4] SIGNATURE DRIFT (generic-path types: in dict, not overridden)")
    if not check_drift:
        print(
            "  skipped (--liger-source is keys-only; drift needs the installed package)"
        )
    elif result.drift:
        for t, msg in result.drift:
            print(f"  ! {t}: {msg}")
    else:
        print("  none")

    print("\n-------------------------------------------------------------")
    if facts.warnings:
        print(
            f"SUMMARY: {result.review_count} item(s) need review; "
            f"{len(facts.warnings)} reliability warning(s) -- results may be incomplete."
        )
        return 2
    print(f"SUMMARY: {result.review_count} item(s) need review.")
    return 1 if result.review_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
