"""``axolotl.core.plan`` imports only the standard library and itself."""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PLAN_ROOT = Path(__file__).resolve().parents[3] / "src" / "axolotl" / "core" / "plan"
OWN_PACKAGE = "axolotl.core.plan"


def _imported_modules(tree: ast.AST, module: str) -> list[tuple[int, str]]:
    package = module.rsplit(".", 1)[0]
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = (
                    package.rsplit(".", node.level - 1)[0]
                    if node.level > 1
                    else package
                )
                name = f"{base}.{node.module}" if node.module else base
            else:
                name = node.module or ""
            found.append((node.lineno, name))
    return found


def _allowed(name: str) -> bool:
    if name == OWN_PACKAGE or name.startswith(OWN_PACKAGE + "."):
        return True
    return name.split(".", 1)[0] in sys.stdlib_module_names


def test_core_plan_imports_only_stdlib():
    files = sorted(PLAN_ROOT.rglob("*.py"))
    assert files, f"no sources under {PLAN_ROOT}"
    offenders = []
    for path in files:
        rel = path.relative_to(PLAN_ROOT.parents[2])
        module = ".".join(rel.with_suffix("").parts).removesuffix(".__init__")
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        offenders.extend(
            f"{rel}:{line}  {name}"
            for line, name in _imported_modules(tree, module)
            if not _allowed(name)
        )
    assert not offenders, (
        "axolotl.core.plan must import only the standard library and itself:\n  "
        + "\n  ".join(offenders)
    )


def test_gate_catches_third_party_and_axolotl_imports():
    tree = ast.parse(
        "import torch\nfrom axolotl.utils.dict import DictDefault\nfrom . import specs\n"
    )
    names = [
        name
        for _, name in _imported_modules(tree, "axolotl.core.plan.slices.mesh")
        if not _allowed(name)
    ]
    assert names == ["torch", "axolotl.utils.dict"]
