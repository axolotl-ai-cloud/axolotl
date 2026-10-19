"""Smoke-check an installed axolotl: import every module and resolve the entry points."""

import argparse
import importlib
import importlib.metadata as md
import importlib.util
import pkgutil
import sys

ENTRY_POINT_GROUPS = (
    "axolotl.plugins",
    "axolotl.model_support",
    "axolotl.cloud_providers",
    "axolotl.cli_commands",
)
REQUIRED_ENTRY_POINTS = {"axolotl.plugins": 2, "axolotl.cloud_providers": 3}
KNOWN_BROKEN_MODULES: set[str] = set()


def import_walk() -> bool:
    import axolotl

    optional: list[tuple[str, str]] = []
    failed: list[tuple[str, str]] = []
    known: list[tuple[str, str]] = []
    names = [
        m.name
        for m in pkgutil.walk_packages(
            axolotl.__path__, "axolotl.", onerror=lambda _name: None
        )
    ]
    for name in names:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001
            top = (getattr(exc, "name", None) or "").partition(".")[0]
            absent = (
                isinstance(exc, ModuleNotFoundError)
                and top
                and not top.startswith("axolotl")
                and importlib.util.find_spec(top) is None
            )
            bucket = (
                known
                if name in KNOWN_BROKEN_MODULES
                else optional
                if absent
                else failed
            )
            bucket.append((name, f"{type(exc).__name__}: {exc}"))
    ok = len(names) - len(optional) - len(failed) - len(known)
    print(
        f"import-walk: ok={ok} optional={len(optional)} "
        f"known-broken={len(known)} failed={len(failed)}"
    )
    for name, why in optional:
        print(f"  optional: {name}: {why}")
    for name, why in known:
        print(f"  known-broken: {name}: {why}")
    for name, why in failed:
        print(f"  FAILED: {name}: {why}")
    return not failed


def check_entry_points() -> bool:
    ok = True
    for group in ENTRY_POINT_GROUPS:
        eps = [
            ep for ep in md.distribution("axolotl").entry_points if ep.group == group
        ]
        print(f"{group}: {len(eps)}")
        for ep in eps:
            try:
                ep.load()
            except Exception as exc:  # noqa: BLE001
                ok = False
                print(f"  FAILED: {ep.value}: {type(exc).__name__}: {exc}")
        required = REQUIRED_ENTRY_POINTS.get(group, 0)
        if len(eps) < required:
            ok = False
            print(f"  FAILED: expected at least {required} entry points in {group}")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--import-walk", action="store_true")
    parser.add_argument("--entry-points", action="store_true")
    parser.add_argument("--require-site-packages", action="store_true")
    args = parser.parse_args()

    ok = True
    if args.require_site_packages:
        import axolotl

        assert "site-packages" in axolotl.__file__, axolotl.__file__
    if args.import_walk:
        ok = import_walk() and ok
    if args.entry_points:
        ok = check_entry_points() and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
