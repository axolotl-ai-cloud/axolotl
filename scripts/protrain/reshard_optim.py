"""Offline cross-world-size reshard tool for Mode-C optimizer state.

Thin CLI wrapper around the core reshard logic at
``src/axolotl/integrations/protrain/api/reshard.py``. The same logic
also runs in-process from the load path when the user opts in via
``protrain_allow_online_reshard=True`` (see ``api/checkpoint.py`` Mode-C
branch). Keeping a single source of truth means the offline and online
paths cannot drift on shard arithmetic.

ProTrain Phase 2 Mode-C (ZeRO-3 sharded) saves a per-rank slice of every
non-persistent chunk's CPU Adam state to ``chunk_<N>_rank_<R>.pt``. The
load path hard-errors when ``saved_world_size != current_world_size``
unless the user opts in to online reshard. This tool is the offline
alternative — runs without GPUs, without ``torch.distributed``, and
without the heavyweight axolotl import chain (transformers, etc.) so
the conversion can happen on a CPU-only host.

To preserve the "no-axolotl-imports" property, the script loads
``api/reshard.py`` via ``importlib.util.spec_from_file_location`` rather
than the regular ``from axolotl... import`` path — that avoids firing
the package's ``__init__.py`` chain (``protrain/__init__.py`` pulls in
plugin.py, which transitively imports transformers).

Usage::

    python -m scripts.protrain.reshard_optim \\
        --src <N1-protrain_optim-dir> \\
        --dst <N2-protrain_optim-dir> \\
        --target-world N2

The ``--src`` directory must be a Mode-C save (``protrain_save_mode ==
"sharded"`` and ``layout_fingerprint`` field present). Mode-B saves
do not need resharding (the load path tolerates world_size drift
natively, see CHECKPOINT_DESIGN_PHASE2.md §4.1 Option B).
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import types


def _load_reshard_module() -> types.ModuleType:
    """Load the core reshard module by file path.

    Why not ``from axolotl.integrations.protrain.api.reshard import
    reshard_mode_c_shards``? Because that path fires
    ``axolotl/integrations/protrain/__init__.py``, which pulls in
    plugin.py, which transitively imports transformers — defeating the
    "this script runs on a vanilla CPU box" property documented above.

    ``importlib.util.spec_from_file_location`` loads the file as an
    isolated module without traversing the package hierarchy.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(os.path.dirname(here))  # scripts/protrain → repo
    target = os.path.join(
        repo_root,
        "src",
        "axolotl",
        "integrations",
        "protrain",
        "api",
        "reshard.py",
    )
    if not os.path.isfile(target):
        raise RuntimeError(
            f"reshard CLI: cannot locate core reshard module at {target!r}. "
            "The repository layout has changed; update _load_reshard_module."
        )
    spec = importlib.util.spec_from_file_location("_protrain_reshard_core", target)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"reshard CLI: importlib failed to build spec for {target!r}"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="reshard_optim",
        description=(
            "Offline cross-world-size reshard tool for ProTrain Mode-C optimizer state."
        ),
    )
    p.add_argument(
        "--src",
        required=True,
        help=(
            "Path to the source protrain_optim/ directory (output of a "
            "Mode-C save at world_size N1)."
        ),
    )
    p.add_argument(
        "--dst",
        required=True,
        help=(
            "Path to the destination directory for the resharded "
            "checkpoint. Must either not exist or be an empty directory; "
            "the resharder refuses to write into a non-empty path."
        ),
    )
    p.add_argument(
        "--target-world",
        type=int,
        required=True,
        help="Target world_size N2.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    if args.target_world < 1:
        parser.error("--target-world must be >= 1")
    reshard_mod = _load_reshard_module()
    reshard_mod.reshard_mode_c_shards(args.src, args.dst, args.target_world)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
