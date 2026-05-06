"""Standalone NCCL benchmark driver for ProTrain's profiler.

Runs ``axolotl.integrations.protrain.profiler.hw_bench.measure_nccl`` under a
proper distributed rendezvous and writes the resulting (gather, reduce)
payload tables to a JSON file. Intended for offline calibration when no
training loop is active — ``plugin.post_model_load`` and ``run_trace``
fire BEFORE the trainer's distributed rendezvous (or in single-rank /
no-PG environments entirely), so they cannot themselves drive a
collective benchmark. NCCL measurements are late-bound: they happen
post-distributed-init via ``plugin.post_trainer_create``, which triggers
``_remeasure_nccl_and_research`` on the freshly initialized process
group. Production traces therefore capture NCCL inline only after dist
init; this script fills the gap when that path is unavailable (offline
calibration on a non-training host, or seeding the cache before a run).

Two ways to invoke:

1. Multi-process via ``torchrun``::

    CUDA_VISIBLE_DEVICES=1,4,5,7 CUDA_DEVICE_ORDER=PCI_BUS_ID \\
        torchrun --standalone --nproc_per_node=4 \\
        scripts/protrain/measure_nccl.py \\
        --output scripts/nccl_results_world4.json

2. Single-spawn (this script self-spawns subprocesses)::

    CUDA_VISIBLE_DEVICES=1,4,5,7 CUDA_DEVICE_ORDER=PCI_BUS_ID \\
        python scripts/protrain/measure_nccl.py \\
        --world-size 4 --output scripts/nccl_results_world4.json

The resulting JSON has two top-level keys, ``gather`` and ``reduce``,
each mapping payload-bytes (string-coerced) to median collective
seconds. ``cost/runtime.py`` keys its communication-cost lookups on
the same payload-byte grid.

Output is written only by rank 0; other ranks exit silently.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404 — script self-spawns under torchrun by design
import sys
from pathlib import Path


def _run_as_rank() -> None:
    """Body executed under torchrun (env vars RANK/WORLD_SIZE/LOCAL_RANK set)."""
    import torch
    import torch.distributed as dist

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    if not torch.cuda.is_available():
        print(
            f"[rank {rank}] CUDA unavailable; NCCL benchmark needs GPUs.",
            file=sys.stderr,
        )
        sys.exit(1)
    torch.cuda.set_device(local_rank)
    backend = "nccl"
    dist.init_process_group(backend=backend)

    from axolotl.integrations.protrain.profiler.hw_bench import measure_nccl

    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results (rank 0 only). "
        "Defaults to ``scripts/nccl_results_world<N>.json``.",
    )
    parser.add_argument("--n-iters", type=int, default=8)
    parser.add_argument("--n-warmup", type=int, default=2)
    args = parser.parse_args()

    if rank == 0:
        print(
            f"[rank 0] measuring NCCL collectives under world_size={world_size} "
            f"(backend={backend}, n_iters={args.n_iters}, n_warmup={args.n_warmup})",
            file=sys.stderr,
        )

    try:
        gather_table, reduce_table = measure_nccl(
            world_size=world_size,
            n_iters=args.n_iters,
            n_warmup=args.n_warmup,
        )

        if rank == 0:
            out_path = Path(
                args.output
                if args.output is not None
                else f"scripts/nccl_results_world{world_size}.json"
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "world_size": world_size,
                "backend": backend,
                "gather": {str(k): v for k, v in gather_table.items()},
                "reduce": {str(k): v for k, v in reduce_table.items()},
                "n_iters": args.n_iters,
                "n_warmup": args.n_warmup,
            }
            out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"[rank 0] wrote {out_path}", file=sys.stderr)
            # Pretty summary
            print(
                "\nNCCL results (world={}):\n  payload (MiB)  gather (ms)  reduce (ms)".format(
                    world_size
                )
            )
            for size in sorted(gather_table.keys()):
                print(
                    f"  {size >> 20:>13}  {gather_table[size] * 1000:>10.3f}  "
                    f"{reduce_table[size] * 1000:>10.3f}"
                )
    finally:
        # No barrier here: a rank-local `success` gate would deadlock if ranks
        # disagree on status, and the output logic above already completes
        # before teardown (only rank 0 writes results, independently of peers).
        # destroy_process_group() always runs to release NCCL state.
        dist.destroy_process_group()


def _self_spawn(world_size: int, extra_args: list[str]) -> int:
    """Re-launch this script under torchrun for the requested world_size."""
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={world_size}",
        __file__,
        *extra_args,
    ]
    print("[self-spawn]", " ".join(cmd), file=sys.stderr)
    return subprocess.call(cmd)  # nosec B603  # noqa: S603 — argv built from sys.executable + this script's own __file__


def main() -> None:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        _run_as_rank()
        return

    # Self-spawn path: parse --world-size, hand off to torchrun.
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument(
        "--world-size",
        type=int,
        default=None,
        help="World size to spawn. Required when not invoked under torchrun.",
    )
    parser.add_argument("--n-iters", type=int, default=8)
    parser.add_argument("--n-warmup", type=int, default=2)
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write JSON results. Defaults to "
        "``scripts/nccl_results_world<N>.json``.",
    )
    args = parser.parse_args()
    if args.world_size is None or args.world_size < 1:
        parser.error(
            "--world-size is required when running outside torchrun "
            "(env vars RANK/WORLD_SIZE not set)."
        )
    if args.world_size == 1:
        # Single-rank just returns empty tables; emit them directly.
        from axolotl.integrations.protrain.profiler.hw_bench import measure_nccl

        gather_table, reduce_table = measure_nccl(
            world_size=1,
            n_iters=args.n_iters,
            n_warmup=args.n_warmup,
        )
        out = {
            "world_size": 1,
            "backend": "single-rank",
            "gather": {str(k): v for k, v in gather_table.items()},
            "reduce": {str(k): v for k, v in reduce_table.items()},
            "n_iters": args.n_iters,
            "n_warmup": args.n_warmup,
        }
        out_path = Path(
            args.output
            if args.output is not None
            else "scripts/nccl_results_world1.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2, sort_keys=True))
        print(f"wrote {out_path} (empty tables — single-rank)", file=sys.stderr)
        return

    # Forward calibration knobs to the spawned ranks so multi-rank runs
    # honour the same --n-iters / --n-warmup / --output values parsed here.
    forwarded = [
        "--n-iters",
        str(args.n_iters),
        "--n-warmup",
        str(args.n_warmup),
    ]
    if args.output is not None:
        forwarded.extend(["--output", args.output])
    rc = _self_spawn(args.world_size, forwarded)
    sys.exit(rc)


if __name__ == "__main__":
    main()
