#!/usr/bin/env python
"""Run the experimental NVFP4 CUDA graph training loop."""

from __future__ import annotations

import argparse
import importlib.machinery
import os
import sys
import types


def _positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError(f"must be > 0 (got {value})")
    return ivalue


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prototype static-batch NVFP4 training loop outside HF Trainer."
    )
    parser.add_argument("config", help="Axolotl YAML config")
    parser.add_argument("--gpu", type=int, default=None, help="physical GPU index")
    parser.add_argument("--steps", type=_positive_int, default=20)
    parser.add_argument("--warmup", type=_positive_int, default=5)
    parser.add_argument("--capture-warmup", type=_positive_int, default=3)
    parser.add_argument("--mode", choices=["eager", "graph", "auto"], default="auto")
    parser.add_argument(
        "--reuse-static-batch",
        action="store_true",
        help="replay/copy one batch only; useful for capture feasibility probes",
    )
    compile_group = parser.add_mutually_exclusive_group()
    compile_group.add_argument("--compile", dest="compile_model", action="store_true")
    compile_group.add_argument(
        "--no-compile", dest="compile_model", action="store_false"
    )
    parser.set_defaults(compile_model=None)
    parser.add_argument("--fullgraph", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument(
        "--no-probe-on-fail", dest="probe_on_fail", action="store_false"
    )
    parser.set_defaults(probe_on_fail=True)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.gpu is not None:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    if not os.environ.get("AXOLOTL_NVFP4_GRAPH_ALLOW_TE_IMPORT"):
        te_stub = types.ModuleType("transformer_engine")
        te_stub.__spec__ = importlib.machinery.ModuleSpec("transformer_engine", None)
        sys.modules.setdefault("transformer_engine", te_stub)

    from axolotl.cli.config import load_cfg
    from axolotl.utils.nvfp4_cuda_graph_loop import (
        GraphLoopOptions,
        format_result,
        run_loop,
    )

    cfg = load_cfg(args.config)
    cfg.max_steps = args.steps
    options = GraphLoopOptions(
        mode=args.mode,
        steps=args.steps,
        warmup_steps=args.warmup,
        capture_warmup_steps=args.capture_warmup,
        reuse_static_batch=args.reuse_static_batch,
        compile_model=args.compile_model,
        fullgraph=args.fullgraph,
        probe_on_fail=args.probe_on_fail,
        probe_only=args.probe_only,
    )
    result = run_loop(cfg, options)
    print(format_result(result))
    return 0 if (result.graph_captured or args.mode != "graph") else 1


if __name__ == "__main__":
    sys.exit(main())
