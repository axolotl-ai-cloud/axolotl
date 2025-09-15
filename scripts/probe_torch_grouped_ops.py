#!/usr/bin/env python
"""
Probe PyTorch for grouped GEMM operator names and namespaces.
Run: python scripts/probe_torch_grouped_ops.py
"""

import sys


def main():
    try:
        import torch
    except Exception as e:
        print("Failed to import torch:", e)
        sys.exit(1)

    print("torch version:", torch.__version__)
    namespaces = [n for n in dir(torch.ops) if not n.startswith("_")]
    print("ops namespaces:", namespaces)

    found_any = False
    for ns in namespaces:
        obj = getattr(torch.ops, ns, None)
        ops = []
        if obj is not None:
            try:
                ops = dir(obj)
            except Exception as e:
                print(f"warning: failed to list ops for namespace {ns}: {e}")
        cands = [
            o
            for o in ops
            if ("group" in o.lower())
            or ("mm_grouped" in o.lower())
            or ("matmul_grouped" in o.lower())
            or ("grouped" in o.lower())
        ]
        if cands:
            found_any = True
            print(f"namespace {ns} candidates:", cands)

    if not found_any:
        print("No grouped GEMM candidates found. PyTorch >= 2.8 is recommended.")


if __name__ == "__main__":
    main()
