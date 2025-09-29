#!/usr/bin/env python
"""Verify a frozen prompted multitask dataset manifest before training.

Usage:
  python scripts/verify_prompted_manifest.py --data DATA.jsonl --manifest DATA.manifest.json

Exit codes:
  0 success
  2 verification failure (mismatch)
  3 other error (file missing, JSON error)
"""
from __future__ import annotations
import argparse
import sys
import json
from pathlib import Path

# local import path safety
import os
import sys as _sys
if 'axolotl' not in _sys.modules:
    _sys.path.insert(0, os.path.abspath('src'))

from axolotl.data.weighted_prompted_dataset import verify_prompted_manifest  # type: ignore

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--non-strict', action='store_true', help='Allow tiny floating point diffs in weight sums')
    args = ap.parse_args()
    try:
        manifest = verify_prompted_manifest(args.data, args.manifest, strict_weights=not args.non_strict)
    except ValueError as e:
        print(f"VERIFICATION FAILED: {e}", file=sys.stderr)
        return 2
    except Exception as e:  # pylint: disable=broad-except
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    print("Manifest verification passed.")
    print(json.dumps(manifest, indent=2))
    return 0

if __name__ == '__main__':
    sys.exit(main())
