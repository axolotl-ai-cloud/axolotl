#!/usr/bin/env python
"""Demonstrate stratified batch/window mixing on WeightedPromptedIterableDataset.

For a given window size, prints task composition of each window to confirm
balanced coverage (round-robin across tasks) and optional intra-window shuffle.

Usage:
  python scripts/demo_stratified_windows.py \
    --data <dataset.jsonl> --epoch-size 40 --window-size 8
"""
from __future__ import annotations
import argparse
from collections import Counter
import os, sys

if 'axolotl' not in sys.modules:
    sys.path.insert(0, os.path.abspath('src'))

from axolotl.data.weighted_prompted_dataset import WeightedPromptedIterableDataset  # type: ignore


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--epoch-size', type=int, default=80)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--window-size', type=int, default=8)
    ap.add_argument('--quota', action='store_true', help='Enable deterministic per-task quotas (enforce_quota)')
    ap.add_argument('--no-shuffle', action='store_true')
    args = ap.parse_args()

    ds = WeightedPromptedIterableDataset(
        path=args.data,
        epoch_size=args.epoch_size,
        seed=args.seed,
        stratify_window_size=args.window_size,
        stratify_shuffle=not args.no_shuffle,
        enforce_quota=args.quota,
    )

    window = []
    win_id = 0
    counts_global = Counter()
    for rec in ds:
        window.append(rec)
        counts_global[rec['task']] += 1
        if len(window) == args.window_size:
            win_counts = Counter(r['task'] for r in window)
            print(f"Window {win_id}:" + ' '.join(f" {t}={win_counts[t]}" for t in sorted(win_counts)))
            window = []
            win_id += 1
    if window:
        win_counts = Counter(r['task'] for r in window)
        print(f"Window {win_id}:" + ' '.join(f" {t}={win_counts[t]}" for t in sorted(win_counts)))
    print('Global counts:', dict(counts_global))

if __name__ == '__main__':
    main()
