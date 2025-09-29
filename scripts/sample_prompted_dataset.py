tyg\
#!/usr/bin/env python
"""Weighted epoch sampler for the prompted multitask dataset.

Purpose:
Generate an epoch sample (JSONL) from a deduplicated prompted dataset using
probabilities proportional to sample_weight for each record.

Features:
- With-replacement sampling to allow fractional effective weights.
- Optional temperature to smooth or sharpen distribution.
- Reproducible via --seed.
- Streams output JSONL of sampled records (original records unchanged) plus
  adds an 'epoch_instance' field (1-based occurrence index for that source record in this epoch).
- Summary stats printed to stderr (empirical counts, expected counts, KL divergence).

Usage:
  python scripts/sample_prompted_dataset.py --data DATA.jsonl --epoch-size 1000 --out epoch_1.jsonl

If --out omitted, writes to stdout.
"""
from __future__ import annotations
import argparse
import json
import math
import random
import sys
from pathlib import Path
from collections import Counter


def load_records(path: Path):
    recs = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line:
                continue
            rec = json.loads(line)
            recs.append(rec)
    return recs


def compute_probs(records, temperature: float):
    weights = [float(r.get('sample_weight', 1.0)) for r in records]
    if temperature != 1.0:
        # apply temperature scaling: p_i ^ (1/temperature) then renormalize
        # temperature >1 flattens, <1 sharpens
        scaled = [w ** (1.0/temperature) for w in weights]
    else:
        scaled = weights
    total = sum(scaled)
    if total <= 0:
        raise ValueError('Non-positive total weight.')
    return [w/total for w in scaled]


def sample_epoch(records, probs, epoch_size: int, seed: int):
    rng = random.Random(seed)
    # cumulative distribution
    cumulative = []
    c = 0.0
    for p in probs:
        c += p
        cumulative.append(c)
    out = []
    appearances = Counter()
    for _ in range(epoch_size):
        r = rng.random()
        # binary search
        lo, hi = 0, len(cumulative)-1
        while lo < hi:
            mid = (lo+hi)//2
            if r <= cumulative[mid]:
                hi = mid
            else:
                lo = mid+1
        idx = lo
        rec = records[idx]
        appearances[idx] += 1
        inst = json.loads(json.dumps(rec))  # shallow copy via serialization
        inst['epoch_instance'] = appearances[idx]
        out.append(inst)
    return out, appearances


def kl_divergence(empirical_counts, probs, epoch_size):
    kl=0.0
    for i,p in enumerate(probs):
        if p == 0: continue
        q = empirical_counts.get(i,0)/epoch_size
        if q == 0: # smoothing (add tiny epsilon)
            q = 1e-12
        kl += p * math.log(p/q)
    return kl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path, help='Deduplicated prompted dataset JSONL')
    ap.add_argument('--epoch-size', type=int, required=True)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--temperature', type=float, default=1.0, help='>1 flatten, <1 sharpen weight distribution')
    ap.add_argument('--out', type=Path)
    ap.add_argument('--summary', action='store_true', help='Print per-task empirical vs expected counts')
    args = ap.parse_args()

    records = load_records(args.data)
    probs = compute_probs(records, args.temperature)
    sampled, appearances = sample_epoch(records, probs, args.epoch_size, args.seed)

    # compute per-task empirical counts
    task_expected = {}
    task_empirical = Counter()
    for i,(rec,p) in enumerate(zip(records, probs)):
        task = rec.get('task')
        task_expected.setdefault(task,0.0)
        task_expected[task] += p * args.epoch_size
    for idx,count in appearances.items():
        task = records[idx].get('task')
        task_empirical[task] += count

    kl = kl_divergence(appearances, probs, args.epoch_size)

    # output
    if args.out:
        out_f = args.out.open('w', encoding='utf-8')
    else:
        out_f = sys.stdout
    for rec in sampled:
        out_f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    if args.out:
        out_f.close()

    # summary to stderr
    sys.stderr.write(f"Epoch size: {args.epoch_size}\n")
    sys.stderr.write(f"Records: {len(records)}\n")
    sys.stderr.write(f"KL(emp||expected): {kl:.6f}\n")

    if args.summary:
        sys.stderr.write('\nPer-task counts (empirical vs expected):\n')
        all_tasks = sorted(task_expected.keys())
        for t in all_tasks:
            exp = task_expected[t]
            emp = task_empirical[t]
            sys.stderr.write(f"  {t}: empirical={emp} expected={exp:.2f} diff={emp-exp:.2f}\n")

if __name__ == '__main__':
    main()
