#!/usr/bin/env python
"""Freeze a prompted multitask dataset by generating a manifest with checksum & stats.

Manifest fields:
  source_path: original dataset file
  sha256: full file hash
  line_count: number of JSONL records
  task_counts: mapping of task -> count
  total_sample_weight: sum of sample_weight
  per_task_sample_weight: mapping task -> summed weight
  created_utc: ISO timestamp

Usage:
  python scripts/freeze_prompted_dataset.py --data DATA.jsonl --out DATA.manifest.json
"""
from __future__ import annotations
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from collections import Counter


def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, type=Path)
    ap.add_argument('--out', type=Path, help='Output manifest path (default: <data>.manifest.json)')
    args = ap.parse_args()

    data_path: Path = args.data
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    sha = hash_file(data_path)
    task_counts = Counter()
    task_weight = Counter()
    line_count = 0
    with data_path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line_count += 1
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                raise SystemExit(f"Invalid JSON on line {line_count}")
            t = rec.get('task', 'UNKNOWN')
            task_counts[t] += 1
            task_weight[t] += float(rec.get('sample_weight', 1.0))

    manifest = {
        'source_path': str(data_path),
        'sha256': sha,
        'line_count': line_count,
        'task_counts': dict(task_counts),
        'total_sample_weight': sum(task_weight.values()),
        'per_task_sample_weight': {k: round(v, 6) for k, v in task_weight.items()},
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'format': 'prompted_multitask_jsonl_v1'
    }

    out_path = args.out or data_path.with_suffix('.manifest.json')
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding='utf-8')
    print(f"Wrote manifest: {out_path}")
    print(json.dumps(manifest, indent=2) )


if __name__ == '__main__':
    main()
