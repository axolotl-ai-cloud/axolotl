"""Summarize an axolotl autotune cache file: winners per kernel and how far a narrowed search could go.

    python scripts/autotune_report.py outputs/autotune-b200.json [--within 0.05]

For every kernel with recorded timings, prints the configs that stay within ``--within`` of the
best time on every key they were measured on (the empirical candidate set), so a profile can be
pinned or narrowed with evidence instead of the full grid.
"""

import argparse
import json
from collections import defaultdict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument(
        "--within", type=float, default=0.05, help="relative slack vs the best time"
    )
    args = parser.parse_args()
    data = json.load(open(args.path))
    print(
        f"{args.path}: {len(data['records'])} records, gpu={data.get('gpu_name')}, written={data.get('written_at')}"
    )
    per_kernel = defaultdict(list)
    for record in data["records"]:
        per_kernel[record["kernel"]].append(record)
    for kernel, records in sorted(per_kernel.items()):
        winners = defaultdict(int)
        for record in records:
            winners[json.dumps(record["config"], sort_keys=True)] += 1
        print(f"\n## {kernel}: {len(records)} key(s)")
        for config, count in sorted(winners.items(), key=lambda kv: -kv[1]):
            print(f"  winner x{count}: {config}")
        timed = [r for r in records if r.get("timings_ms")]
        if not timed:
            print(
                "  (no timing tables recorded; run with autotune_cache_path on this branch to collect them)"
            )
            continue
        good_everywhere = None
        grid = 0
        for record in timed:
            table = record["timings_ms"]
            grid = max(grid, len(table))
            best = min(table.values())
            good = {c for c, t in table.items() if t <= best * (1 + args.within)}
            good_everywhere = (
                good if good_everywhere is None else good_everywhere & good
            )
        print(
            f"  grid size {grid}; configs within {args.within:.0%} of best on all {len(timed)} timed key(s): {len(good_everywhere or [])}"
        )
        for config in sorted(good_everywhere or []):
            print(f"    {config}")


if __name__ == "__main__":
    main()
