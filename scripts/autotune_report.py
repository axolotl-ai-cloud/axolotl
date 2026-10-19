"""Summarize axolotl autotune cache files: winners per kernel and how far a narrowed search could go.

    python scripts/autotune_report.py outputs/autotune-b200.json [more.json ...] [--within 0.05]

The SonicMoE/quack file written next to a Triton cache (``<stem>.quack.json``) is read too when it
exists. For every kernel this prints the number of shape keys, how each winner was chosen, the
winners, and for keys with recorded timings the best-time range, the slowdown of the slowest
working config, failed configs, and the configs that stay within ``--within`` of the best time on
every key they were measured on (the empirical candidate set), so a profile can be pinned or
narrowed with evidence instead of the full grid.
"""

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict


def _paths(given):
    paths = []
    for path in given:
        paths.append(path)
        sibling = f"{os.path.splitext(path)[0]}.quack.json"
        if not path.endswith(".quack.json") and os.path.exists(sibling):
            paths.append(sibling)
    return list(dict.fromkeys(paths))


def _report_kernel(name, records, within):
    sources = Counter(record.get("source", "unknown") for record in records)
    print(
        f"\n## {name}: {len(records)} key(s) ("
        + ", ".join(f"{n} {source}" for source, n in sources.most_common())
        + ")"
    )
    winners = Counter(
        json.dumps(record["config"], sort_keys=True) for record in records
    )
    for config, count in winners.most_common():
        print(f"  winner x{count}: {config}")
    timed = [record for record in records if record.get("timings_ms")]
    if not timed:
        print("  (no timing tables recorded)")
        return
    best = [min(record["timings_ms"].values()) for record in timed]
    spread = [
        max(record["timings_ms"].values()) / min(record["timings_ms"].values())
        for record in timed
        if min(record["timings_ms"].values()) > 0
    ]
    benchmarked = sum(
        record.get("configs_benchmarked", len(record["timings_ms"])) for record in timed
    )
    failed = sum(record.get("failed_configs", 0) for record in timed)
    print(
        f"  {len(timed)} timed key(s): best {min(best):.4g}-{max(best):.4g} ms; "
        f"slowest/best median {statistics.median(spread) if spread else float('nan'):.2f}x, "
        f"max {max(spread) if spread else float('nan'):.2f}x; "
        f"{failed}/{benchmarked} benchmarked configs failed"
    )
    good_everywhere = None
    grid = 0
    for record in timed:
        table = record["timings_ms"]
        grid = max(grid, len(table))
        fastest = min(table.values())
        good = {c for c, t in table.items() if t <= fastest * (1 + within)}
        good_everywhere = good if good_everywhere is None else good_everywhere & good
    print(
        f"  grid size {grid}; configs within {within:.0%} of best on all "
        f"{len(timed)} timed key(s): {len(good_everywhere or [])}"
    )
    for config in sorted(good_everywhere or []):
        print(f"    {config}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument(
        "--within", type=float, default=0.05, help="relative slack vs the best time"
    )
    args = parser.parse_args()
    for path in _paths(args.paths):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        packages = ", ".join(f"{k}={v}" for k, v in data.get("packages", {}).items())
        print(
            f"# {path}: {len(data['records'])} records, gpu={data.get('gpu_name')}, "
            f"torch={data.get('torch')}{', ' + packages if packages else ''}, "
            f"written={data.get('written_at')}"
        )
        per_kernel = defaultdict(list)
        for record in data["records"]:
            backend = record.get("backend")
            name = (
                record["kernel"] if not backend else f"{record['kernel']} [{backend}]"
            )
            per_kernel[(record.get("module_fqn") or record.get("module"), name)].append(
                record
            )
        names = Counter(name for _module, name in per_kernel)
        for (module, name), records in sorted(per_kernel.items()):
            label = name if names[name] == 1 else f"{name} ({module})"
            _report_kernel(label, records, args.within)
        print()


if __name__ == "__main__":
    main()
