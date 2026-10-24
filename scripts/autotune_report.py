"""Summarize axolotl autotune cache files: winners per kernel and how far a narrowed search could go.

    python scripts/autotune_report.py outputs/autotune-b200.json [more.json ...] [--within 0.05]
        [--quack-cache-dir $QUACK_CACHE_DIR] [--min-sweep-seconds 60] [--top 1]
        [--check-allow config.yaml]

The SonicMoE/quack file written next to a Triton cache (``<stem>.quack.json``) is read too when it
exists. For every kernel this prints the number of shape keys, how each winner was chosen, the
winners, and for keys with recorded timings the best-time range, the slowdown of the slowest
working config, failed configs, sweep time, and the configs that stay within ``--within`` of the
best time on every key they were measured on (the empirical candidate set), so a profile can be
pinned or narrowed with evidence instead of the full grid.

It ends with a ready-to-paste ``autotune_profiles`` block covering only the measured keys whose
sweep (compiles included) took at least ``--min-sweep-seconds``, each matched to its exact shapes
and pinned to its winner (or its ``--top`` fastest configs); cheaper keys keep autotuning.
"""

import argparse
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict


def _paths(given):
    paths = []
    for path in given:
        sibling = f"{os.path.splitext(path)[0]}.quack.json"
        has_sibling = not path.endswith(".quack.json") and os.path.exists(sibling)
        if os.path.exists(path) or not has_sibling:
            paths.append(path)
        else:
            print(f"# {path}: not written yet (no Triton winners); reading {sibling}")
        if has_sibling:
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
    sweeps = [r["sweep_seconds"] for r in records if r.get("sweep_seconds") is not None]
    if sweeps:
        print(
            f"  sweep time: {len(sweeps)} key(s), max {max(sweeps):.1f} s, "
            f"total {sum(sweeps):.1f} s"
        )
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


def _load_quack_disk_cache(root):
    per_kernel = defaultdict(list)
    for dirpath, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(".autotune.json"):
                with open(os.path.join(dirpath, name), encoding="utf-8") as handle:
                    per_kernel[name[: -len(".autotune.json")]].append(json.load(handle))
    return per_kernel


def _disk_times(entry):
    times = {}
    failed = 0
    for config, timing in entry.get("configs_timings", []):
        ms = timing[0] if isinstance(timing, list) else timing
        if ms is None or not math.isfinite(float(ms)):
            failed += 1
        else:
            times[config] = float(ms)
    return times, failed


def _report_quack_disk_cache(root, within):
    """quack's own ``<QUACK_CACHE_DIR>/<hash>/<fn>.autotune.json`` files: every config's time."""
    per_kernel = _load_quack_disk_cache(root)
    print(f"# quack disk cache {root}: {sum(map(len, per_kernel.values()))} key(s)")
    for kernel, entries in sorted(per_kernel.items()):
        print(f"\n## {kernel} [quack disk cache]: {len(entries)} key(s)")
        for entry in entries:
            times, failed = _disk_times(entry)
            if not times:
                print(f"  key {entry.get('key')}: every config failed")
                continue
            best_config, best = min(times.items(), key=lambda kv: kv[1])
            near = sum(1 for t in times.values() if t <= best * (1 + within))
            print(
                f"  key {entry.get('key')}\n    best {best:.4g} ms {best_config}; "
                f"{near} within {within:.0%}; {failed}/{failed + len(times)} failed"
            )
    print(
        "# (quack disk-cache keys carry no sweep time or argument names, so they are not "
        "turned into profiles)"
    )


def _profile_config(config):
    """A winner as an ``autotune_profiles`` config: quack ``GemmConfig`` fields flattened."""
    if (
        isinstance(config, dict)
        and list(config) == ["config"]
        and isinstance(config["config"], dict)
    ):
        return config["config"]
    return config


def _profile_entry(record, gpu, top):
    if record.get("backend") == "quack":
        match = record.get("call_args")
        if not match:
            return None
    else:
        match = {k: v for k, v in record["key"].items() if k != "_extra"}
    configs = [_profile_config(record["config"])]
    if top > 1 and record.get("timings_ms"):
        configs = [_profile_config(json.loads(c)) for c in record["timings_ms"]][:top]
    lines = [f"  - kernel: {record['kernel']}"]
    if record.get("backend") == "quack":
        lines.append("    backend: quack")
    if gpu:
        lines.append(f"    gpu: {json.dumps(gpu)}")
    lines.append(f"    match: {json.dumps(match)}")
    lines.append(f"    # sweep took {record['sweep_seconds']:.0f} s")
    lines.append("    configs:")
    lines.extend(f"      - {json.dumps(c)}" for c in configs)
    return "\n".join(lines)


def _profile_snippet(datasets, min_seconds, top):
    entries, no_sweep, no_args = [], 0, 0
    for data in datasets:
        gpu = (data.get("gpu_name") or "").replace("NVIDIA ", "")
        for record in data["records"]:
            if record.get("source") != "measured":
                continue
            seconds = record.get("sweep_seconds")
            if seconds is None:
                no_sweep += 1
                continue
            if seconds < min_seconds:
                continue
            entry = _profile_entry(record, gpu, top)
            if entry is None:
                no_args += 1
                continue
            entries.append((seconds, entry))
    print(
        f"# autotune_profiles for keys whose sweep took >= {min_seconds:g} s "
        f"({'winner only' if top <= 1 else f'{top} fastest configs'})"
    )
    if no_sweep:
        print(f"# skipped {no_sweep} measured key(s) with no recorded sweep_seconds")
    if no_args:
        print(f"# skipped {no_args} quack key(s) with no call_args to match on")
    if not entries:
        print("# (no key qualifies)")
        return
    print("autotune_profiles:")
    for _seconds, entry in sorted(entries, key=lambda e: -e[0]):
        print(entry)


_FIELD = re.compile(r"(\w+)=([^,()]+)")


def _parse_gemm_config(text):
    """``config: GemmConfig(tile_m=256, ...)`` (quack's disk-cache spelling) -> fields."""
    fields = {}
    for name, raw in _FIELD.findall(text):
        raw = raw.strip()
        if raw in ("True", "False"):
            fields[name] = raw == "True"
        elif raw == "None":
            fields[name] = None
        else:
            try:
                fields[name] = int(raw)
            except ValueError:
                fields[name] = raw.strip("'\"")
    return fields


def _measured_keys(datasets, disk):
    """(kernel, gpu, call_args or None, label, {fields json: ms}) per measured quack key."""
    for data in datasets:
        for record in data["records"]:
            if record.get("backend") != "quack" or not record.get("timings_ms"):
                continue
            times = {
                json.dumps(_profile_config(json.loads(c)), sort_keys=True): t
                for c, t in record["timings_ms"].items()
            }
            label = json.dumps(record.get("call_args") or record["key"]["signature"])
            yield (
                record["kernel"],
                data.get("gpu_name"),
                record.get("call_args"),
                label,
                times,
            )
    for kernel, entries in disk.items():
        for entry in entries:
            times, _failed = _disk_times(entry)
            times = {
                json.dumps(_parse_gemm_config(c), sort_keys=True): t
                for c, t in times.items()
            }
            yield kernel, None, None, json.dumps(entry.get("key")), times


def _check_allow(config_path, datasets, disk, within):
    """List the measured keys whose winner (or a config within ``within`` of it) a profile's
    ``allow`` filter would prune. Returns the number of pruned winners."""
    import fnmatch
    from types import SimpleNamespace

    import yaml

    from axolotl.integrations.kernels.autotune_profiles import (
        _quack_matches,
        config_allowed,
    )

    with open(config_path, encoding="utf-8") as handle:
        profiles = yaml.safe_load(handle).get("autotune_profiles") or []
    pruned_winners = 0
    for index, profile in enumerate(profiles):
        allow = profile.get("allow")
        if profile.get("backend") != "quack" or not allow:
            continue
        print(f"\n## check allow: profile {index} ({profile['kernel']})")
        checked = unchecked_match = 0
        for kernel, gpu, call_args, label, times in _measured_keys(datasets, disk):
            if not fnmatch.fnmatchcase(kernel, profile["kernel"]):
                continue
            wanted_gpu = profile.get("gpu")
            if wanted_gpu and gpu and wanted_gpu.lower() not in gpu.lower():
                continue
            if profile.get("match"):
                if call_args is None:
                    unchecked_match += 1
                else:
                    args = {
                        name.removesuffix(".shape"): (
                            SimpleNamespace(shape=value)
                            if name.endswith(".shape")
                            else value
                        )
                        for name, value in call_args.items()
                    }
                    if not _quack_matches(profile["match"], args):
                        continue
            if not times:
                continue
            checked += 1
            best = min(times.values())
            winner = min(times, key=times.get)
            near = [c for c, t in times.items() if t <= best * (1 + within)]
            dropped = [c for c in near if not config_allowed(json.loads(c), allow)]
            if winner in dropped:
                pruned_winners += 1
                print(f"  PRUNES WINNER {best:.4g} ms {winner}\n    key {label}")
            elif dropped:
                print(
                    f"  prunes {len(dropped)}/{len(near)} configs within {within:.0%} "
                    f"of best\n    key {label}"
                )
                for config in dropped:
                    print(f"    {times[config]:.4g} ms {config}")
        print(f"  {checked} measured key(s) checked")
        if unchecked_match:
            print(
                f"  {unchecked_match} of them from quack's disk cache: no argument names, "
                "so match was not applied"
            )
    return pruned_winners


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="*")
    parser.add_argument(
        "--within", type=float, default=0.05, help="relative slack vs the best time"
    )
    parser.add_argument(
        "--quack-cache-dir", help="also summarize quack's own *.autotune.json files"
    )
    parser.add_argument(
        "--min-sweep-seconds",
        type=float,
        default=60.0,
        help="emit autotune_profiles only for keys whose sweep took at least this long",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=1,
        help="configs per emitted profile: 1 pins the winner, N keeps the N fastest",
    )
    parser.add_argument(
        "--check-allow",
        metavar="CONFIG_YAML",
        help="list measured keys whose winner (or a config within --within of it) the "
        "config's quack allow filters would prune; exits 1 if a winner would be",
    )
    args = parser.parse_args()
    datasets = []
    for path in _paths(args.paths):
        with open(path, encoding="utf-8") as handle:
            data = json.load(handle)
        datasets.append(data)
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
    if args.quack_cache_dir:
        _report_quack_disk_cache(args.quack_cache_dir, args.within)
        print()
    if datasets:
        _profile_snippet(datasets, args.min_sweep_seconds, args.top)
    if args.check_allow:
        disk = (
            _load_quack_disk_cache(args.quack_cache_dir) if args.quack_cache_dir else {}
        )
        if _check_allow(args.check_allow, datasets, disk, args.within):
            sys.exit(1)


if __name__ == "__main__":
    main()
