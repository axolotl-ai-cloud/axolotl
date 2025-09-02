#!/usr/bin/env python
"""
Sort a JSONL dataset by the numeric 'hole' field and write to a new file.

Defaults are set to the Bethpage Black multitask dataset in this repo.
Within each hole group, original order is preserved (stable sort).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                items.append(obj)
            except json.JSONDecodeError as e:
                raise SystemExit(f"JSON decode error at {path}:{lineno}: {e}")
    return items


def write_jsonl(path: Path, items: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False))
            f.write("\n")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    default_in = repo_root / "data" / "bethpage_black" / "train_multitask_case_fixed.jsonl"
    default_out = repo_root / "data" / "bethpage_black" / "train_multitask_sorted_by_hole.jsonl"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", "-i", type=Path, default=default_in, help="Path to input JSONL")
    p.add_argument("--output", "-o", type=Path, default=default_out, help="Path to output JSONL")
    p.add_argument(
        "--desc", action="store_true", help="Sort holes in descending order (default ascending)"
    )
    args = p.parse_args()

    inp: Path = args.input
    out: Path = args.output
    desc: bool = args.desc

    if not inp.exists():
        raise SystemExit(f"Input file not found: {inp}")

    items = read_jsonl(inp)

    # Stable sort by numeric 'hole' only; entries lacking 'hole' go to the end.
    def hole_key(obj: Dict[str, Any]) -> int:
        try:
            return int(obj.get("hole", 10_000))
        except (TypeError, ValueError):
            return 10_000

    items_sorted = sorted(items, key=hole_key, reverse=desc)

    write_jsonl(out, items_sorted)

    # Emit a small summary by hole to aid quick review.
    counts = Counter(hole_key(o) for o in items_sorted)
    holes = sorted(k for k in counts.keys() if k != 10_000)
    print(f"Wrote sorted JSONL: {out}")
    print(f"Total records: {len(items_sorted)}")
    print("Records per hole:")
    for h in holes:
        print(f"  Hole {h:>2}: {counts[h]}")
    if 10_000 in counts:
        print(f"  (no hole): {counts[10_000]}")


if __name__ == "__main__":
    main()
