"""Build a balanced strategy eval split from enriched v3.

Selects examples across yard buckets (<230, 230-260, 260-290, >=290) based on
tee_shot_strategy_options cutoff distances. Emits strategy_selection prompts that
mirror training prompts and include the explicit list of available cutoffs.

Usage:
  python scripts/build_balanced_eval_split.py \
    --enriched data/bethpage_black/strategy_augmented_holes_enriched.jsonl \
    --out data/bethpage_black/test_enriched_balanced.jsonl \
    --per-bucket 25
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BUCKETS = [
    ("< 230", lambda c: c < 230),
    ("230-260", lambda c: 230 <= c <= 260),
    ("260-290", lambda c: 260 < c <= 290),
    (">= 290", lambda c: c >= 290),
]


def load_enriched(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def facts_block(obj: dict) -> str:
    attr = obj.get("attributes", {}) or {}
    notes = obj.get("derived_notes", {}) or {}
    parts = [f"Hole {obj.get('hole')} | Par {obj.get('par')} | Yardage {obj.get('yardage')}"]
    def add(label, key):
        v = attr.get(key)
        if v:
            parts.append(f"{label}: {v}")
    add("Gradient", "gradient")
    add("Shape", "hole_shape")
    add("PrimaryHazard", "primary_hazard_type")
    add("PreferredMiss", "preferred_miss")
    if attr.get("key_hazards"):
        parts.append("KeyHazards: " + "; ".join(attr["key_hazards"]))
    if notes.get("elevation_adjust_note"):
        parts.append(f"ElevationNote: {notes['elevation_adjust_note']}")
    return "\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-bucket", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    random.seed(args.seed)

    bucket_samples = {name: [] for name, _ in BUCKETS}
    for obj in load_enriched(Path(args.enriched)):
        strategies = obj.get("tee_shot_strategy_options") or []
        cutoffs = [s.get("cutoff_distance") for s in strategies if isinstance(s.get("cutoff_distance"), int)]
        if not cutoffs:
            continue
        cutoffs_sorted = sorted(set(cutoffs))
        cutoff_list_text = ", ".join(f"{c} yards" for c in cutoffs_sorted)
        facts = facts_block(obj)
        for c in cutoffs_sorted:
            for name, pred in BUCKETS:
                if pred(c):
                    prompt = (
                        f"{facts}\n\nTee Strategy Selection Task:\n"
                        f"Player average drive: {c} yards. Available tee strategy cutoffs: {cutoff_list_text}.\n"
                        "Return only: 'Strategy: <number> yards' for the recommended tee shot cutoff distance."
                    )
                    ex = {
                        "task_type": "strategy_selection",
                        "prompt": prompt,
                        "completion": f"Strategy: {c} yards",
                        "expected_cutoff_yards": c,
                        "hole": obj.get("hole"),
                    }
                    bucket_samples[name].append(ex)
                    break

    # sample per bucket
    out = []
    for name, _ in BUCKETS:
        picks = bucket_samples[name]
        random.shuffle(picks)
        out.extend(picks[: args.per_bucket])

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in out:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(out)} examples across buckets -> {out_path}")


if __name__ == "__main__":
    main()
