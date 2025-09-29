import json
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).parent
BASICS_PATH = BASE_DIR / "basics.jsonl"
DESCRIPTIONS_PATH = BASE_DIR / "data_template_5desc_merged.jsonl"
OUTPUT_PATH = BASE_DIR / "strategy_augmented_holes.jsonl"

# Fields from description file we want to carry over verbatim if present
DESCRIPTION_ATTRIBUTE_FIELDS = [
    "water_hazard",
    "gradient",
    "hole_length",
    "hole_shape",
    "elevation_delta_yards",
    "tee_landing_zone_width_yards",
    "primary_hazard_type",
    "secondary_hazard_type",
    "green_size",
    "preferred_miss",
    "punish_miss",
    "strategic_theme",
    "key_hazards",
    "green_complex_notes",
]

DESCRIPTION_VARIANT_PREFIX = "new_description_"


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def gather_strategy_options(basics_rows):
    per_hole = defaultdict(lambda: {"par": None, "yardage": None, "strategies": [], "raw_examples": []})
    for row in basics_rows:
        hole = row["hole"]
        entry = per_hole[hole]
        # Prefer first encountered par/yardage (assumed consistent); if mismatch, keep first and ignore
        if entry["par"] is None:
            entry["par"] = row.get("par")
        if entry["yardage"] is None:
            entry["yardage"] = row.get("yardage")
        # accumulate raw examples for potential future derived fields (gold selection heuristics etc.)
        entry["raw_examples"].append(row)
        for strat in row.get("tee_shot_strategies", []):
            sid = strat.get("strategy_id")
            if sid and all(existing.get("strategy_id") != sid for existing in entry["strategies"]):
                entry["strategies"].append(strat)
    # sort strategies per hole by cutoff_distance numeric if present
    for hole, entry in per_hole.items():
        entry["strategies"].sort(key=lambda s: s.get("cutoff_distance", 0))
    return per_hole


def build_unified_records(desc_rows, strategy_index):
    unified = []
    for row in desc_rows:
        hole = row["hole"]
        strat_entry = strategy_index.get(hole, {})
        descriptions = []
        style_tags = row.get("style_tags") or []
        # collect 1..5 description variants (assumes consecutive and present)
        variant_num = 1
        while True:
            key = f"{DESCRIPTION_VARIANT_PREFIX}{variant_num}"
            if key in row:
                descriptions.append(row[key].strip())
                variant_num += 1
            else:
                break
        record = {
            "hole": hole,
            "par": strat_entry.get("par", row.get("par")),
            "yardage": strat_entry.get("yardage", row.get("yardage")),
            "attributes": {fld: row.get(fld) for fld in DESCRIPTION_ATTRIBUTE_FIELDS},
            "tee_shot_strategy_options": strat_entry.get("strategies", []),
            "descriptions": descriptions,
            "style_tags": style_tags,
            # Placeholder future derived fields
            "gold_description_variant_id": None,
            "derived_notes": {},
            "_meta": {
                "raw_strategy_examples_count": len(strat_entry.get("raw_examples", [])),
            },
        }
        unified.append(record)
    # sort by hole for determinism
    unified.sort(key=lambda r: r["hole"])
    return unified


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    basics_rows = load_jsonl(BASICS_PATH)
    desc_rows = load_jsonl(DESCRIPTIONS_PATH)
    strategy_index = gather_strategy_options(basics_rows)
    unified = build_unified_records(desc_rows, strategy_index)
    write_jsonl(OUTPUT_PATH, unified)
    print(f"Wrote {len(unified)} unified hole records to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
