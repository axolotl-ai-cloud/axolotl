"""Validate and transform enriched Bethpage Black template into training JSONL.

Reads data/bethpage_black/data_template.json (one JSON object per line, holes 1-18)
and emits two files:
  data/bethpage_black/descriptions_enriched.jsonl  (description_synthesis examples)
  data/bethpage_black/descriptions_enriched_stats.json  (summary stats)

We IGNORE user omissions for: wind_exposure, green_receptiveness (treated as None).

Each input line must include core structural fields plus 2-5 description fields:
    new_description_1 .. new_description_5 (only first existing ones used). At least 2 required.
    Optional: gold_description (int 1..5) to flag preferred variant (for weighting/oversampling downstream).

Validation rules (non-fatal produce warnings):
  * Duplicate phrasing overlap > 85% between new_description_1 and new_description_2 (Levenshtein ratio) -> warning.
  * Any description < 40 chars or > 1200 chars -> warning.
  * Missing nullable fields just set to None.

Errors (fail):
  * Hole outside 1..18 or duplicate hole.
  * Required non-null fields missing or empty strings.

Prompt Format Produced (one example per description, so 2 per hole):
  TASK: description_synthesis\n
  Use the structured facts below to write a vivid, specific, NOT generic hole description for Bethpage Black Hole {hole}. Avoid boilerplate like "combines strategic positioning". Do not repeat phrases verbatim from other holes. End after the description (no strategy recap section).\n\nFACTS:\n- Par: {par}\n- Yardage: {yardage_or_unknown}\n- Gradient: {gradient}\n- Elevation Delta (yards): {elevation_delta_yards}\n- Hole Length: {hole_length}\n- Shape: {hole_shape}\n- Primary Hazard: {primary_hazard_type or 'none'}\n- Secondary Hazard: {secondary_hazard_type or 'none'}\n- Key Hazards: {comma list}\n- Preferred Miss: {preferred_miss or 'n/a'}\n- Punish Miss: {punish_miss or 'n/a'}\n- Strategic Theme: {strategic_theme}\n- Green Size: {green_size}\n- Green Notes: {green_complex_notes}\n- Tee Landing Width (yds): {tee_landing_zone_width_yards or 'n/a'}\n- Elevation Adjustment: {('uphill'/'downhill'/ 'flat' from gradient)}\n
  DESCRIPTION:

Completion is the cleaned description (with leading/trailing whitespace stripped, single spaces).

We add fields: task_type=description_synthesis, split=train, variation_type=f"enriched_{desc_idx}", use_for_training=true

Run: python scripts/validate_and_prepare_bethpage_enriched.py --input data/bethpage_black/data_template.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple

try:
    from rapidfuzz import fuzz  # faster than python-Levenshtein; optional
    def similarity(a: str, b: str) -> float:
        return fuzz.token_sort_ratio(a, b) / 100.0
except Exception:  # pragma: no cover - fallback
    def similarity(a: str, b: str) -> float:  # naive fallback
        import difflib
        return difflib.SequenceMatcher(None, a, b).ratio()


REQUIRED_STRING_FIELDS = [
    "gradient",
    "hole_length",
    "hole_shape",
    "strategic_theme",
    "green_size",
    "green_complex_notes",
    "new_description_1",
    "new_description_2",
]

OPTIONAL_STRING_FIELDS = [
    "primary_hazard_type",
    "secondary_hazard_type",
    "preferred_miss",
    "punish_miss",
]

LIST_FIELDS = ["key_hazards"]


def normalize_space(s: str) -> str:
    return " ".join(s.strip().split())


def build_prompt(rec: Dict[str, Any], description_variant: int, gold: bool) -> str:
    yardage = rec.get("yardage") or "unknown"
    def fmt(v):
        return v if v is not None and v != "" else "n/a"
    key_haz = ", ".join(rec.get("key_hazards", []) or []) or "none"
    gold_note = " (GOLD EXAMPLE)" if gold else ""
    prompt_lines = [
        "TASK: description_synthesis",
        "Use the structured facts below to write a vivid, specific, NOT generic hole description for Bethpage Black Hole {hole}. Avoid boilerplate like 'combines strategic positioning' or 'multi-layered challenge'. Do not add meta commentary. End after the description.{gold}".format(hole=rec['hole'], gold=gold_note),
        "",
        "FACTS:",
        f"- Par: {rec['par']}",
        f"- Yardage: {yardage}",
        f"- Gradient: {rec['gradient']}",
        f"- Elevation Delta (yards): {rec['elevation_delta_yards']}",
        f"- Hole Length: {rec['hole_length']}",
        f"- Shape: {rec['hole_shape']}",
        f"- Primary Hazard: {fmt(rec.get('primary_hazard_type'))}",
        f"- Secondary Hazard: {fmt(rec.get('secondary_hazard_type'))}",
        f"- Key Hazards: {key_haz}",
        f"- Preferred Miss: {fmt(rec.get('preferred_miss'))}",
        f"- Punish Miss: {fmt(rec.get('punish_miss'))}",
        f"- Strategic Theme: {rec['strategic_theme']}",
        f"- Green Size: {rec['green_size']}",
        f"- Green Notes: {rec['green_complex_notes']}",
        f"- Tee Landing Width (yds): {fmt(rec.get('tee_landing_zone_width_yards'))}",
        "",
        "DESCRIPTION:",
        "",
    ]
    return "\n".join(prompt_lines)


def validate_record(raw: Dict[str, Any], seen_holes: set, warnings: List[str]):
    errors = []
    hole = raw.get("hole")
    if not isinstance(hole, int) or not (1 <= hole <= 18):
        errors.append(f"Invalid hole: {hole}")
    elif hole in seen_holes:
        errors.append(f"Duplicate hole: {hole}")
    for f in REQUIRED_STRING_FIELDS:
        if not raw.get(f):
            errors.append(f"Missing required field '{f}' for hole {hole}")
    for f in LIST_FIELDS:
        if f not in raw or not isinstance(raw[f], list):
            errors.append(f"Field '{f}' must be a list for hole {hole}")
    # Non-fatal quality checks
    descs = []
    for i in range(1, 6):
        key = f"new_description_{i}"
        if raw.get(key):
            descs.append(normalize_space(raw[key]))
    if len(descs) < 2:
        errors.append(f"Hole {hole}: need at least 2 descriptions, found {len(descs)}")
    # Pairwise similarity checks (first 5 only)
    for i in range(len(descs)):
        for j in range(i + 1, len(descs)):
            sim = similarity(descs[i].lower(), descs[j].lower())
            if sim > 0.85:
                warnings.append(f"Hole {hole}: desc {i+1} vs {j+1} similarity {sim:.2f} > 0.85")
    for idx, d in enumerate(descs, start=1):
        if d and len(d) < 40:
            warnings.append(f"Hole {hole} desc {idx} very short ({len(d)} chars)")
        if d and len(d) > 1200:
            warnings.append(f"Hole {hole} desc {idx} very long ({len(d)} chars)")
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data/bethpage_black/data_template.json")
    p.add_argument("--out", default="data/bethpage_black/descriptions_enriched.jsonl")
    p.add_argument("--stats", default="data/bethpage_black/descriptions_enriched_stats.json")
    args = p.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input file not found: {inp}", file=sys.stderr)
        sys.exit(1)

    records = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"JSON decode error: {e}", file=sys.stderr)
                sys.exit(1)
            records.append(obj)

    warnings: List[str] = []
    seen_holes = set()
    errors_all: List[str] = []
    out_examples = []

    for rec in records:
        errs = validate_record(rec, seen_holes, warnings)
        if errs:
            errors_all.extend(errs)
            continue
        seen_holes.add(rec["hole"])
        # Determine gold (optional)
        gold_idx = rec.get("gold_description")
        if gold_idx is not None:
            try:
                gold_idx = int(gold_idx)
                if not (1 <= gold_idx <= 5):
                    warnings.append(f"Hole {rec['hole']}: gold_description out of range -> ignoring")
                    gold_idx = None
            except Exception:
                warnings.append(f"Hole {rec['hole']}: gold_description not int -> ignoring")
                gold_idx = None
        # Collect up to 5 description fields
        for idx in range(1, 6):
            field = f"new_description_{idx}"
            if not rec.get(field):
                continue
            desc = normalize_space(rec[field])
            is_gold = gold_idx == idx
            prompt = build_prompt(rec, idx, is_gold)
            example = {
                "hole": rec["hole"],
                "par": rec["par"],
                "yardage": rec.get("yardage"),
                "prompt": prompt,
                "completion": desc,
                "task_type": "description_synthesis",
                "variation_type": f"enriched_{idx}",
                "use_for_training": True,
                "is_gold": is_gold,
                "split": "train",
                # Keep some structured metadata (might help later filtering / analysis)
                "meta": {
                    k: rec.get(k) for k in [
                        "gradient",
                        "hole_length",
                        "hole_shape",
                        "elevation_delta_yards",
                        "tee_landing_zone_width_yards",
                        "primary_hazard_type",
                        "secondary_hazard_type",
                        "preferred_miss",
                        "punish_miss",
                        "strategic_theme",
                        "green_size",
                        "green_complex_notes",
                        "key_hazards",
                    ]
                },
            }
            out_examples.append(example)

    if errors_all:
        print("Validation FAILED:", file=sys.stderr)
        for e in errors_all:
            print(" -", e, file=sys.stderr)
        sys.exit(2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in out_examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    gold_count = sum(1 for e in out_examples if e.get("is_gold"))
    stats = {
        "num_holes": len(seen_holes),
        "num_examples": len(out_examples),
        "gold_examples": gold_count,
        "warnings": warnings,
    }
    with Path(args.stats).open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print(f"Wrote {len(out_examples)} examples for {len(seen_holes)} holes -> {out_path}")
    if warnings:
        print("Warnings:")
        for w in warnings:
            print(" -", w)


if __name__ == "__main__":  # pragma: no cover
    main()
