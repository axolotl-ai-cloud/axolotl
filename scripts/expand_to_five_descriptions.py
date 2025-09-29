"""Expand existing 2-description template to 5-description template with placeholders.

Reads data/bethpage_black/data_template.json and rewrites (in-place or to --out) adding
new_description_3..5 if missing, initialized with "TODO: add variant" text including hole number.
Does not overwrite existing fields. Adds gold_description if absent (defaults to 1).

Usage:
  python scripts/expand_to_five_descriptions.py --input data/bethpage_black/data_template.json --out data/bethpage_black/data_template_5desc.json
  (Review & then optionally mv over original.)
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

PLACEHOLDER = "TODO: add variant"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/bethpage_black/data_template.json")
    ap.add_argument("--out", default="data/bethpage_black/data_template_5desc.json")
    ap.add_argument("--inplace", action="store_true", help="Write back to input file instead of --out")
    args = ap.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(1)

    out_path = inp if args.inplace else Path(args.out)

    updated = []
    with inp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            # Ensure descriptions 1..5
            for i in range(1, 6):
                key = f"new_description_{i}"
                if not obj.get(key):
                    obj[key] = f"{PLACEHOLDER} {obj.get('hole')} #{i}"
            if not obj.get("gold_description"):
                obj["gold_description"] = 1
            updated.append(obj)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for o in updated:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print(f"Wrote {len(updated)} holes -> {out_path}")
    missing_placeholders = sum(
        1 for o in updated for i in range(3,6) if PLACEHOLDER in o[f'new_description_{i}']
    )
    print(f"Added placeholder descriptions: {missing_placeholders}")

if __name__ == "__main__":
    main()
