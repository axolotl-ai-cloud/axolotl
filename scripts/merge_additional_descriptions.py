#!/usr/bin/env python
"""Merge base hole template (with new_description_1/2) and externally supplied
additional variants (new_description_3..5) into a 5-variant enriched file.

Usage:
  python scripts/merge_additional_descriptions.py \
      --base data/bethpage_black/data_template.json \
      --additional data/bethpage_black/additional_descriptions.jsonl \
      --out data/bethpage_black/data_template_5desc_merged.json

Validation rules:
  * Each additional line must supply hole + new_description_3/4/5 (non-empty)
  * Word count bounds (default 40..170) except we allow variant 5 to be as low as 25 for a concise summary style
  * Reject boilerplate openings (configurable list)
  * Detect near-duplicate (>0.80 Jaccard) within the 5 variants for a hole
  * Optional style_tags are carried through but not required.

Exit code non-zero on validation failure; prints a concise report.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Tuple

BOILERPLATE_PREFIXES = [
    "this par ",
    "at ",
    "hole ",
]

MIN_WORDS_DEFAULT = 40
MIN_WORDS_VARIANT5 = 25
MAX_WORDS = 170
NEAR_DUP_JACCARD = 0.80

def tokenize(s: str) -> List[str]:
    return [t.lower() for t in s.strip().replace('\n', ' ').split() if t]

def jaccard(a: str, b: str) -> float:
    sa, sb = set(tokenize(a)), set(tokenize(b))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)

def has_boilerplate(s: str) -> bool:
    ls = s.lstrip().lower()
    return any(ls.startswith(p) for p in BOILERPLATE_PREFIXES)

def load_base(path: Path) -> Dict[int, Dict[str, Any]]:
    holes: Dict[int, Dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            hole = rec['hole']
            holes[hole] = rec
    return holes

def load_additional(path: Path) -> Dict[int, Dict[str, Any]]:
    """Allow multiple partial records per hole; merge keys (last wins for conflicts)."""
    out: Dict[int, Dict[str, Any]] = {}
    with path.open('r', encoding='utf-8') as f:
        for lineno, line in enumerate(f, 1):
            raw = line.rstrip('\n')
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"JSON parse error {path}:{lineno}: {e}\nLine: {raw}")
            hole = rec.get('hole')
            if hole is None:
                raise SystemExit(f"Missing hole field in additional file line {lineno}")
            if hole not in out:
                out[hole] = {'hole': hole}
            for k, v in rec.items():
                if k == 'hole':
                    continue
                out[hole][k] = v
    return out

def validate_and_merge(base: Dict[int, Dict[str, Any]], addl: Dict[int, Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    merged: List[Dict[str, Any]] = []
    errors: List[str] = []
    for hole, base_rec in sorted(base.items()):
        add = addl.get(hole, {})
        # Require all three be present & non-empty
        missing_any = False
        for key in ('new_description_3','new_description_4','new_description_5'):
            if not add.get(key):
                errors.append(f"Hole {hole}: {key} empty or missing")
                missing_any = True
        if missing_any:
            # Still merge what we have so user can inspect partial output? We skip to avoid downstream noise.
            continue
        # Word count checks
        for key in ('new_description_3','new_description_4','new_description_5'):
            txt = add.get(key, '')
            wc = len(tokenize(txt))
            min_words = MIN_WORDS_VARIANT5 if key == 'new_description_5' else MIN_WORDS_DEFAULT
            if wc < min_words:
                errors.append(f"Hole {hole}: {key} too short ({wc} < {min_words})")
            if wc > MAX_WORDS:
                errors.append(f"Hole {hole}: {key} too long ({wc} > {MAX_WORDS})")
            if txt and has_boilerplate(txt):
                errors.append(f"Hole {hole}: {key} starts with boilerplate prefix")
        # Duplicate / similarity checks across all five
        variants = [
            base_rec.get('new_description_1',''),
            base_rec.get('new_description_2',''),
            add.get('new_description_3',''),
            add.get('new_description_4',''),
            add.get('new_description_5',''),
        ]
        labels = ['1','2','3','4','5']
        for i in range(len(variants)):
            for j in range(i+1, len(variants)):
                if not variants[i] or not variants[j]:
                    continue
                sim = jaccard(variants[i], variants[j])
                if sim >= NEAR_DUP_JACCARD:
                    errors.append(f"Hole {hole}: variants {labels[i]} & {labels[j]} Jaccard {sim:.2f} >= {NEAR_DUP_JACCARD}")
        # Merge record
        merged_rec = dict(base_rec)
        for k in ('new_description_3','new_description_4','new_description_5','style_tags'):
            if k in add:
                merged_rec[k] = add[k]
        merged.append(merged_rec)
    return merged, errors

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='Path to base data_template.json')
    ap.add_argument('--additional', required=True, help='Path to filled additional descriptions jsonl')
    ap.add_argument('--out', required=True, help='Output path for merged 5-variant file (json or jsonl)')
    ap.add_argument('--jsonl', action='store_true', help='Write JSONL (one line per hole) instead of single JSON array')
    args = ap.parse_args()

    base_path = Path(args.base)
    add_path = Path(args.additional)
    out_path = Path(args.out)

    base = load_base(base_path)
    addl = load_additional(add_path)
    merged, errors = validate_and_merge(base, addl)
    if errors:
        print("VALIDATION ERRORS:")
        for e in errors:
            print(" -", e)
        print(f"Aborting. {len(errors)} issues found.", file=sys.stderr)
        sys.exit(1)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if args.jsonl:
        with out_path.open('w', encoding='utf-8') as f:
            for rec in merged:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    else:
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(merged)} merged hole records to {out_path}")

if __name__ == '__main__':
    main()
