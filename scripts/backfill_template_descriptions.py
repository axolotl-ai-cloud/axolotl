"""Backfill placeholder description variants (new_description_3..5) using legacy description_synthesis prompts.

Sources: one or more legacy JSONL files (e.g., data/bethpage_black/train_multitask_case_fixed.jsonl, train.jsonl)
Strategy:
  * Collect completions per hole for task_type == description_synthesis.
  * Rank by (penalize generic phrases, prefer longer up to 220 words, unique token count).
  * Fill any template record whose new_description_3..5 still contain placeholder 'TODO:' text.
  * Skip duplicates (similarity > 0.85 vs existing variants 1..5).
  * Do not overwrite existing custom user text.

Usage:
  python scripts/backfill_template_descriptions.py \
      --template data/bethpage_black/data_template_5desc.json \
      --legacy data/bethpage_black/train_multitask_case_fixed.jsonl data/bethpage_black/train.jsonl \
      --out data/bethpage_black/data_template_5desc_filled.json
"""
from __future__ import annotations
import argparse, json, sys, re
from pathlib import Path
from typing import Dict, List

try:
    from rapidfuzz import fuzz
    def sim(a,b): return fuzz.token_sort_ratio(a,b)/100.0
except Exception:
    import difflib
    def sim(a,b): return difflib.SequenceMatcher(None,a,b).ratio()

GENERIC_PATTERNS = [
    re.compile(p, re.I) for p in [
        r"combines strategic positioning",
        r"multi-layered challenge",
        r"course management",
        r"informed decisions",
        r"requires careful evaluation",
    ]
]

def score(desc: str) -> float:
    tokens = desc.split()
    length = len(tokens)
    unique = len(set(t.lower().strip(',.!?') for t in tokens))
    generic_hits = sum(1 for pat in GENERIC_PATTERNS if pat.search(desc))
    # heuristic: base on unique density, penalize generic, mild length preference up to 220 words
    length_factor = min(length, 220) / 220
    return unique / max(length,1) * 0.6 + length_factor * 0.4 - generic_hits * 0.2

PLACEHOLDER_PREFIX = "TODO: add variant"

def load_legacy(files: List[Path]) -> Dict[int, List[str]]:
    per_hole: Dict[int, List[str]] = {}
    for fp in files:
        if not fp.exists():
            continue
        with fp.open('r', encoding='utf-8') as f:
            for line in f:
                line=line.strip()
                if not line: continue
                try:
                    obj=json.loads(line)
                except Exception:
                    continue
                if obj.get('task_type') != 'description_synthesis':
                    continue
                h = obj.get('hole')
                if not isinstance(h,int):
                    continue
                completion = obj.get('completion') or ""
                if not completion:
                    continue
                per_hole.setdefault(h,[]).append(completion)
    # dedupe exact
    for h, lst in per_hole.items():
        uniq=[]
        seen=set()
        for d in lst:
            d_clean=' '.join(d.split())
            if d_clean in seen: continue
            seen.add(d_clean)
            uniq.append(d_clean)
        per_hole[h]=uniq
    return per_hole

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--template', required=True)
    ap.add_argument('--legacy', nargs='+', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--max_fill', type=int, default=3, help='How many new slots to attempt to fill (max 3).')
    args=ap.parse_args()

    template_path=Path(args.template)
    if not template_path.exists():
        print('Template not found', file=sys.stderr)
        sys.exit(1)
    legacy_paths=[Path(p) for p in args.legacy]
    legacy=load_legacy(legacy_paths)

    updated=[]
    filled_count=0
    with template_path.open('r', encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            if not line: continue
            obj=json.loads(line)
            hole=obj.get('hole')
            if isinstance(hole,int) and hole in legacy:
                # Collect existing descriptions
                existing=[obj.get(f'new_description_{i}') for i in range(1,6) if obj.get(f'new_description_{i}')]
                # Candidate legacy pool scored
                candidates=sorted(legacy[hole], key=score, reverse=True)
                for cand in candidates:
                    if all(sim(cand, e) <= 0.85 for e in existing):
                        # find next placeholder slot
                        for slot in range(3,6):
                            key=f'new_description_{slot}'
                            if obj.get(key, '').startswith(PLACEHOLDER_PREFIX):
                                obj[key]=cand
                                existing.append(cand)
                                filled_count+=1
                                break
                    if filled_count >= args.max_fill * 18:  # rough cap
                        break
            updated.append(obj)

    out_path=Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8') as f:
        for o in updated:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    print(f'Wrote {len(updated)} holes -> {out_path}; filled slots: {filled_count}')

if __name__=='__main__':
    main()
