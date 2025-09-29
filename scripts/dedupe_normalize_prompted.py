#!/usr/bin/env python
"""Deduplicate and normalize a prompted multitask dataset.

Operations:
1. Unicode / mojibake normalization (common cp1252 -> utf-8 artifacts)
2. Exact duplicate removal based on (task, assistant message content)
   - Keeps the first occurrence, merges sample_weight (summing) across duplicates
3. Writes a new JSONL with updated sample_weight.
4. Emits a summary report to stderr.

Normalization steps:
- Replace mojibake sequences (â€™ -> ’, â€œ -> “, â€ -> ”, â€“ -> – etc.)
- Collapse stray 'Ã', 'Â' preceding symbols
- Strip trailing spaces
- Optional: dedupe repeated whitespace

Usage:
  python scripts/dedupe_normalize_prompted.py INPUT.jsonl -o OUTPUT.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
import re
from pathlib import Path
from collections import defaultdict

MOJIBAKE_REPLACEMENTS = [
    ("â€™", "’"),
    ("â€œ", "“"),
    ("â€", "”"),
    ("â€“", "–"),
    ("â€”", "—"),
    ("â€¦", "…"),
    ("Â", ""),
    ("Ã", ""),
]
WS_RE = re.compile(r"\s+")

def normalize_text(t: str) -> str:
    if not t:
        return t
    for src, dst in MOJIBAKE_REPLACEMENTS:
        t = t.replace(src, dst)
    # collapse multiple spaces
    t = WS_RE.sub(" ", t)
    return t.strip()

def process(in_path: Path, out_path: Path):
    seen = {}
    merged = []
    duplicates = 0
    weight_merged = 0.0

    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # normalize assistant content
            for m in rec.get("messages", []):
                if m.get("role") == "assistant":
                    m["content"] = normalize_text(m.get("content", ""))
            # key
            assistant_content = next((m.get("content") for m in rec.get("messages", []) if m.get("role") == "assistant"), None)
            key = (rec.get("task"), assistant_content)
            if key in seen:
                # merge weight
                existing = seen[key]
                existing_weight = float(existing.get("sample_weight", 1.0))
                new_weight = float(rec.get("sample_weight", 1.0))
                existing["sample_weight"] = existing_weight + new_weight
                duplicates += 1
                weight_merged += new_weight
            else:
                seen[key] = rec
                merged.append(rec)

    with out_path.open("w", encoding="utf-8") as out:
        for rec in merged:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # report
    original = duplicates + len(merged)
    distinct = len(merged)
    sys.stderr.write(
        f"Original examples: {original}\nDistinct after dedupe: {distinct}\nDuplicates merged: {duplicates}\nTotal merged weight added: {weight_merged}\n"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=Path)
    ap.add_argument("-o", "--out", type=Path, required=True)
    args = ap.parse_args()
    process(args.input, args.out)

if __name__ == "__main__":
    main()
