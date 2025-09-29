#!/usr/bin/env python
"""Evaluate prompted multitask dataset quality.

Metrics:
- task_counts: number of examples per task
- sample_weight_sum: aggregate weights per task
- avg_message_chars: average total character length of all messages per record by task
- hazard_coverage: fraction of selection/description records whose user prompt includes all key hazards listed in meta.original.input.attributes.key_hazards (if present)
- ngram4_unique_ratio (by task): unique 4-grams / total 4-grams over assistant outputs
- template_phrase_rate: proportion of assistant outputs starting with one of common boilerplate phrases
- encoding_artifact_rate: proportion of assistant outputs containing mojibake artifacts (e.g., , , â€™)
- duplicate_assistant_hashes: count of exact duplicate assistant outputs per task (hash frequency >1)
- style_rewrite_word_budget_violations: fraction of style_rewrite outputs exceeding 45 words

Outputs printed as pretty JSON; optionally save via --out path.
"""
from __future__ import annotations
import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any, List

BOILERPLATE_PREFIXES = [
    "The only par", "Playing as the hardest", "Par 4", "Par 3", "Par 5", "This long par", "Most players", "Par 4 389y"
]
MOJIBAKE_PATTERNS = ["â€™", "Ã", "Â", "â€œ", "â€"]
WORD_BUDGET_STYLE = 45

TOKEN_SPLIT_RE = re.compile(r"\W+")

def tokenize(text: str) -> List[str]:
    return [t.lower() for t in TOKEN_SPLIT_RE.split(text) if t]

def fourgrams(tokens: List[str]):
    for i in range(len(tokens) - 3):
        yield tuple(tokens[i:i+4])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path, help="Path to prompted dataset JSONL")
    ap.add_argument("--out", type=Path, help="Optional path to write metrics JSON")
    args = ap.parse_args()

    task_counts = Counter()
    task_weight = Counter()
    total_chars = Counter()
    hazard_full_coverage = Counter()
    hazard_applicable = Counter()
    ngram4_counts = defaultdict(Counter)
    duplicate_tracker = defaultdict(Counter)
    template_hits = Counter()
    encoding_hits = Counter()
    style_rewrite_budget_over = 0
    style_rewrite_total = 0

    with args.data.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            task = rec.get("task")
            task_counts[task] += 1
            w = float(rec.get("sample_weight", 1.0))
            task_weight[task] += w

            # message char length
            chars = sum(len(m.get("content", "")) for m in rec.get("messages", []))
            total_chars[task] += chars

            messages = rec.get("messages", [])
            assistant_msg = None
            for m in messages:
                if m.get("role") == "assistant":
                    assistant_msg = m.get("content", "")
            if assistant_msg is None:
                continue

            # template phrase
            if any(assistant_msg.startswith(p) for p in BOILERPLATE_PREFIXES):
                template_hits[task] += 1

            # encoding artifacts
            if any(p in assistant_msg for p in MOJIBAKE_PATTERNS):
                encoding_hits[task] += 1

            # duplicate hash
            duplicate_tracker[task][assistant_msg] += 1

            # style rewrite budget
            if task == "style_rewrite":
                style_rewrite_total += 1
                words = tokenize(assistant_msg)
                if len(words) > WORD_BUDGET_STYLE:
                    style_rewrite_budget_over += 1

            # hazard coverage (only if we have hazards list in meta)
            try:
                meta = rec.get("meta", {})
                orig = meta.get("original", {})
                inp = orig.get("input", {})
                attrs = inp.get("attributes", {})
                hazards = attrs.get("key_hazards")
                if isinstance(hazards, list) and hazards:
                    user_msg = next((m.get("content", "") for m in messages if m.get("role") == "user"), "")
                    hazard_applicable[task] += 1
                    # all hazard strings appear (case-insensitive)
                    if all(h.lower() in user_msg.lower() for h in hazards):
                        hazard_full_coverage[task] += 1
            except Exception:
                pass

            # ngram stats on assistant
            toks = tokenize(assistant_msg)
            for fg in fourgrams(toks):
                ngram4_counts[task][fg] += 1

    metrics: Dict[str, Any] = {}
    metrics["task_counts"] = dict(task_counts)
    metrics["task_weight_sums"] = {k: round(v, 3) for k, v in task_weight.items()}
    metrics["avg_message_chars"] = {k: round(total_chars[k] / task_counts[k], 1) for k in task_counts}
    metrics["hazard_coverage"] = {k: {
        "applicable": hazard_applicable[k],
        "full_coverage": hazard_full_coverage[k],
        "rate": round(hazard_full_coverage[k]/hazard_applicable[k], 3) if hazard_applicable[k] else None
    } for k in task_counts}

    # ngram uniqueness
    ngram_uniqueness = {}
    for task, ctr in ngram4_counts.items():
        total = sum(ctr.values())
        unique = sum(1 for g,c in ctr.items() if c == 1)
        ngram_uniqueness[task] = {
            "total_4grams": total,
            "unique_4grams": unique,
            "unique_ratio": round(unique/total, 4) if total else None
        }
    metrics["ngram4_uniqueness"] = ngram_uniqueness

    metrics["template_phrase_rate"] = {k: round(template_hits[k]/task_counts[k], 3) for k in task_counts}
    metrics["encoding_artifact_rate"] = {k: round(encoding_hits[k]/task_counts[k], 3) for k in task_counts}

    duplicate_stats = {}
    for task, ctr in duplicate_tracker.items():
        dup_groups = {txt: c for txt, c in ctr.items() if c > 1}
        duplicate_stats[task] = {
            "duplicate_groups": len(dup_groups),
            "total_duplicate_examples": sum(c for c in dup_groups.values()),
            "max_dup": max(dup_groups.values()) if dup_groups else 1
        }
    metrics["duplicate_assistant_outputs"] = duplicate_stats

    if style_rewrite_total:
        metrics["style_rewrite_word_budget"] = {
            "total": style_rewrite_total,
            "over_budget": style_rewrite_budget_over,
            "violation_rate": round(style_rewrite_budget_over / style_rewrite_total, 3)
        }

    out_json = json.dumps(metrics, indent=2, sort_keys=True)
    print(out_json)
    if args.out:
        args.out.write_text(out_json, encoding="utf-8")

if __name__ == "__main__":
    main()
