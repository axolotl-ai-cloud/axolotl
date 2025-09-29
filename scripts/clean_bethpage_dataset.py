#!/usr/bin/env python3
"""
Clean and validate the Bethpage Black multitask dataset.

Actions:
- Unicode NFC normalization, trim, and whitespace squashing on prompt/completion
- Deduplicate by (task_type, prompt, completion)
- Basic toxicity filter (lightweight bad-word list)
- Strategy label fixes for strategy_selection:
  * enforce expected_cutoff_yards in available_cutoffs when provided
  * compute cutoff from prompt drive and strategies (highest <= drive else min)
  * standardize completion to "Strategy: {N} yards"
- Drop clearly broken rows (missing prompt/completion, absurd numbers)
- Write a compact JSON report with stats

Usage:
  python scripts/clean_bethpage_dataset.py \
    --input data/bethpage_black/train_multitask_case_clean.jsonl \
    --output data/bethpage_black/train_multitask_case_validated.jsonl \
    --report outputs/bethpage-lora/clean_report.json
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata as ud
from typing import Any, Dict, List, Optional, Tuple


DRIVE_PATTERNS = [
    r"average drive:\s*(\d{2,4})\s*yards",
    r"drives\s*(\d{2,4})\s*yards",
    r"with a\s*(\d{2,4})-yard drive",
    r"drive is exactly\s*(\d{2,4})\s*yards",
    r"average drive of\s*(\d{2,4})\s*yards",
    r"drive is\s*(\d{2,4})\s*yards",
]

BAD_WORDS = {
    # keep small and obvious; not exhaustive
    "fuck",
    "shit",
    "asshole",
}


def nfc_clean(s: Optional[str]) -> str:
    if not s:
        return ""
    s = ud.normalize("NFC", s)
    # collapse whitespace; preserve newlines minimally (convert multiple to one)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def has_toxicity(text: str) -> bool:
    t = text.lower()
    return any(bw in t for bw in BAD_WORDS)


def extract_drive(prompt: str) -> Optional[int]:
    for pat in DRIVE_PATTERNS:
        m = re.search(pat, prompt, flags=re.IGNORECASE)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def choose_cutoff(strategies: List[Dict[str, Any]], drive: Optional[int]) -> Optional[int]:
    if not strategies:
        return None
    cutoffs = sorted(
        [s.get("cutoff_distance") for s in strategies if isinstance(s.get("cutoff_distance"), int)]
    )
    if not cutoffs:
        return None
    if drive is None:
        return cutoffs[-1]
    eligible = [c for c in cutoffs if c <= drive]
    return max(eligible) if eligible else cutoffs[0]


STRAT_PATTERN = re.compile(r"Strategy:\s*(\d{2,4})\s*yards", re.IGNORECASE)


def extract_completion_cutoff(completion: str) -> Optional[int]:
    m = STRAT_PATTERN.search(completion or "")
    return int(m.group(1)) if m else None


def standardize_strategy_completion(cutoff: int) -> str:
    return f"Strategy: {cutoff} yards"


def clean_record(rec: Dict[str, Any], report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Normalize text
    rec = dict(rec)
    rec["prompt"] = nfc_clean(rec.get("prompt"))
    rec["completion"] = nfc_clean(rec.get("completion"))

    if not rec["prompt"] or not rec["completion"]:
        report["dropped_missing"] += 1
        return None

    # quick absurd length checks
    if len(rec["prompt"]) < 10 or len(rec["completion"]) < 5:
        report["dropped_too_short"] += 1
        return None

    # toxicity filter (only drop training examples; leave eval to separate file typically)
    if has_toxicity(rec["completion"]):
        report["dropped_toxic"] += 1
        return None

    task = rec.get("task_type")
    if task == "strategy_selection":
        drive = extract_drive(rec["prompt"])
        strategies = rec.get("tee_shot_strategies") or []
        avail_cutoffs = [s.get("cutoff_distance") for s in strategies if isinstance(s.get("cutoff_distance"), int)]
        avail_set = set(avail_cutoffs)

        expected = rec.get("expected_cutoff_yards")
        predicted_from_comp = extract_completion_cutoff(rec.get("completion", ""))
        rule_choice = choose_cutoff(strategies, drive)

        # Correct expected_cutoff if missing or inconsistent
        target = expected if isinstance(expected, int) else None
        if target is None:
            target = rule_choice
            report["fixed_missing_expected"] += 1
        if target is not None and avail_set and target not in avail_set:
            # snap to nearest valid by our rule
            target = rule_choice
            report["fixed_expected_not_in_available"] += 1
        if drive is not None and target is not None and target > drive and avail_cutoffs:
            # enforce <= drive if possible
            corrected = max([c for c in avail_cutoffs if c <= drive] or [min(avail_cutoffs)])
            if corrected != target:
                target = corrected
                report["fixed_expected_gt_drive"] += 1

        # If still None, fallback to 300
        if target is None:
            target = 300
            report["fallback_default_cutoff"] += 1

        # Standardize completion
        std_comp = standardize_strategy_completion(target)
        if rec["completion"] != std_comp:
            rec["completion"] = std_comp
            report["standardized_completion"] += 1
        rec["expected_cutoff_yards"] = int(target)

    else:
        # description_synthesis or others: lightly validate numbers
        # Drop completions with ridiculous yardages like 4889
        bad_num = re.search(r"\b(\d{4,})\b", rec["completion"]) is not None
        if bad_num:
            report["dropped_absurd_numbers"] += 1
            return None

    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input JSONL path")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    ap.add_argument("--report", required=False, default=None, help="Optional JSON report path")
    args = ap.parse_args()

    report = {
        "read": 0,
        "written": 0,
        "dropped_missing": 0,
        "dropped_too_short": 0,
        "dropped_toxic": 0,
        "dropped_absurd_numbers": 0,
        "fixed_missing_expected": 0,
        "fixed_expected_not_in_available": 0,
        "fixed_expected_gt_drive": 0,
        "fallback_default_cutoff": 0,
        "standardized_completion": 0,
        "deduped": 0,
    }

    seen: set[Tuple[str, str, str]] = set()
    written_records: List[Dict[str, Any]] = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            report["read"] += 1
            try:
                rec = json.loads(line)
            except Exception:
                report["dropped_missing"] += 1
                continue
            cleaned = clean_record(rec, report)
            if cleaned is None:
                continue
            key = (
                str(cleaned.get("task_type", "")),
                cleaned.get("prompt", ""),
                cleaned.get("completion", ""),
            )
            if key in seen:
                report["deduped"] += 1
                continue
            seen.add(key)
            written_records.append(cleaned)

    with open(args.output, "w", encoding="utf-8") as out:
        for r in written_records:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")
            report["written"] += 1

    if args.report:
        with open(args.report, "w", encoding="utf-8") as rf:
            json.dump(report, rf, indent=2, ensure_ascii=False)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
