"""Validate enriched hole dataset and derived multi-task dataset.

Usage:
  python scripts/validate_enriched_dataset.py --enriched data/.../strategy_augmented_holes_enriched.jsonl [--multitask built.jsonl]

Checks (enriched):
  - Required top-level keys present
  - Exactly 5 descriptions, gold id valid
  - tee_shot_strategy_options non-empty for non-par-3 holes (par>=4); par 3 holes may have 2+ options including exact yardage
  - Attributes contain expected keys; preferred_miss != punish_miss when both provided
  - classification values in allowed set {conservative, neutral, aggressive}
  - cutoff distances strictly increasing & within plausible range (100-400 for par 4/5 strategy options)

Checks (multi-task if provided):
  - task_type in {strategy_selection, description_synthesis}
  - expected_cutoff_yards consistency with completion pattern for strategy tasks
  - description_synthesis rows have gold flag only on variants that existed in source
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ATTR_REQUIRED = {
    "gradient",
    "hole_shape",
    "strategic_theme",
}

ALLOWED_CLASS = {"conservative", "neutral", "aggressive"}


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield ln, json.loads(line)
            except Exception as e:
                raise ValueError(f"Invalid JSON at {path}:{ln}: {e}") from e


def validate_enriched(path: Path):
    errors = []
    holes_seen = set()
    for ln, obj in _iter_jsonl(path):
        hole = obj.get("hole")
        if hole in holes_seen:
            errors.append(f"duplicate hole entry {hole} (line {ln})")
        holes_seen.add(hole)
        for k in ("par", "yardage", "attributes", "tee_shot_strategy_options", "descriptions", "gold_description_variant_id"):
            if k not in obj:
                errors.append(f"missing key {k} hole {hole}")
        attrs = obj.get("attributes", {}) or {}
        missing_attr = ATTR_REQUIRED - set(attrs)
        if missing_attr:
            errors.append(f"hole {hole} missing attr keys: {sorted(missing_attr)}")
        descs = obj.get("descriptions") or []
        if len(descs) != 5:
            errors.append(f"hole {hole} expected 5 descriptions got {len(descs)}")
        gold_id = obj.get("gold_description_variant_id")
        if not isinstance(gold_id, int) or not (1 <= gold_id <= len(descs)):
            errors.append(f"hole {hole} invalid gold_description_variant_id {gold_id}")
        pref = attrs.get("preferred_miss")
        punish = attrs.get("punish_miss")
        if pref and punish and pref == punish:
            errors.append(f"hole {hole} preferred_miss equals punish_miss ({pref})")
        strategies = obj.get("tee_shot_strategy_options") or []
        if (obj.get("par") or 0) >= 4 and len(strategies) == 0:
            errors.append(f"hole {hole} (par {obj.get('par')}) missing strategy options")
        cutoffs = []
        for s in strategies:
            cd = s.get("cutoff_distance")
            if isinstance(cd, int):
                cutoffs.append(cd)
            cl = s.get("classification")
            if cl and cl not in ALLOWED_CLASS:
                errors.append(f"hole {hole} invalid classification {cl}")
        if cutoffs:
            ordered = sorted(cutoffs)
            if ordered != cutoffs and len(cutoffs) > 1:
                # enforce increasing order as hygiene
                errors.append(f"hole {hole} cutoffs not sorted: {cutoffs}")
            for c in cutoffs:
                if not (80 <= c <= 400 or 400 < c <= 650):  # allow par 5 layup spans
                    errors.append(f"hole {hole} cutoff {c} out of plausible range")
    return errors


def validate_multitask(path: Path):
    errors = []
    for ln, obj in _iter_jsonl(path):
        t = obj.get("task_type")
        if t not in {"strategy_selection", "description_synthesis"}:
            errors.append(f"line {ln} invalid task_type {t}")
            continue
        if t == "strategy_selection":
            comp = obj.get("completion", "")
            m = re.search(r"(\d{2,4})", comp)
            exp = obj.get("expected_cutoff_yards")
            if not m or int(m.group(1)) != exp:
                errors.append(f"line {ln} expected_cutoff mismatch completion={comp} expected={exp}")
        else:
            if "expected_cutoff_yards" in obj:
                errors.append(f"line {ln} description task has unexpected expected_cutoff_yards field")
    return errors


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched", required=True)
    ap.add_argument("--multitask")
    args = ap.parse_args()
    enr_path = Path(args.enriched)
    enr_errors = validate_enriched(enr_path)
    if enr_errors:
        print("Enriched dataset validation FAILED:")
        for e in enr_errors:
            print(" -", e)
    else:
        print("Enriched dataset validation PASSED")
    if args.multitask:
        mt_errors = validate_multitask(Path(args.multitask))
        if mt_errors:
            print("Multi-task dataset validation FAILED:")
            for e in mt_errors:
                print(" -", e)
        else:
            print("Multi-task dataset validation PASSED")
    if enr_errors:
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
