#!/usr/bin/env python3
"""
Augment the multitask dataset by:
- Adding concise description variants (60–100 words) distilled from existing descriptions/insights
- Adding robustness variants for strategy prompts with typo/unit noise

Outputs a single merged JSONL preserving original records and adding synthetics
with use_for_training=True.
"""
import json
import os
import random
import re
from copy import deepcopy
from argparse import ArgumentParser

random.seed(42)

GUIDANCE_CONCISE = (
    "Write a single concise paragraph (60–100 words). Focus on the most relevant hazards, lie, and approach angle. "
    "Avoid meta language and headings. Use specific distances only when informative."
)

# Lightweight normalizer to generate noisy prompt variants
TYPO_MAP = {
    "strategy": ["stratergy", "strategry", "stratery"],
    "yards": ["yd", "yds", "yrds"],
    "maximum": ["maximun", "max"],
}

def _inject_typos(text: str) -> str:
    t = text
    for canonical, variants in TYPO_MAP.items():
        if re.search(rf"\b{canonical}\b", t, re.IGNORECASE) and random.random() < 0.7:
            t = re.sub(
                rf"\b{canonical}\b",
                random.choice(variants),
                t,
                flags=re.IGNORECASE,
            )
    return t


def _make_concise_desc_example(ex):
    hole = ex.get("hole")
    par = ex.get("par")
    yd = ex.get("yardage")
    # Compose a synthesis prompt using available fields
    base = (
        f"TASK: description_synthesis\n"
        f"Distill to a concise description for Hole {hole} at Bethpage Black (Par {par}, {yd} yards).\n\n"
    )
    # Prefer insights if present
    insights = ex.get("insights") or []
    desc = ex.get("description") or ex.get("completion") or ""
    bullets = "".join([f"- {b}\n" for b in insights[:6]])
    prompt = base + bullets
    if not bullets:
        prompt += desc[:400]
    prompt += "\n\nGuidance: " + GUIDANCE_CONCISE
    return {
        "task_type": "description_synthesis",
        "hole": hole,
        "par": par,
        "yardage": yd,
        "prompt": prompt,
        # Leave completion empty so the trainer builds a standard target from original completion
        "completion": ex.get("completion", ""),
        "use_for_training": True,
    }


def _make_noisy_strategy_example(ex):
    # Build a variant of the strategy prompt with typos/units
    hole = ex.get("hole")
    par = ex.get("par")
    yd = ex.get("yardage")
    # Try to infer a plausible drive from the prompt/completion
    prompt = ex.get("prompt", "")
    m = re.search(r"(\d{2,3})\s*yard", prompt, re.IGNORECASE)
    drive = int(m.group(1)) if m else None
    if drive is None:
        # fallback around available cutoffs
        cutoffs = sorted([s.get("cutoff_distance") for s in (ex.get("tee_shot_strategies") or []) if isinstance(s.get("cutoff_distance"), int)])
        drive = cutoffs[-1] if cutoffs else 280
    base = (
        f"TASK: strategy_selection\n"
        f"Hole {hole}, Bethpage Black: Par {par}, {yd} yards. Golfer drives {drive} yards. Best stratergy?\n"
        f"Respond with: Strategy: <number> yrds"
    )
    noisy = _inject_typos(base)
    out = deepcopy(ex)
    out.update({
        "task_type": "strategy_selection",
        "prompt": noisy,
        "use_for_training": True,
    })
    return out


def main():
    ap = ArgumentParser()
    ap.add_argument("--infile", default="data/bethpage_black/train_multitask_case_fixed.jsonl")
    ap.add_argument("--outfile", default="data/bethpage_black/train_multitask_augmented.jsonl")
    ap.add_argument("--desc_frac", type=float, default=0.4, help="Fraction of examples to add concise desc variants for")
    ap.add_argument("--strat_frac", type=float, default=0.4, help="Fraction of strategy examples to add noisy prompt variants for")
    args = ap.parse_args()

    records = []
    with open(args.infile, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            records.append(json.loads(line))

    out = []
    strategy_pool = [r for r in records if r.get("task_type") == "strategy_selection"]
    desc_pool = [r for r in records if r.get("task_type") == "description_synthesis"]

    out.extend(records)

    # Add concise description variants
    n_desc = int(len(desc_pool) * args.desc_frac)
    random.shuffle(desc_pool)
    for r in desc_pool[:n_desc]:
        out.append(_make_concise_desc_example(r))

    # Add noisy strategy prompt variants
    n_strat = int(len(strategy_pool) * args.strat_frac)
    random.shuffle(strategy_pool)
    for r in strategy_pool[:n_strat]:
        out.append(_make_noisy_strategy_example(r))

    with open(args.outfile, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(out)} records to {args.outfile} (from {len(records)})")

if __name__ == "__main__":
    main()
