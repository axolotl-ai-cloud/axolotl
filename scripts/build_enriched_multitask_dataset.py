"""Build multi-task training JSONL from enriched hole dataset.

Input (enriched file schema per strategy_augmented_holes_enriched.jsonl): one JSON object per hole with keys:
  hole, par, yardage, attributes (dict with structured fields),
  tee_shot_strategy_options (list of {cutoff_distance, remaining_distance, ... classification}),
  descriptions (list[str] length 5), gold_description_variant_id (1-5),
  derived_notes (dict), description_scoring (dict of variant metrics)

Output JSONL examples (compatible with existing training script):
  {
    "task_type": "strategy_selection",
    "prompt": <hole facts + player scenario + explicit list of strategy cutoffs>,
    "completion": "Strategy: <CUT> yards",
    "expected_cutoff_yards": <CUT>,
    "hole": <int>,
    "tee_shot_strategies": [ {cutoff_distance, classification}, ... ],
    "par": <int>,
    "yardage": <int>,
    "use_for_training": true
  }
  {
    "task_type": "description_synthesis",
    "prompt": <hole facts + directive>,
    "completion": <one description variant text>,
    "hole": <int>,
    "par": <int>,
    "yardage": <int>,
    "gold": bool (true iff variant id == gold_description_variant_id),
    "variant_id": <int 1-5>,
    "use_for_training": true
  }

The strategy examples: one per strategy option plus one extra example framing a mid‑range drive (median of cutoffs) to encourage correct selection when multiple cutoffs available.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def _load_enriched(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _facts_block(obj: dict) -> str:
    attr = obj.get("attributes", {})
    notes = obj.get("derived_notes", {})
    parts = [
        f"Hole {obj.get('hole')} | Par {obj.get('par')} | Yardage {obj.get('yardage')}",
    ]
    def add(label, key):
        v = attr.get(key)
        if v is not None and v != "":
            parts.append(f"{label}: {v}")
    add("Gradient", "gradient")
    add("Shape", "hole_shape")
    add("PrimaryHazard", "primary_hazard_type")
    add("SecondaryHazard", "secondary_hazard_type")
    add("PreferredMiss", "preferred_miss")
    add("PunishMiss", "punish_miss")
    add("Theme", "strategic_theme")
    # key hazards list
    kh = attr.get("key_hazards") or []
    if kh:
        parts.append("KeyHazards: " + "; ".join(kh))
    # derived notes (risk/angle/elevation)
    for k, label in (("risk_note", "RiskNote"), ("angle_note", "AngleNote"), ("elevation_adjust_note", "ElevationNote")):
        v = notes.get(k)
        if v:
            parts.append(f"{label}: {v}")
    return "\n".join(parts)


def build_examples(enriched_iter):
    for obj in enriched_iter:
        hole = obj.get("hole")
        strategies = obj.get("tee_shot_strategy_options") or []
        descriptions = obj.get("descriptions") or []
        # Strategy selection examples
        cutoffs = [s.get("cutoff_distance") for s in strategies if isinstance(s.get("cutoff_distance"), int)]
        cutoffs_sorted = sorted({c for c in cutoffs if c is not None})
        strat_list_text = ", ".join(f"{c} yards" for c in cutoffs_sorted) if cutoffs_sorted else "(no structured options)"
        facts = _facts_block(obj)
        # one example per strategy option
        for s in strategies:
            cutoff = s.get("cutoff_distance")
            if not isinstance(cutoff, int):
                continue
            classification = s.get("classification")
            scenario = f"Player average drive: {cutoff} yards. Available tee strategy cutoffs: {strat_list_text}."
            prompt = (
                f"{facts}\n\nTee Strategy Selection Task:\n{scenario}\n"
                "Return only: 'Strategy: <number> yards' for the recommended tee shot cutoff distance."
            )
            yield {
                "task_type": "strategy_selection",
                "prompt": prompt,
                "completion": f"Strategy: {cutoff} yards",
                "expected_cutoff_yards": cutoff,
                "hole": hole,
                "par": obj.get("par"),
                "yardage": obj.get("yardage"),
                "tee_shot_strategies": [
                    {"cutoff_distance": s2.get("cutoff_distance"), "classification": s2.get("classification")}
                    for s2 in strategies
                    if isinstance(s2.get("cutoff_distance"), int)
                ],
                "classification": classification,
                "use_for_training": True,
            }
        # extra median scenario (if >1 cutoff)
        if len(cutoffs_sorted) > 1:
            med_val = int(median(cutoffs_sorted))
            # choose the largest cutoff <= median, else smallest
            target = max([c for c in cutoffs_sorted if c <= med_val] or [cutoffs_sorted[0]])
            prompt = (
                f"{facts}\n\nTee Strategy Selection Task:\nPlayer average drive: {med_val} yards. "
                f"Available tee strategy cutoffs: {strat_list_text}.\nReturn only: 'Strategy: <number> yards'."
            )
            yield {
                "task_type": "strategy_selection",
                "prompt": prompt,
                "completion": f"Strategy: {target} yards",
                "expected_cutoff_yards": target,
                "hole": hole,
                "par": obj.get("par"),
                "yardage": obj.get("yardage"),
                "tee_shot_strategies": [
                    {"cutoff_distance": s2.get("cutoff_distance"), "classification": s2.get("classification")}
                    for s2 in strategies
                    if isinstance(s2.get("cutoff_distance"), int)
                ],
                "classification": None,
                "use_for_training": True,
            }
        # Description synthesis examples
        gold_id = obj.get("gold_description_variant_id")
        for idx, desc in enumerate(descriptions, start=1):
            if not isinstance(desc, str) or not desc.strip():
                continue
            directive = "Generate a concise but information-dense strategic hole description grounded only in the provided facts."
            prompt = f"{facts}\n\nDescription Synthesis Task:\n{directive}\nReturn only the description paragraph."
            yield {
                "task_type": "description_synthesis",
                "prompt": prompt,
                "completion": desc.strip(),
                "hole": hole,
                "par": obj.get("par"),
                "yardage": obj.get("yardage"),
                "variant_id": idx,
                "gold": bool(idx == gold_id),
                "use_for_training": True,
            }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--enriched-file", required=True, help="Path to enriched holes JSONL file")
    ap.add_argument("--out", required=True, help="Output JSONL file for multi-task examples")
    args = ap.parse_args()
    in_path = Path(args.enriched_file)
    out_path = Path(args.out)
    examples = list(build_examples(_load_enriched(in_path)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"Wrote {len(examples)} examples -> {out_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
