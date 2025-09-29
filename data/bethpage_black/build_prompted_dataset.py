import json
from pathlib import Path
from typing import Any, Dict, List

INPUT_FILE = Path(__file__).parent / "training_bethpage_multitask.weighted.train.jsonl"
OUTPUT_FILE = Path(__file__).parent / "training_bethpage_multitask.weighted.prompted.train.jsonl"
SPEC_FILE = Path(__file__).parent / "prompt_template_spec.md"

SYSTEM_MESSAGE = (
    "You are a concise, knowledgeable golf course strategy analyst. Always be precise, "
    "avoid hallucinations, and keep units in yards when distances are given. If data is absent, do not invent it."
)


def _norm(v: Any) -> str:
    if v is None or v == "":
        return "unknown"
    return str(v)


def _join_list(values: List[Any]) -> str:
    if not values:
        return "None"
    return ", ".join(str(v) for v in values)


def _build_strategies_block(candidates):
    lines = []
    for c in candidates:
        lines.append(
            f"- {c['strategy_id']} ({c.get('classification','unknown')}) remaining={_norm(c.get('remaining_distance'))}y"
        )
    return "\n" + "\n".join(lines) if lines else ""


def _condense_derived(derived):
    if not derived:
        return "None"
    parts = []
    for k, v in derived.items():
        if v:
            parts.append(f"{k}:{v}")
    return "; ".join(parts) if parts else "None"


def _build_selection_user(example: Dict[str, Any], negative: bool = False) -> str:
    inp = example["input"]
    attrs = inp["attributes"]
    header = (
        "Task: Identify least optimal tee shot strategy." if negative else "Task: Select optimal tee shot strategy."
    )
    user = [
        header,
        f"Hole: {example['hole']} | Par {inp['par']} | Yardage {inp['yardage']}y",
        "Attributes:",
        f"- Gradient: {_norm(attrs.get('gradient'))}",
        f"- Shape: {_norm(attrs.get('hole_shape'))}",
        f"- Length: {_norm(attrs.get('hole_length'))}",
        f"- Elevation Δ: {_norm(attrs.get('elevation_delta_yards'))}y",
        f"- Landing Zone Width: {_norm(attrs.get('tee_landing_zone_width_yards'))}y",
        f"- Primary Hazard: {_norm(attrs.get('primary_hazard_type'))}",
        f"- Secondary Hazard: {_norm(attrs.get('secondary_hazard_type'))}",
        f"- Green Size: {_norm(attrs.get('green_size'))}",
        f"Key Hazards: {_join_list(attrs.get('key_hazards') or [])}",
        f"Strategic Theme: {_norm(attrs.get('strategic_theme'))}",
        "Candidate Strategies:",
        _build_strategies_block(inp.get('candidate_strategies') or []),
        "Instruction: Return ONLY the chosen strategy_id on a single line. If rationale known, append a short justification after a tab.",
    ]
    return "\n".join([s for s in user if s is not None])


def _build_description_user(example: Dict[str, Any]) -> str:
    inp = example["input"]
    attrs = inp["attributes"]
    strat = inp.get("strategy_context", {})
    derived = inp.get("derived_notes", {})
    user = [
        "Task: Write hole description.",
        f"Hole: {example['hole']} | Par {inp['par']} | Yardage {inp['yardage']}y",
        f"Theme: {_norm(attrs.get('strategic_theme'))}",
        f"Strategy: {_norm(strat.get('strategy_id'))} ({_norm(strat.get('classification'))}) cutoff={_norm(strat.get('cutoff_distance'))}y",
        f"Key Hazards: {_join_list(attrs.get('key_hazards') or [])}",
        f"Derived Notes: {_condense_derived(derived)}",
        "Instruction: Write a vivid, factual description of the hole incorporating strategic context and hazards. Do not add scoring statistics unless provided. Avoid repetition.",
    ]
    return "\n".join(user)


def _build_style_rewrite_user(example: Dict[str, Any]) -> str:
    inp = example["input"]
    user = [
        "Task: Rewrite hole description into concise TV commentary (<45 words).",
        "Original:",
        inp.get("base_description", "unknown"),
        "Instruction: Keep strategic tone and core hazards; remove fluff.",
    ]
    return "\n".join(user)


def _assistant_output(example: Dict[str, Any]) -> str:
    task = example["task"]
    out = example.get("output") or {}
    if task in ("strategy_selection", "strategy_selection_negative"):
        strat = out.get("strategy_id", "")
        rationale = out.get("rationale")
        if rationale:
            return f"{strat}\t{rationale}"
        return strat
    if task == "description_generation":
        desc = out.get("description", "")
        rat = out.get("rationale")
        if isinstance(rat, list) and rat:
            # each element is a singleton dict
            pairs = []
            for d in rat:
                for k, v in d.items():
                    if v:
                        pairs.append(f"{k}:{v}")
            if pairs:
                return desc.rstrip() + "\n---\nRationale: " + "; ".join(pairs)
        return desc
    if task == "style_rewrite":
        return out.get("description", "")
    return json.dumps(out)


def build():
    records = []
    with INPUT_FILE.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))

    # ensure UTF-8 encoding for Windows environments (avoid cp1252 errors on Δ etc.)
    out_f = OUTPUT_FILE.open("w", encoding="utf-8")
    count = 0
    for ex in records:
        task = ex["task"]
        if task == "strategy_selection":
            user = _build_selection_user(ex, negative=False)
        elif task == "strategy_selection_negative":
            user = _build_selection_user(ex, negative=True)
        elif task == "description_generation":
            user = _build_description_user(ex)
        elif task == "style_rewrite":
            user = _build_style_rewrite_user(ex)
        else:
            # skip unknown
            continue
        assistant = _assistant_output(ex)
        prompted = {
            "task": task,
            "hole": ex.get("hole"),
            "sample_weight": ex.get("sample_weight", 1.0),
            "messages": [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ],
            "meta": {"original": ex},
        }
        out_f.write(json.dumps(prompted, ensure_ascii=False) + "\n")
        count += 1
    out_f.close()
    print(f"Wrote {count} prompted examples to {OUTPUT_FILE.name}")


if __name__ == "__main__":
    build()
