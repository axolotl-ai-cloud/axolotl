import json
from pathlib import Path
import random
import re
from collections import Counter
from datetime import datetime

BASE_DIR = Path(__file__).parent
ENRICHED_PATH = BASE_DIR / "strategy_augmented_holes_enriched.jsonl"
OUT_TRAIN = BASE_DIR / "training_bethpage_multitask.train.jsonl"
OUT_EVAL = BASE_DIR / "training_bethpage_multitask.eval.jsonl"
QA_REPORT = BASE_DIR / "training_bethpage_multitask.qa_report.json"

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Task definitions:
# 1. strategy_selection: choose best tee shot strategy.
# 1b. strategy_selection_negative: identify the WORST (clearly suboptimal) strategy and explain why (contrast signal).
# 2. description_generation: generate high-quality narrative (gold variant target) + rationale (hazard/angle/elevation mapping) for transparency.
# 3. style_rewrite: rewrite gold description into a concise television commentary style (brevity + punch) to promote stylistic control.

# Output example schema per line:
# {"task": "strategy_selection", "hole": 1, "input": { ... prompt content ... }, "output": {"strategy_id": "cutoff_320", "classification": "aggressive", "rationale": "..."}}
# {"task": "description_generation", "hole": 1, "input": {...}, "output": {"description": "..."}}


def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def build_strategy_selection_examples(rec):
    strategies = rec.get("tee_shot_strategy_options", [])
    if not strategies:
        return []
    # Choose target = most aggressive (classification == aggressive) else highest cutoff
    aggressive = [s for s in strategies if s.get("classification") == "aggressive"]
    if aggressive:
        target = aggressive[0]
    else:
        # fallback highest cutoff distance
        target = max(strategies, key=lambda s: s.get("cutoff_distance") or 0)

    distractors = [s for s in strategies if s is not target]

    # Simple rationale using derived notes + delta vs conservative if exists
    notes = rec.get("derived_notes", {})
    risk_note = notes.get("risk_note")
    angle_note = notes.get("angle_note")
    elevation_note = notes.get("elevation_adjust_note")
    rationale_bits = []
    for bit in [risk_note, angle_note, elevation_note]:
        if bit:
            rationale_bits.append(bit)
    rationale = ", ".join(rationale_bits) or None

    prompt = {
        "par": rec.get("par"),
        "yardage": rec.get("yardage"),
        "attributes": rec.get("attributes", {}),
        "candidate_strategies": [
            {"strategy_id": s.get("strategy_id"), "cutoff_distance": s.get("cutoff_distance"), "classification": s.get("classification"), "remaining_distance": s.get("remaining_distance")} for s in strategies
        ],
        "instruction": "Select the best tee shot strategy for this hole (one strategy_id)."
    }

    output = {
        "strategy_id": target.get("strategy_id"),
        "classification": target.get("classification"),
        "rationale": rationale
    }
    examples = [{
        "task": "strategy_selection",
        "hole": rec.get("hole"),
        "input": prompt,
        "output": output
    }]

    # Negative (worst) strategy example: pick most conservative (lowest cutoff) if not the chosen one else next lowest
    sorted_by_cutoff = sorted(strategies, key=lambda s: s.get("cutoff_distance") or 0)
    worst = None
    for cand in sorted_by_cutoff:
        if cand.get("strategy_id") != output["strategy_id"]:
            worst = cand
            break
    if worst and worst.get("strategy_id") != output["strategy_id"]:
        worst_rationale_bits = []
        if risk_note:
            worst_rationale_bits.append("fails to mitigate " + risk_note.split(',')[0])
        if angle_note:
            worst_rationale_bits.append("poor angle vs dogleg")
        if elevation_note:
            worst_rationale_bits.append("leaves tougher elevation adjustment")
        worst_rationale = ", ".join(dict.fromkeys(worst_rationale_bits)) or None
        neg_prompt = {
            "par": rec.get("par"),
            "yardage": rec.get("yardage"),
            "attributes": rec.get("attributes", {}),
            "candidate_strategies": prompt["candidate_strategies"],
            "instruction": "Identify the least optimal tee shot strategy (one strategy_id) and justify briefly."
        }
        neg_output = {
            "strategy_id": worst.get("strategy_id"),
            "classification": worst.get("classification"),
            "rationale": worst_rationale
        }
        examples.append({
            "task": "strategy_selection_negative",
            "hole": rec.get("hole"),
            "input": neg_prompt,
            "output": neg_output
        })

    return examples


def build_description_generation_examples(rec):
    gold_id = rec.get("gold_description_variant_id")
    descriptions = rec.get("descriptions", [])
    if not gold_id or gold_id < 1 or gold_id > len(descriptions):
        return []
    target_text = descriptions[gold_id - 1]
    notes = rec.get("derived_notes", {})

    # Provide one strategy classification summary: prefer aggressive present else gold chooses aggressive vs conservative by heuristic
    strategies = rec.get("tee_shot_strategy_options", [])
    aggressive = [s for s in strategies if s.get("classification") == "aggressive"]
    chosen = aggressive[0] if aggressive else (strategies[-1] if strategies else None)

    prompt = {
        "par": rec.get("par"),
        "yardage": rec.get("yardage"),
        "attributes": rec.get("attributes", {}),
        "strategy_context": {
            "strategy_id": chosen.get("strategy_id") if chosen else None,
            "classification": chosen.get("classification") if chosen else None,
            "cutoff_distance": chosen.get("cutoff_distance") if chosen else None
        },
        "derived_notes": notes,
        "instruction": "Write a high-quality, specific hole description incorporating strategic context and hazards."
    }

    # Build rationale mapping hazards & structural notes present
    rationale_items = []
    if notes.get("risk_note"):
        rationale_items.append({"risk_note": notes["risk_note"]})
    if notes.get("angle_note"):
        rationale_items.append({"angle_note": notes["angle_note"]})
    if notes.get("elevation_adjust_note"):
        rationale_items.append({"elevation_adjust_note": notes["elevation_adjust_note"]})
    rationale = rationale_items or None

    output = {"description": target_text, "rationale": rationale}

    examples = [{
        "task": "description_generation",
        "hole": rec.get("hole"),
        "input": prompt,
        "output": output
    }]

    # Style rewrite task (concise tv commentary: 45 word cap)
    concise = compress_description(rec, target_text, notes, chosen)
    rewrite_prompt = {
        "base_description": target_text,
        "instruction": "Rewrite this hole description as concise television commentary (<45 words), keep key hazards & strategy tone." ,
        "max_words": 45
    }
    examples.append({
        "task": "style_rewrite",
        "hole": rec.get("hole"),
        "input": rewrite_prompt,
        "output": {"description": concise}
    })
    return examples


def compress_description(rec, full_text, notes, chosen_strategy, max_words=45):
    """Heuristic compression: prioritize structure -> opening context, key risk, strategic directive, green/elevation if present."""
    hole = rec.get("hole")
    attrs = rec.get("attributes", {})
    pieces = []
    # Opening: Par / yardage (compact)
    par = rec.get("par")
    yard = rec.get("yardage")
    if par and yard:
        pieces.append(f"Par {par} {yard}y")
    # Shape + elevation
    shape = attrs.get("hole_shape") or ""
    elev = attrs.get("elevation_delta_yards")
    elev_phrase = None
    if isinstance(elev, (int, float)) and abs(elev) >= 25:
        elev_phrase = "downhill" if elev < 0 else "uphill"
    if shape:
        shape_tokens = shape.replace("gentle_", "").replace("sharp_", "").replace("_", " ")
        if elev_phrase:
            pieces.append(f"{elev_phrase} {shape_tokens}")
        else:
            pieces.append(shape_tokens)
    elif elev_phrase:
        pieces.append(elev_phrase)
    # Strategy directive
    if chosen_strategy:
        cls = chosen_strategy.get("classification")
        cutoff = chosen_strategy.get("cutoff_distance")
        if cls and cutoff:
            pieces.append(f"{cls} tee plays to ~{cutoff}")
    # Risk / angle / elevation notes
    if notes.get("risk_note"):
        pieces.append(notes["risk_note"].split(',')[0])
    if notes.get("angle_note"):
        pieces.append(notes["angle_note"].split(' demands')[0])
    if notes.get("elevation_adjust_note") and elev_phrase is None:
        pieces.append(notes["elevation_adjust_note"].split(' (~')[0])
    # Preferred miss
    pref = attrs.get("preferred_miss")
    if pref:
        pieces.append(f"favor {pref}")
    # Green complexity succinct
    gc = attrs.get("green_complex_notes")
    if gc:
        # take first clause up to comma
        first_clause = re.split(r'[.;]', gc)[0]
        simple = first_clause.replace("two-tiered", "2-tier")
        if len(simple.split()) <= 6:
            pieces.append(simple)
    sentence = ", ".join(pieces)
    words = sentence.split()
    if len(words) > max_words:
        sentence = " ".join(words[:max_words])
    return sentence


def split_train_eval(all_examples, eval_ratio=0.22):
    # group by hole to avoid leakage: hold out some holes entirely
    holes = sorted({ex["hole"] for ex in all_examples})
    eval_holes_count = max(1, int(len(holes) * eval_ratio))
    eval_holes = set(random.sample(holes, eval_holes_count))

    train, eval_set = [], []
    for ex in all_examples:
        if ex["hole"] in eval_holes:
            eval_set.append(ex)
        else:
            train.append(ex)
    return train, eval_set, eval_holes


# -------------------- QA VALIDATION --------------------
def _tokenize_words(text: str):
    # Simple word split; treat commas & punctuation as separators
    return [w for w in re.split(r"[^A-Za-z0-9+'-]+", text.strip()) if w]


def validate_style_rewrites(examples, max_words_default=45):
    records = []
    total = 0
    over_limit = 0
    length_distribution = Counter()
    for ex in examples:
        if ex.get("task") != "style_rewrite":
            continue
        total += 1
        output = ex.get("output", {})
        desc = output.get("description", "")
        words = _tokenize_words(desc)
        length = len(words)
        length_distribution[length] += 1
        # Input may carry explicit max_words
        input_obj = ex.get("input", {})
        max_words = input_obj.get("max_words", max_words_default)
        if length > max_words:
            over_limit += 1
            records.append({
                "hole": ex.get("hole"),
                "length": length,
                "max_words": max_words,
                "description": desc
            })
    return {
        "task": "style_rewrite",
        "total": total,
        "over_limit_count": over_limit,
        "over_limit_examples": records[:10],  # cap examples
        "length_distribution_top": length_distribution.most_common(25)
    }


def validate_rationales(examples):
    # Ensure any rationale hazard/angle/elevation notes appear (substring case-insensitive) in either: original description (if available) OR attributes/notes fields.
    issues = []
    checked = 0
    missing_notes = 0
    for ex in examples:
        if ex.get("task") != "description_generation":
            continue
        checked += 1
        out = ex.get("output", {})
        rationale = out.get("rationale")
        if not rationale:
            continue
        # Build searchable corpus
        input_obj = ex.get("input", {})
        attrs = input_obj.get("attributes", {}) or {}
        derived = input_obj.get("derived_notes", {}) or {}
        text_corpus = " ".join([
            json.dumps(attrs, ensure_ascii=False),
            json.dumps(derived, ensure_ascii=False),
            out.get("description", "")
        ]).lower()
        for item in rationale:
            # item is dict with single key
            for k, v in item.items():
                if not v:
                    continue
                probe = v.split(' (~')[0].split(',')[0].lower().strip()
                if probe and probe not in text_corpus:
                    missing_notes += 1
                    issues.append({
                        "hole": ex.get("hole"),
                        "note_type": k,
                        "value": v
                    })
    return {
        "task": "description_generation.rationale",
        "examples_with_rationale": checked,
        "missing_note_occurrences": missing_notes,
        "missing_examples": issues[:15]
    }


def run_qa(train_examples, eval_examples):
    style_report = validate_style_rewrites(train_examples + eval_examples)
    rationale_report = validate_rationales(train_examples + eval_examples)
    aggregate = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "style_rewrite": style_report,
        "rationale_grounding": rationale_report
    }
    with QA_REPORT.open('w', encoding='utf-8') as f:
        json.dump(aggregate, f, ensure_ascii=False, indent=2)
    print(f"QA report written: {QA_REPORT.name} | style over-limit: {style_report['over_limit_count']} | missing rationale notes: {rationale_report['missing_note_occurrences']}")


def main():
    records = load_jsonl(ENRICHED_PATH)
    examples = []
    for rec in records:
        examples.extend(build_strategy_selection_examples(rec))
        examples.extend(build_description_generation_examples(rec))

    train, eval_set, eval_holes = split_train_eval(examples)
    write_jsonl(OUT_TRAIN, train)
    write_jsonl(OUT_EVAL, eval_set)
    print(f"Wrote {len(train)} train and {len(eval_set)} eval examples. Eval holes: {sorted(eval_holes)}")
    # Run QA after writing
    run_qa(train, eval_set)


if __name__ == "__main__":
    main()
