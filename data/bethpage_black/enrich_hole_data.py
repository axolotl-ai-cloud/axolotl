import json
import math
from pathlib import Path
from statistics import mean

BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR / "strategy_augmented_holes.jsonl"
OUTPUT_PATH = BASE_DIR / "strategy_augmented_holes_enriched.jsonl"

# Heuristic configuration
GOLD_MIN_WORDS = 70
GOLD_MAX_WORDS = 190

RISK_KEYWORDS = ["bunker", "water", "rough", "tree", "cross", "carry", "hazard", "fescue"]
ANGLE_KEYWORDS = ["angle", "dogleg", "corner", "line", "run up", "approach", "left", "right"]
ELEVATION_KEYWORDS = ["uphill", "downhill", "elevated", "drops", "rise", "climb"]
SPEED_KEYWORDS = ["firm", "soft", "hold", "release"]


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


def classify_strategy(strat, all_cutoffs):
    cutoff = strat.get("cutoff_distance") or 0
    if not all_cutoffs:
        return "neutral"
    mn = min(all_cutoffs)
    mx = max(all_cutoffs)
    # simple tiering: lowest = conservative, highest = aggressive, middle = neutral
    if cutoff == mn and cutoff != mx:
        return "conservative"
    if cutoff == mx and cutoff != mn:
        return "aggressive"
    return "neutral"


def extract_notes(record):
    attrs = record.get("attributes", {})
    theme = attrs.get("strategic_theme", "") or ""
    hazards = attrs.get("key_hazards", []) or []
    elevation = attrs.get("elevation_delta_yards")

    # risk note
    risk_bits = []
    for h in hazards:
        lower = h.lower()
        if any(k in lower for k in RISK_KEYWORDS):
            risk_bits.append(h)
    if attrs.get("water_hazard"):
        risk_bits.append("water in play")
    risk_note = ", ".join(dict.fromkeys(risk_bits)) or None

    # angle note
    angle_note = None
    shape = attrs.get("hole_shape") or ""
    if "dogleg" in shape:
        side = "right" if "right" in shape else ("left" if "left" in shape else "")
        angle_note = f"Dogleg {side} demands positional tee shot".strip()
    elif "gentle" in shape:
        angle_note = "Subtle bend influences preferred landing lane"

    # elevation note
    elevation_note = None
    if isinstance(elevation, (int, float)) and abs(elevation) >= 25:
        direction = "uphill" if elevation > 0 else "downhill"
        elevation_note = f"Significant {direction} change (~{elevation} ft est.)"

    # recommendation note (simple blend)
    rec = []
    if attrs.get("preferred_miss"):
        rec.append(f"miss {attrs['preferred_miss']}")
    recommended_for = ", ".join(rec) or None

    return {
        "risk_note": risk_note,
        "angle_note": angle_note,
        "elevation_adjust_note": elevation_note,
        "recommended_for": recommended_for,
        "theme": theme or None,
    }


def score_description(text, attrs):
    t = text.lower()
    words = [w for w in t.split() if w.isalpha() or any(c.isalpha() for c in w)]
    word_count = len(words)

    # coverage: fraction of hazard keywords present
    hazards = attrs.get("key_hazards", []) or []
    hazard_hits = 0
    for h in hazards:
        tokens = [tok for tok in h.lower().split() if tok.isalpha()]
        if tokens and all(tok in t for tok in tokens[:2]):
            hazard_hits += 1
    coverage = hazard_hits / max(1, len(hazards)) if hazards else 0

    # keyword diversity: count of unique risk/angle/surface tokens
    diversity_tokens = set()
    for kw in RISK_KEYWORDS + ANGLE_KEYWORDS + ELEVATION_KEYWORDS + SPEED_KEYWORDS:
        if kw in t:
            diversity_tokens.add(kw)
    diversity = len(diversity_tokens) / 20.0  # rough scale

    # length score center around 110 words (gaussian)
    mu = 110
    sigma = 35
    length_score = math.exp(-((word_count - mu) ** 2) / (2 * sigma ** 2))

    # final composite
    score = 0.45 * coverage + 0.25 * diversity + 0.30 * length_score
    return {
        "word_count": word_count,
        "coverage_score": round(coverage, 4),
        "diversity_score": round(diversity, 4),
        "length_score": round(length_score, 4),
        "composite_score": round(score, 4),
    }


def choose_gold_variant(descriptions, attrs):
    scored = []
    for idx, d in enumerate(descriptions, start=1):
        s = score_description(d, attrs)
        s["variant_id"] = idx
        s["text"] = d
        scored.append(s)
    # filter acceptable length range
    acceptable = [s for s in scored if GOLD_MIN_WORDS <= s["word_count"] <= GOLD_MAX_WORDS]
    pool = acceptable or scored
    pool.sort(key=lambda x: x["composite_score"], reverse=True)
    top = pool[0]
    return top["variant_id"], {f"variant_{s['variant_id']}": s for s in scored}


def enrich(records):
    enriched = []
    for rec in records:
        strategies = rec.get("tee_shot_strategy_options", [])
        cutoffs = [s.get("cutoff_distance") for s in strategies if s.get("cutoff_distance") is not None]
        # classify strategies
        for s in strategies:
            s["classification"] = classify_strategy(s, cutoffs)
        # derived notes
        notes = extract_notes(rec)
        # gold description
        gold_id, scoring = choose_gold_variant(rec.get("descriptions", []), rec.get("attributes", {}))
        rec["gold_description_variant_id"] = gold_id
        rec["derived_notes"] = notes
        rec["description_scoring"] = scoring
        enriched.append(rec)
    return enriched


def main():
    rows = load_jsonl(INPUT_PATH)
    enriched_rows = enrich(rows)
    write_jsonl(OUTPUT_PATH, enriched_rows)
    print(f"Enriched {len(enriched_rows)} holes -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
