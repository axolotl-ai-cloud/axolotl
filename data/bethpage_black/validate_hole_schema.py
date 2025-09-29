import json
import hashlib
from pathlib import Path
from datetime import datetime, UTC
from typing import List, Dict, Any

BASE_DIR = Path(__file__).parent
ENRICHED_PATH = BASE_DIR / "strategy_augmented_holes_enriched.jsonl"
REPORT_PATH = BASE_DIR / "strategy_augmented_holes_enriched.validation_report.json"

# Define enumerations / constraints based on observed schema (minus removed fields like wind_exposure, green_receptiveness)
GRADIENT_ENUM = {"uphill", "downhill", "flat"}
HOLE_LENGTH_ENUM = {"short", "medium", "long"}
# hole_shape already includes prefixes like sharp_/gentle_ and direction; allow any that end with dogleg_left/right or straight
ALLOWED_HOLE_SHAPE_PREFIXES = ("sharp_dogleg_left", "sharp_dogleg_right", "gentle_dogleg_left", "gentle_dogleg_right", "straight")
PRIMARY_HAZARD_TYPES = {"trees", "bunker", "water", "fescue", "crosswinds", "narrow green", "bunker", "fescue"}
SECONDARY_HAZARD_TYPES = {"none", "trees", "bunker", "fescue", "water", "narrow green"}
GREEN_SIZE_ENUM = {"small", "medium", "large"}

REQUIRED_TOP_LEVEL = [
    "hole", "par", "yardage", "attributes", "tee_shot_strategy_options", "descriptions", "gold_description_variant_id", "derived_notes"
]
REQUIRED_ATTRIBUTE_FIELDS = [
    "water_hazard", "gradient", "hole_length", "hole_shape", "elevation_delta_yards", "primary_hazard_type", "secondary_hazard_type", "green_size", "strategic_theme", "key_hazards", "green_complex_notes"
]

# Forbidden fields (explicitly removed earlier in pipeline)
FORBIDDEN_FIELDS = {"wind_exposure", "green_receptiveness"}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def validate_record(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    issues = []
    # Top-level required
    for field in REQUIRED_TOP_LEVEL:
        if field not in rec:
            issues.append({"severity": "error", "field": field, "message": "missing required top-level field"})
    # Forbidden
    forbidden_present = FORBIDDEN_FIELDS.intersection(rec.keys())
    for f in forbidden_present:
        issues.append({"severity": "error", "field": f, "message": "forbidden field present"})

    attrs = rec.get("attributes", {}) or {}
    for field in REQUIRED_ATTRIBUTE_FIELDS:
        if field not in attrs:
            issues.append({"severity": "error", "field": f"attributes.{field}", "message": "missing required attribute"})
    # Attribute enums
    grad = attrs.get("gradient")
    if grad and grad not in GRADIENT_ENUM:
        issues.append({"severity": "error", "field": "attributes.gradient", "value": grad, "message": "invalid gradient"})
    hl = attrs.get("hole_length")
    if hl and hl not in HOLE_LENGTH_ENUM:
        issues.append({"severity": "error", "field": "attributes.hole_length", "value": hl, "message": "invalid hole_length"})
    shape = attrs.get("hole_shape")
    if shape and not any(shape.startswith(prefix) for prefix in ALLOWED_HOLE_SHAPE_PREFIXES):
        issues.append({"severity": "warning", "field": "attributes.hole_shape", "value": shape, "message": "unexpected hole_shape pattern"})
    gsize = attrs.get("green_size")
    if gsize and gsize not in GREEN_SIZE_ENUM:
        issues.append({"severity": "error", "field": "attributes.green_size", "value": gsize, "message": "invalid green_size"})
    prim = attrs.get("primary_hazard_type")
    if prim is not None and prim not in PRIMARY_HAZARD_TYPES and prim != None:
        issues.append({"severity": "warning", "field": "attributes.primary_hazard_type", "value": prim, "message": "unrecognized primary_hazard_type"})
    sec = attrs.get("secondary_hazard_type")
    if sec and sec not in SECONDARY_HAZARD_TYPES:
        issues.append({"severity": "warning", "field": "attributes.secondary_hazard_type", "value": sec, "message": "unrecognized secondary_hazard_type"})

    # Numeric sanity
    elev = attrs.get("elevation_delta_yards")
    if isinstance(elev, (int, float)):
        if abs(elev) > 150:
            issues.append({"severity": "warning", "field": "attributes.elevation_delta_yards", "value": elev, "message": "elevation magnitude unusually large"})
    else:
        issues.append({"severity": "error", "field": "attributes.elevation_delta_yards", "value": elev, "message": "elevation must be numeric"})

    # Strategy options
    strategies = rec.get("tee_shot_strategy_options", [])
    if not strategies:
        issues.append({"severity": "error", "field": "tee_shot_strategy_options", "message": "no strategies present"})
    else:
        seen_ids = set()
        cutoff_prev = -1
        for s in strategies:
            sid = s.get("strategy_id")
            if not sid:
                issues.append({"severity": "error", "field": "tee_shot_strategy_options.strategy_id", "message": "missing strategy_id"})
            elif sid in seen_ids:
                issues.append({"severity": "error", "field": "tee_shot_strategy_options.strategy_id", "value": sid, "message": "duplicate strategy_id"})
            else:
                seen_ids.add(sid)
            cutoff = s.get("cutoff_distance")
            if not isinstance(cutoff, (int, float)):
                issues.append({"severity": "error", "field": f"strategy[{sid}].cutoff_distance", "value": cutoff, "message": "cutoff_distance must be numeric"})
            elif cutoff < 0:
                issues.append({"severity": "error", "field": f"strategy[{sid}].cutoff_distance", "value": cutoff, "message": "negative cutoff_distance"})
            classification = s.get("classification")
            if classification not in {"conservative", "neutral", "aggressive"}:
                issues.append({"severity": "error", "field": f"strategy[{sid}].classification", "value": classification, "message": "invalid classification"})
            # enforce non-decreasing cutoff ordering expectation (heuristic)
            if cutoff is not None and cutoff < cutoff_prev:
                issues.append({"severity": "warning", "field": f"strategy[{sid}].cutoff_distance", "value": cutoff, "message": "cutoff_distance not sorted ascending"})
            if isinstance(cutoff, (int, float)):
                cutoff_prev = cutoff

    # Descriptions
    descriptions = rec.get("descriptions", [])
    if not descriptions or len(descriptions) != 5:
        issues.append({"severity": "error", "field": "descriptions", "message": "expected exactly 5 descriptions"})
    else:
        # Gold id validity
        gold_id = rec.get("gold_description_variant_id")
        if not isinstance(gold_id, int) or not (1 <= gold_id <= len(descriptions)):
            issues.append({"severity": "error", "field": "gold_description_variant_id", "value": gold_id, "message": "gold id out of range"})
        # Basic duplicate detection
        normalized = [d.strip().lower() for d in descriptions]
        dup_counts = {}
        for d in normalized:
            dup_counts[d] = dup_counts.get(d, 0) + 1
        for text, count in dup_counts.items():
            if count > 1:
                issues.append({"severity": "warning", "field": "descriptions", "message": "duplicate description text", "snippet": text[:80]})

    # Derived notes coherence: if risk_note mentions 'water' ensure water_hazard True OR primary/secondary hazard water
    derived = rec.get("derived_notes", {}) or {}
    risk_note = derived.get("risk_note") or ""
    if risk_note:
        risk_lower = risk_note.lower()
        if "water" in risk_lower:
            if not attrs.get("water_hazard") and attrs.get("primary_hazard_type") != "water" and attrs.get("secondary_hazard_type") != "water":
                issues.append({"severity": "warning", "field": "derived_notes.risk_note", "value": risk_note, "message": "mentions water but no water hazard in attributes"})

    return issues


def main():
    if not ENRICHED_PATH.exists():
        raise SystemExit(f"Missing source file: {ENRICHED_PATH}")
    records = load_jsonl(ENRICHED_PATH)
    all_issues: Dict[int, List[Dict[str, Any]]] = {}
    severity_counts = {"error": 0, "warning": 0}
    for rec in records:
        issues = validate_record(rec)
        if issues:
            all_issues[rec.get("hole")] = issues
            for it in issues:
                severity_counts[it["severity"]] = severity_counts.get(it["severity"], 0) + 1
    report = {
        "generated_at": datetime.now(UTC).isoformat(),
        "source_file": str(ENRICHED_PATH.name),
        "source_sha256": file_sha256(ENRICHED_PATH),
        "hole_count": len(records),
        "severity_counts": severity_counts,
        "issues": all_issues
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"Validation complete: {REPORT_PATH.name} errors={severity_counts['error']} warnings={severity_counts['warning']}")
    if severity_counts["error"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
