import argparse
import json
import os
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

DRIVE_RE_LIST = [
    re.compile(r"average drive:\s*(\d{2,4})\s*yards", re.IGNORECASE),
    re.compile(r"drives\s*(\d{2,4})\s*yards", re.IGNORECASE),
    re.compile(r"with a\s*(\d{2,4})-yard drive", re.IGNORECASE),
    re.compile(r"drive is exactly\s*(\d{2,4})\s*yards", re.IGNORECASE),
    re.compile(r"average drive of\s*(\d{2,4})\s*yards", re.IGNORECASE),
    re.compile(r"drive is\s*(\d{2,4})\s*yards", re.IGNORECASE),
]
NUM_YARD_RE = re.compile(r"(\d{2,4})\s*-?\s*yard", re.IGNORECASE)


def extract_drive_from_prompt(prompt: str) -> Optional[int]:
    for rx in DRIVE_RE_LIST:
        m = rx.search(prompt)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def extract_cutoff_from_text(text: str) -> Optional[int]:
    # Return the last number of yards mentioned; often the "correct" is last
    matches = NUM_YARD_RE.findall(text or "")
    if matches:
        try:
            return int(matches[-1])
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
        return cutoffs[0] if len(cutoffs) == 1 else cutoffs[-1]
    eligible = [c for c in cutoffs if c <= drive]
    return max(eligible) if eligible else cutoffs[0]


def ensure_strategy_ids(strategies: List[Dict[str, Any]]) -> None:
    seen = set()
    for s in strategies:
        cutoff = s.get("cutoff_distance")
        sid = s.get("strategy_id")
        if not sid:
            sid = f"cutoff_{cutoff}" if isinstance(cutoff, int) else f"strategy_{len(seen)}"
        # Enforce uniqueness
        base_sid = sid
        i = 1
        while sid in seen:
            sid = f"{base_sid}_{i}"
            i += 1
        s["strategy_id"] = sid
        # Optional human label
        if not s.get("strategy_label"):
            s["strategy_label"] = f"cutoff {cutoff} yards" if isinstance(cutoff, int) else base_sid
        seen.add(sid)


def augment_record(obj: Dict[str, Any]) -> Dict[str, Any]:
    strategies = obj.get("tee_shot_strategies") or []
    if isinstance(strategies, list):
        ensure_strategy_ids(strategies)
        obj["tee_shot_strategies"] = strategies

    # Expected cutoff: prefer deriving from prompt+strategies; else from completion text
    drive = extract_drive_from_prompt(obj.get("prompt", ""))
    expected = choose_cutoff(strategies, drive)
    if expected is None:
        expected = extract_cutoff_from_text(obj.get("completion", ""))
    if expected is not None:
        obj["expected_cutoff_yards"] = int(expected)
        # Map to a strategy_id if present
        sid = None
        for s in strategies:
            if s.get("cutoff_distance") == expected:
                sid = s.get("strategy_id") or s.get("strategy_label")
                break
        if sid:
            obj["expected_strategy_id"] = sid
    return obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/bethpage_black/basics.jsonl", help="Path to input JSONL")
    ap.add_argument(
        "--output",
        default=None,
        help="Path to output JSONL (default: in-place update; backup created)",
    )
    args = ap.parse_args()

    in_path = args.input
    out_path = args.output

    with open(in_path, "r", encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]

    new_lines: List[str] = []
    updated = 0
    for line in lines:
        obj = json.loads(line)
        before_has = ("expected_cutoff_yards" in obj) or ("expected_strategy_id" in obj)
        obj = augment_record(obj)
        after_has = ("expected_cutoff_yards" in obj) or ("expected_strategy_id" in obj)
        if after_has and not before_has:
            updated += 1
        new_lines.append(json.dumps(obj, ensure_ascii=False))

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"Wrote augmented dataset to {out_path} (updated {updated} records)")
    else:
        # In-place with backup
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{in_path}.{ts}.bak"
        with open(backup_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        with open(in_path, "w", encoding="utf-8") as f:
            f.write("\n".join(new_lines) + "\n")
        print(f"Updated {in_path} in place (backup at {backup_path}; updated {updated} records)")


if __name__ == "__main__":
    main()
