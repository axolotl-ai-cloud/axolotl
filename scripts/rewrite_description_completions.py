#!/usr/bin/env python3
"""
Rewrite description_synthesis completions to be hole-specific using prompt bullets.
Generates a cleaned JSONL beside the original.

Usage:
  python scripts/rewrite_description_completions.py \
    --in data/bethpage_black/train_multitask_case_fixed.jsonl \
    --out data/bethpage_black/train_multitask_case_clean.jsonl

Optionally preview specific holes:
  python scripts/rewrite_description_completions.py --preview 8 9
"""

import json
import re
import argparse
from typing import List, Dict, Any


def fix_mojibake(s: str) -> str:
    repl = {
        "â€™": "’",
        "â€“": "–",
        "â€”": "—",
        "â€œ": "“",
        "â€": "”",
        "â€˜": "‘",
        "â€¦": "…",
        "Ã©": "é",
        "Ã": "A",
        "Â": "",
    }
    for k, v in repl.items():
        s = s.replace(k, v)
    return s


BULLET_RE = re.compile(r"\n\s*\d+\.\s*(.+?)\n", flags=re.DOTALL)

# Feature extractors: (name, regex list)
FEATURE_PATTERNS = {
    "water": [r"\b(pond|water|lake)\b"],
    "direction": [r"dogleg\s+(left|right)", r"(right|left)[-\s]*to[-\s]*(left|right)"],
    "elevation": [r"\b(elevated|uphill|downhill)\b"],
    "bunker_left": [r"bunker\s+(on|to)\s+the\s+left"],
    "bunker_right": [r"bunker\s+(on|to)\s+the\s+right"],
    "tree_right": [r"tree\s+(on|to)\s+the\s+right"],
    "tree_left": [r"tree\s+(on|to)\s+the\s+left"],
    "gully_blind": [r"\bgully\b", r"\bblind\b"],
    "wind": [r"\bwind\b", r"swirl"],
    "green_depth": [r"\b(\d{2})\s*yards?\s*deep\b"],
    "tiers": [r"\b(back\s+tier|tiered)\b"],
    "narrow": [r"\bnarrow\b", r"\bchute\b"],
}

GENERIC_BANS = [
    re.compile(r"combines strategic positioning with technical execution", re.I),
    re.compile(r"Success\b.*depends", re.I),
    re.compile(r"multi[-\s]?layered challenge", re.I),
    re.compile(r"course management takes precedence", re.I),
]


def extract_bullets(text: str) -> List[str]:
    bullets = [fix_mojibake(m.strip()) for m in BULLET_RE.findall((text or "") + "\n")]  # ensure trailing newline
    # Filter short fluff
    return [b for b in bullets if len(b) > 8]


def detect_features(bullets: List[str]) -> Dict[str, str]:
    feats: Dict[str, str] = {}
    joined = " \n ".join(bullets)
    for name, patterns in FEATURE_PATTERNS.items():
        for pat in patterns:
            m = re.search(pat, joined, flags=re.I)
            if m:
                feats[name] = m.group(0)
                break
    return feats


def sanitize(text: str) -> str:
    t = fix_mojibake(text)
    for ban in GENERIC_BANS:
        t = ban.sub("", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def build_paragraph(hole: int, par: int | None, yardage: int | None, bullets: List[str], feats: Dict[str, str]) -> str:
    # Sentences based on available features
    parts: List[str] = []
    # Intro avoids restating hole number; include par/yardage if known.
    intro_bits = []
    if par:
        intro_bits.append(f"par {par}")
    if yardage:
        intro_bits.append(f"{yardage} yards")
    # Alternate a few neutral openers to reduce boilerplate learning
    openers = [
        "rewards precise placement over brute force",
        "demands accuracy from tee to green",
        "punishes imprecision but offers scoring chances when well-positioned",
    ]
    opener = openers[(hole or 0) % len(openers)] if hole and isinstance(hole, int) else openers[0]
    intro = "This " + (", ".join(intro_bits) + " hole" if intro_bits else "hole") + f" {opener}."
    parts.append(intro)

    # Direction/elevation
    if "direction" in feats:
        parts.append(f"It moves {feats['direction']} off the tee, shaping how you choose your line and preferred shot shape.")
    if "elevation" in feats:
        parts.append("Elevation changes influence both distance control and spin on approach.")

    # Hazards / specific features
    haz = []
    if "water" in feats:
        haz.append("a pond guarding the front")
    if "bunker_left" in feats:
        haz.append("left‑side bunkers")
    if "bunker_right" in feats:
        haz.append("right‑side bunkers")
    if "tree_right" in feats:
        haz.append("a prominent tree on the right")
    if "tree_left" in feats:
        haz.append("a guarding tree on the left")
    if haz:
        parts.append("Off the tee, players must account for " + ", ".join(haz) + ".")

    # Wind / visibility
    if "wind" in feats:
        parts.append("Wind can swirl near the green, making club selection and start lines tricky.")
    if "gully_blind" in feats:
        parts.append("Missing to the safe side can leave a gully or even a blind second if you bail too far.")

    # Green details
    green_bits = []
    if "green_depth" in feats:
        depth_num = re.search(r"(\d{2})", feats["green_depth"]).group(1)
        green_bits.append(f"a green about {depth_num} yards deep")
    if "tiers" in feats:
        green_bits.append("a distinct back tier")
    if "narrow" in feats:
        green_bits.append("a narrow entrance")
    if green_bits:
        parts.append("The approach targets " + ", ".join(green_bits) + ", so landing angle and spin control are vital.")

    # Fallback from bullets if few features detected
    if len(parts) < 3 and bullets:
        sample = bullets[0]
        parts.append(sample)

    # Assemble and length tune (120–180 words target)
    text = " ".join(parts)
    text = sanitize(text)
    words = text.split()
    if len(words) < 110 and bullets:
        # append one more specific bullet line (sanitized)
        for b in bullets:
            b2 = sanitize(b)
            if b2 and b2 not in text and len(b2.split()) > 6:
                text = (text + " " + b2).strip()
                words = text.split()
            if len(words) >= 120:
                break
    # Trim lightly if overly long
    if len(words) > 190:
        text = " ".join(words[:185]).rstrip(",;") + "."
    return text


def rewrite_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    if rec.get("task_type") != "description_synthesis":
        return rec
    prompt = rec.get("prompt") or ""
    bullets = extract_bullets(prompt)
    feats = detect_features(bullets)
    par = rec.get("par") if isinstance(rec.get("par"), int) else None
    yard = rec.get("yardage") if isinstance(rec.get("yardage"), int) else None
    hole = rec.get("hole") if isinstance(rec.get("hole"), int) else None
    completion = build_paragraph(hole or -1, par, yard, bullets, feats)
    rec = dict(rec)
    rec["completion"] = completion
    rec["use_for_training"] = True
    return rec


def process(in_path: str, out_path: str, previews: List[int] | None = None) -> None:
    previews = previews or []
    with open(in_path, "r", encoding="utf-8-sig") as fin, open(out_path, "w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            rec = json.loads(line)
            newrec = rewrite_record(rec)
            fout.write(json.dumps(newrec, ensure_ascii=False) + "\n")
    if previews:
        print("Preview (cleaned completions):")
        count = 0
        with open(out_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("task_type") == "description_synthesis" and rec.get("hole") in previews:
                    print(f"\nHole {rec.get('hole')} ->\n{rec.get('completion')}\n")
                    count += 1
                    if count >= 6:
                        break


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", default="data/bethpage_black/train_multitask_case_fixed.jsonl")
    ap.add_argument("--out", dest="out_path", default="data/bethpage_black/train_multitask_case_clean.jsonl")
    ap.add_argument("--preview", dest="preview", nargs="*", type=int)
    args = ap.parse_args()
    process(args.in_path, args.out_path, args.preview)


if __name__ == "__main__":
    main()
