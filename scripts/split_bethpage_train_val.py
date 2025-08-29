import argparse
import json
import os
import random
import re
import hashlib
from typing import Dict, List, Tuple, Optional

DRIVE_RE_LIST = [
    re.compile(r"average drive:\s*(\d{2,4})\s*yards", re.IGNORECASE),
    re.compile(r"drives\s*(\d{2,4})\s*yards", re.IGNORECASE),
    re.compile(r"with a\s*(\d{2,4})-yard drive", re.IGNORECASE),
    re.compile(r"drive is exactly\s*(\d{2,4})\s*yards", re.IGNORECASE),
    re.compile(r"average drive of\s*(\d{2,4})\s*yards", re.IGNORECASE),
    re.compile(r"drive is\s*(\d{2,4})\s*yards", re.IGNORECASE),
]

def extract_drive_from_prompt(prompt: str) -> Optional[int]:
    for rx in DRIVE_RE_LIST:
        m = rx.search(prompt or "")
        if m:
            try:
                return int(m.group(1))
            except Exception:
                return None
    return None


def group_key(rec: Dict) -> str:
    hole = rec.get("hole", "NA")
    drive = extract_drive_from_prompt(rec.get("prompt", ""))
    return f"{hole}:{drive if drive is not None else 'NA'}"


def assign_split(keys: List[str], val_ratio: float, seed: int) -> Dict[str, str]:
    # deterministic via hashing; stable across runs
    split_map: Dict[str, str] = {}
    for k in keys:
        h = hashlib.md5((str(seed) + k).encode("utf-8")).hexdigest()
        # map hash to [0,1)
        frac = int(h[:8], 16) / 0xFFFFFFFF
        split_map[k] = "val" if frac < val_ratio else "train"
    return split_map


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--train-out", required=True)
    ap.add_argument("--val-out", required=True)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Build groups by (hole, drive) to avoid leakage among paraphrases
    groups: Dict[str, List[Dict]] = {}
    for r in records:
        k = group_key(r)
        groups.setdefault(k, []).append(r)

    split_map = assign_split(list(groups.keys()), args.val_ratio, args.seed)

    train_records: List[Dict] = []
    val_records: List[Dict] = []

    for k, recs in groups.items():
        split = split_map[k]
        for r in recs:
            r = dict(r)  # shallow copy
            r["use_for_training"] = (split == "train") and (r.get("is_correct", True) is not False)
            r["split"] = split
            if split == "train":
                train_records.append(r)
            else:
                val_records.append(r)

    # Write outputs
    with open(args.train_out, "w", encoding="utf-8") as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(args.val_out, "w", encoding="utf-8") as f:
        for r in val_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_records)} train and {len(val_records)} val records.")

if __name__ == "__main__":
    main()
