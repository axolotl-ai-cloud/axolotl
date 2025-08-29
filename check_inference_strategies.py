import json
import re
import sys

# Usage: python check_inference_strategies.py --inference-file <path> --data-file <path> --log-file <path>
# Default files:
#   inference_output_jsonl: outputs/bethpage-lora/inference_1hour.jsonl
#   dataset_jsonl: data/bethpage_black/basics.jsonl

def load_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

STD_RE = re.compile(r"^\s*Strategy:\s*(\d{2,4})\s*yards\s*$", re.IGNORECASE)
NUM_YARD_RE = re.compile(r"(\d{2,4})\s*-?\s*yard", re.IGNORECASE)
DRIVE_RE_LIST = [
    re.compile(r"average drive:\s*(\d{2,4})\s*yards", re.IGNORECASE),
    re.compile(r"drives\s*(\d{2,4})\s*yards", re.IGNORECASE),
    re.compile(r"with a\s*(\d{2,4})-yard drive", re.IGNORECASE),
    re.compile(r"drive is exactly\s*(\d{2,4})\s*yards", re.IGNORECASE),
    re.compile(r"average drive of\s*(\d{2,4})\s*yards", re.IGNORECASE),
    re.compile(r"drive is\s*(\d{2,4})\s*yards", re.IGNORECASE),
]

def extract_drive_from_prompt(prompt: str):
    for rx in DRIVE_RE_LIST:
        m = rx.search(prompt)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None


def extract_strategy_from_completion(completion):
    m = STD_RE.search(completion)
    if m:
        return int(m.group(1))
    # fallback to any yards number
    m = NUM_YARD_RE.search(completion)
    if m:
        return int(m.group(1))
    return None


def expected_from_strategies(strategies, drive):
    cutoffs = sorted(s.get("cutoff_distance") for s in strategies if isinstance(s.get("cutoff_distance"), int))
    if not cutoffs:
        return None
    if drive is None:
        return cutoffs[-1] if len(cutoffs) > 1 else cutoffs[0]
    eligible = [c for c in cutoffs if c <= drive]
    return max(eligible) if eligible else cutoffs[0]


def test_strategies(inference_records, dataset_records):
    errors = []
    for entry, data in zip(inference_records, dataset_records):
        if not data.get("is_correct", True):
            # ignore negatives in validation pairwise zip
            continue
        strategies = data.get("tee_shot_strategies", [])
        prompt = entry["prompt"]
        completion = entry["completion"]
        drive = extract_drive_from_prompt(prompt)
        # Prefer explicit expected target in the dataset when present
        exp_explicit = data.get("expected_cutoff_yards")
        expected = int(exp_explicit) if isinstance(exp_explicit, int) else expected_from_strategies(strategies, drive)
        chosen = extract_strategy_from_completion(completion)
        if expected != chosen:
            errors.append({
                "hole": data.get("hole"),
                "expected": expected,
                "chosen": chosen,
                "prompt": prompt,
                "completion": completion,
            })
    return errors


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--inference-file", default="outputs/bethpage-lora/inference_1hour.jsonl")
    parser.add_argument("--data-file", default="data/bethpage_black/basics.jsonl")
    parser.add_argument("--log-file", default="outputs/bethpage-lora/inference_strategy_check_log.txt")
    args = parser.parse_args()
    inf_path = args.inference_file
    data_path = args.data_file
    log_path = args.log_file
    inference_records = load_jsonl(inf_path)
    dataset_records = load_jsonl(data_path)
    errors = test_strategies(inference_records, dataset_records)
    with open(log_path, "w", encoding="utf-8") as logf:
        logf.write(f"Strategy selection mismatches: {len(errors)}\n")
        for e in errors:
            logf.write(f"ERROR: Hole {e['hole']} | Expected: {e['expected']} | Chosen: {e['chosen']}\n")
            logf.write(f"Prompt: {e['prompt']}\nCompletion: {e['completion']}\n\n")
        if not errors:
            logf.write("✅ All strategy checks passed.\n")
    print(f"Log written to {log_path}")

if __name__ == "__main__":
    main()
