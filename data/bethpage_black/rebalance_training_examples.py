import json
from pathlib import Path
from collections import Counter
import math

BASE_DIR = Path(__file__).parent
TRAIN_PATH = BASE_DIR / "training_bethpage_multitask.train.jsonl"
WEIGHTED_PATH = BASE_DIR / "training_bethpage_multitask.weighted.train.jsonl"
SUMMARY_PATH = BASE_DIR / "training_bethpage_multitask.weighting_summary.json"

# Chosen weighting rationale:
# - description_generation: rich semantic grounding; double it (2.0)
# - style_rewrite: stylistic control but derivative of description; modest boost (1.5)
# - strategy_selection (+ negative): keep but slightly under description weight (1.0 baseline)
# Weight application uses integer replication = ceil(weight * base_multiplier)
# We compute base_multiplier such that the heaviest weight (2.0) roughly doubles without exploding dataset size.

WEIGHTS = {
    "description_generation": 2.0,
    "style_rewrite": 1.5,
    "strategy_selection": 1.0,
    "strategy_selection_negative": 1.0,
}

BASE_MULTIPLIER = 1  # since weights already encode replication scale; adjust if larger dataset desired


def load_jsonl(path: Path):
    items = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: Path, rows):
    with path.open('w', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')


def main():
    data = load_jsonl(TRAIN_PATH)
    orig_counts = Counter(d['task'] for d in data)
    weighted = []
    for ex in data:
        task = ex['task']
        w = WEIGHTS.get(task, 1.0)
        replication = max(1, math.ceil(w * BASE_MULTIPLIER))
        for i in range(replication):
            # Add explicit per-sample weight (model trainer can also use field)
            clone = ex.copy()
            clone['sample_weight'] = w
            if replication > 1:
                clone['_replica'] = i + 1
            weighted.append(clone)
    new_counts = Counter(d['task'] for d in weighted)

    summary = {
        'original_counts': orig_counts,
        'weights': WEIGHTS,
        'replication_strategy': 'ceil(weight * BASE_MULTIPLIER)',
        'base_multiplier': BASE_MULTIPLIER,
        'new_counts': new_counts,
        'total_original': len(data),
        'total_weighted': len(weighted)
    }
    # convert Counters to normal dict
    summary = json.loads(json.dumps(summary, default=lambda o: dict(o)))
    write_jsonl(WEIGHTED_PATH, weighted)
    Path(SUMMARY_PATH).write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(f"Weighted dataset written: {WEIGHTED_PATH.name} (orig={len(data)} -> weighted={len(weighted)})")
    for t in sorted(orig_counts.keys()):
        print(f"  {t}: {orig_counts[t]} -> {new_counts[t]}")


if __name__ == '__main__':
    main()
