"""Lightweight preprocessing adapter for legacy `train_lora_bethpage_strat.py`.

Current dataset (`train_multitask_case_fixed.jsonl`) already matches what the
training script expects: each JSON line has a `prompt` and a `completion` plus
metadata fields. This script exists so future changes (e.g. switching to a
`messages` list, adding weighting, or introducing multi-file merges) can be
centralized without touching the training script again.

Usage (PowerShell):
  ./axo-env/Scripts/python.exe scripts/prepare_bethpage_dataset.py \
      --in data/bethpage_black/train_multitask_case_fixed.jsonl \
      --out data/bethpage_black/train_multitask_case_fixed.prepared.jsonl

If no transformation is needed the script copies (streams) the lines while:
    * Optionally filtering out lines where `use_for_training` is false
    * Optionally downsampling for a quick debug subset
    * Optionally carving out a validation subset (random or stratified by task_type)

You can extend `_transform(record)` to rewrite structure (e.g. build prompt
from messages) later.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from typing import Iterable, Dict, Any


def stream_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for ln, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:  # pragma: no cover - defensive
                print(f"Skipping malformed line {ln}: {e}", file=sys.stderr)


def _transform(rec: Dict[str, Any]) -> Dict[str, Any]:
    """Rewrite hook.

    Currently pass-through: dataset already has `prompt` and `completion`.
    Modify here if future schema changes (e.g. messages -> prompt/completion).
    """
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, help="Source JSONL file")
    ap.add_argument("--out", dest="dst", required=True, help="Destination JSONL file")
    ap.add_argument("--keep-all", action="store_true", help="Keep records even if use_for_training is False")
    ap.add_argument("--sample", type=int, default=None, help="Optional random sample size (after filtering)")
    ap.add_argument("--seed", type=int, default=42, help="RNG seed for sampling and validation carve-out")
    ap.add_argument("--val-out", type=str, default=None, help="Optional validation JSONL output path to carve out examples")
    ap.add_argument("--val-size", type=int, default=0, help="Number of examples to reserve for validation")
    ap.add_argument("--val-stratify-task", action="store_true", help="Stratify validation selection across task_type values")
    ap.add_argument("--val-min-per-task", type=int, default=1, help="Minimum per task_type when stratifying (capped by availability)")
    # New: test split & advanced stratification / audit
    ap.add_argument("--test-out", type=str, default=None, help="Optional test JSONL output path to carve out examples (after val)")
    ap.add_argument("--test-size", type=int, default=0, help="Number of examples to reserve for test (applied after validation carve-out)")
    ap.add_argument("--stratify-yard-buckets", action="store_true", help="When stratifying val/test also balance coarse yardage buckets")
    ap.add_argument("--manifest-out", type=str, default=None, help="Optional path to write a dataset manifest JSON with counts & hashes")
    ap.add_argument("--audit-out", type=str, default=None, help="Optional path to write leakage/distribution audit JSON")
    ap.add_argument("--yard-col", type=str, default="expected_cutoff_yards", help="Column containing target yardage")
    ap.add_argument("--bucket-spec", type=str, default="<230,230-260,260-290,>=290", help="Yard bucket spec for stratification/audit")
    ap.add_argument("--leakage-check", action="store_true", help="Enable leakage scan of prompts for target yard numbers")
    args = ap.parse_args()

    rows = []
    for rec in stream_jsonl(args.src):
        if not args.keep_all and rec.get("use_for_training") is False:
            continue
        if "prompt" not in rec or "completion" not in rec:
            # In future: construct from messages if present
            msgs = rec.get("messages")
            if msgs and isinstance(msgs, list):
                # naive fallback: join user messages as prompt, last assistant as completion
                user_parts = [m.get("content", "") for m in msgs if m.get("role") == "user"]
                assistant_parts = [m.get("content", "") for m in msgs if m.get("role") == "assistant"]
                if user_parts and assistant_parts:
                    rec["prompt"] = "\n".join(user_parts)
                    rec["completion"] = assistant_parts[-1]
                else:
                    continue  # skip if we can't map
        rows.append(_transform(rec))

    if args.sample is not None and args.sample < len(rows):
        random.seed(args.seed)
        rows = random.sample(rows, args.sample)

    # Helper: compute yard bucket
    def parse_bucket_spec(spec: str):
        parts = [p.strip() for p in spec.split(',') if p.strip()]
        buckets = []
        for p in parts:
            if p.startswith('<'):
                try:
                    v = int(p[1:])
                    buckets.append(('lt', v))
                except ValueError:
                    pass
            elif p.startswith('>='):
                try:
                    v = int(p[2:])
                    buckets.append(('ge', v))
                except ValueError:
                    pass
            elif '-' in p:
                try:
                    a,b = p.split('-',1)
                    buckets.append(('range', int(a), int(b)))
                except ValueError:
                    pass
        return buckets

    bucket_defs = parse_bucket_spec(args.bucket_spec)

    def yard_bucket(val):
        if not isinstance(val, int):
            return 'UNK'
        for b in bucket_defs:
            if b[0] == 'lt' and val < b[1]:
                return f"< {b[1]}"
            if b[0] == 'ge' and val >= b[1]:
                return f">= {b[1]}"
            if b[0] == 'range' and b[1] <= val <= b[2]:
                return f"{b[1]}-{b[2]}"
        return 'OTHER'

    for r in rows:
        if args.stratify_yard_buckets:
            r['_yard_bucket'] = yard_bucket(r.get(args.yard_col))

    # Validation carve-out
    val_rows = []
    if args.val_out and args.val_size > 0 and len(rows) > args.val_size:
        random.seed(args.seed)
        indices = list(range(len(rows)))
        if args.val_stratify_task:
            from collections import defaultdict
            grouped = defaultdict(list)
            for idx, r in enumerate(rows):
                # composite key if yard bucket stratification is enabled
                if args.stratify_yard_buckets:
                    key = f"{r.get('task_type','UNKNOWN')}||{r.get('_yard_bucket','UNK')}"
                else:
                    key = str(r.get("task_type", "UNKNOWN"))
                grouped[key].append(idx)
            tasks = list(grouped.keys())
            allocation = {t: 0 for t in tasks}
            remaining = args.val_size
            # seed with minimum
            for t in tasks:
                take = min(args.val_min_per_task, len(grouped[t]))
                allocation[t] = take
                remaining -= take
                if remaining <= 0:
                    break
            # distribute remaining
            while remaining > 0:
                progressed = False
                for t in tasks:
                    if remaining <= 0:
                        break
                    available = len(grouped[t]) - allocation[t]
                    if available > 0:
                        allocation[t] += 1
                        remaining -= 1
                        progressed = True
                if not progressed:
                    break
            chosen = []
            for t in tasks:
                random.shuffle(grouped[t])
                chosen.extend(grouped[t][: allocation[t]])
            chosen_set = set(chosen)
        else:
            random.shuffle(indices)
            chosen_set = set(indices[: args.val_size])
        new_rows = []
        for idx, r in enumerate(rows):
            if idx in chosen_set:
                val_copy = dict(r)
                val_copy["split"] = "val"
                val_rows.append(val_copy)
            else:
                new_rows.append(r)
        rows = new_rows

    # Test carve-out (from remaining rows)
    test_rows = []
    if args.test_out and args.test_size > 0 and len(rows) > args.test_size:
        random.seed(args.seed + 7)
        indices = list(range(len(rows)))
        if args.val_stratify_task:  # reuse same stratification flag semantics
            from collections import defaultdict
            grouped = defaultdict(list)
            for idx, r in enumerate(rows):
                if args.stratify_yard_buckets:
                    key = f"{r.get('task_type','UNKNOWN')}||{r.get('_yard_bucket','UNK')}"
                else:
                    key = str(r.get("task_type", "UNKNOWN"))
                grouped[key].append(idx)
            tasks = list(grouped.keys())
            allocation = {t: 0 for t in tasks}
            remaining = args.test_size
            while remaining > 0:
                progressed = False
                for t in tasks:
                    if remaining <= 0:
                        break
                    available = len(grouped[t]) - allocation[t]
                    if available > 0:
                        allocation[t] += 1
                        remaining -= 1
                        progressed = True
                if not progressed:
                    break
            chosen = []
            for t in tasks:
                random.shuffle(grouped[t])
                chosen.extend(grouped[t][: allocation[t]])
            chosen_set = set(chosen)
        else:
            random.shuffle(indices)
            chosen_set = set(indices[: args.test_size])
        new_rows = []
        for idx, r in enumerate(rows):
            if idx in chosen_set:
                t_copy = dict(r)
                t_copy['split'] = 'test'
                test_rows.append(t_copy)
            else:
                new_rows.append(r)
        rows = new_rows

    with open(args.dst, "w", encoding="utf-8") as out:
        for r in rows:
            out.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} train records -> {args.dst}")
    if val_rows and args.val_out:
        with open(args.val_out, "w", encoding="utf-8") as vout:
            for r in val_rows:
                vout.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(val_rows)} validation records -> {args.val_out}")
    if test_rows and args.test_out:
        with open(args.test_out, "w", encoding="utf-8") as tout:
            for r in test_rows:
                tout.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"Wrote {len(test_rows)} test records -> {args.test_out}")

    # Leakage & distribution audit
    if args.audit_out:
        import hashlib, math
        total = len(rows) + len(val_rows) + len(test_rows)
        # Build prompt leakage stats if enabled
        leakage_hits = 0
        leakage_any_strategy = 0
        total_checked = 0
        if args.leakage_check:
            for subset in (rows, val_rows, test_rows):
                for rec in subset:
                    prompt = str(rec.get('prompt',''))
                    tgt = rec.get(args.yard_col)
                    found_direct = False
                    found_any = False
                    if isinstance(tgt, int) and str(tgt) in prompt:
                        leakage_hits += 1
                        found_direct = True
                    # any strategy cutoff
                    strategies = rec.get('tee_shot_strategies')
                    if isinstance(strategies, list):
                        for s in strategies:
                            c = s.get('cutoff_distance') if isinstance(s, dict) else None
                            if isinstance(c, int) and str(c) in prompt:
                                leakage_any_strategy += 1
                                found_any = True
                                break
                    total_checked += 1
        # Distribution counts
        from collections import Counter
        def count_field(subset, field):
            return Counter(str(r.get(field,'UNK')) for r in subset)
        def count_bucket(subset):
            return Counter(str(r.get('_yard_bucket','UNK')) for r in subset)
        audit = {
            'total_examples': total,
            'train_count': len(rows),
            'val_count': len(val_rows),
            'test_count': len(test_rows),
            'task_type_train': count_field(rows,'task_type'),
            'task_type_val': count_field(val_rows,'task_type'),
            'task_type_test': count_field(test_rows,'task_type'),
            'yard_bucket_train': count_bucket(rows),
            'yard_bucket_val': count_bucket(val_rows),
            'yard_bucket_test': count_bucket(test_rows),
            'leakage_direct_count': leakage_hits if args.leakage_check else None,
            'leakage_any_strategy_count': leakage_any_strategy if args.leakage_check else None,
            'leakage_total_checked': total_checked if args.leakage_check else None,
        }
        # Add simple hashes for splits for reproducibility
        def hash_file(path):
            if not path:
                return None
            h = hashlib.sha256()
            try:
                with open(path,'rb') as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk: break
                        h.update(chunk)
                return h.hexdigest()[:16]
            except Exception:
                return None
        audit['hash_train'] = hash_file(args.dst)
        audit['hash_val'] = hash_file(args.val_out)
        audit['hash_test'] = hash_file(args.test_out)
        with open(args.audit_out,'w',encoding='utf-8') as af:
            json.dump(audit, af, indent=2, ensure_ascii=False)
        print(f"Wrote audit -> {args.audit_out}")

    # Manifest JSON (lightweight)
    if args.manifest_out:
        manifest = {
            'source': args.src,
            'train_file': args.dst,
            'val_file': args.val_out,
            'test_file': args.test_out,
            'seed': args.seed,
            'val_size': len(val_rows),
            'test_size': len(test_rows),
            'stratify_task': args.val_stratify_task,
            'stratify_yard_buckets': args.stratify_yard_buckets,
            'bucket_spec': args.bucket_spec,
            'yard_col': args.yard_col,
        }
        with open(args.manifest_out,'w',encoding='utf-8') as mf:
            json.dump(manifest, mf, indent=2, ensure_ascii=False)
        print(f"Wrote manifest -> {args.manifest_out}")


if __name__ == "__main__":  # pragma: no cover
    main()
