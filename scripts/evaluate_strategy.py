import os
import json
import argparse
import re
from statistics import mean, median
from typing import Optional, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
try:
    from peft import AutoPeftModelForCausalLM  # optional; used when adapter present
except Exception:
    AutoPeftModelForCausalLM = None

# Reuse similar extraction logic as in training script (kept self-contained to avoid import side effects)
YARD_PATTERNS = [
    r"(\d{2,4})\s*-?\s*yard",  # generic yard mention
    r"Strategy:\s*(\d{2,4})",   # prefixed strategy pattern
]

def extract_first_yard_number(text: str) -> Optional[int]:
    for pat in YARD_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            try:
                num = int(m.group(1))
                # Basic sanity filter
                if 30 <= num <= 600:
                    return num
            except Exception:
                pass
    return None

def load_jsonl(path: str, max_samples: Optional[int] = None):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
            if max_samples is not None and (i + 1) >= max_samples:
                break

def main():
    parser = argparse.ArgumentParser(description="Evaluate strategy yard predictions via generation.")
    parser.add_argument("--ckpt-dir", required=True, help="Checkpoint directory with adapter / model weights.")
    parser.add_argument("--data-file", required=True, help="Validation JSONL file.")
    parser.add_argument("--out-file", default=None, help="Optional JSONL to write per-sample predictions.")
    parser.add_argument("--yard-col", type=str, default="expected_cutoff_yards", help="Target yard column name.")
    parser.add_argument("--bucket-spec", type=str, default="<230,230-260,260-290,>=290", help="Bucket spec for yard strat metrics.")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit number of validation samples.")
    parser.add_argument("--max-new", type=int, default=24, help="Max new tokens to generate.")
    parser.add_argument("--temperature", type=float, default=0.1, help="Generation temperature.")
    parser.add_argument("--top-p", type=float, default=0.95, help="Top-p nucleus sampling.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--device", type=str, default=None, help="Force device (cpu / cuda). Default: auto.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    # Auto device selection
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load tokenizer/model (supports LoRA adapters saved in ckpt)
    tokenizer = AutoTokenizer.from_pretrained(args.ckpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Try PEFT-aware loader first if adapter present
    adapter_cfg = os.path.join(args.ckpt_dir, "adapter_config.json")
    if AutoPeftModelForCausalLM and os.path.isfile(adapter_cfg):
        model = AutoPeftModelForCausalLM.from_pretrained(args.ckpt_dir)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.ckpt_dir)
    model.to(device)
    model.eval()

    predictions = []

    # Prepare bucket parser
    def parse_bucket_spec(spec: str):
        parts = [p.strip() for p in spec.split(',') if p.strip()]
        out = []
        for p in parts:
            if p.startswith('<'):
                try: out.append(('lt', int(p[1:])))
                except: pass
            elif p.startswith('>='):
                try: out.append(('ge', int(p[2:])))
                except: pass
            elif '-' in p:
                try:
                    a,b = p.split('-',1)
                    out.append(('range', int(a), int(b)))
                except: pass
        return out
    bucket_defs = parse_bucket_spec(args.bucket_spec)
    def bucketize(val):
        if not isinstance(val, int):
            return 'UNK'
        for b in bucket_defs:
            if b[0]=='lt' and val < b[1]: return f"< {b[1]}"
            if b[0]=='ge' and val >= b[1]: return f">= {b[1]}"
            if b[0]=='range' and b[1] <= val <= b[2]: return f"{b[1]}-{b[2]}"
        return 'OTHER'

    for ex in load_jsonl(args.data_file, args.max_samples):
        prompt = ex.get("prompt", "")
        # Mirror training formatting (no extra suffix used there)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen_kwargs = dict(max_new_tokens=args.max_new, pad_token_id=tokenizer.eos_token_id)
            if args.temperature and args.temperature > 0:
                gen_kwargs.update(dict(do_sample=True, temperature=args.temperature, top_p=args.top_p))
            else:
                gen_kwargs.update(dict(do_sample=False))
            gen_ids = model.generate(**inputs, **gen_kwargs)
        full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        # Extract only the newly generated tail beyond prompt tokens
        prompt_len = len(inputs["input_ids"][0])
        gen_tail_ids = gen_ids[0][prompt_len:]
        generated = tokenizer.decode(gen_tail_ids, skip_special_tokens=True)

        pred_yards = extract_first_yard_number(generated)
        expected = ex.get("expected_cutoff_yards")
        record: Dict[str, Any] = {
            "prompt": prompt,
            "generated": generated.strip(),
            "full_text": full_text.strip(),
            "predicted_yards": pred_yards,
            "expected_cutoff_yards": expected,
            "task_type": ex.get("task_type"),
        }
        if isinstance(expected, int) and isinstance(pred_yards, int):
            record["abs_error"] = abs(pred_yards - expected)
        record['yard_bucket'] = bucketize(expected)
        predictions.append(record)

    # Aggregate metrics
    abs_errors = [r["abs_error"] for r in predictions if "abs_error" in r]
    exact_matches = [r for r in predictions if r.get("abs_error") == 0]
    within_5 = [r for r in predictions if isinstance(r.get("abs_error"), int) and r["abs_error"] <= 5]
    within_10 = [r for r in predictions if isinstance(r.get("abs_error"), int) and r["abs_error"] <= 10]

    metrics = {}
    def summarize(records):
        ae = [r['abs_error'] for r in records if 'abs_error' in r]
        if not ae:
            return None
        em = sum(1 for r in records if r.get('abs_error') == 0)
        w5 = sum(1 for r in records if isinstance(r.get('abs_error'), int) and r['abs_error'] <= 5)
        w10 = sum(1 for r in records if isinstance(r.get('abs_error'), int) and r['abs_error'] <= 10)
        return {
            'count': len(ae),
            'mean_abs_error': round(mean(ae),3),
            'median_abs_error': round(median(ae),3),
            'exact_match_rate': round(em/len(ae),4),
            'within_5_rate': round(w5/len(ae),4),
            'within_10_rate': round(w10/len(ae),4),
        }

    if abs_errors:
        metrics = summarize(predictions)
        # Slice by task_type
        from collections import defaultdict
        by_task = defaultdict(list)
        by_bucket = defaultdict(list)
        for r in predictions:
            by_task[str(r.get('task_type'))].append(r)
            by_bucket[str(r.get('yard_bucket'))].append(r)
        metrics['by_task_type'] = {k: summarize(v) for k,v in by_task.items()}
        metrics['by_yard_bucket'] = {k: summarize(v) for k,v in by_bucket.items()}

    # Output handling
    if args.out_file:
        os.makedirs(os.path.dirname(args.out_file), exist_ok=True)
        with open(args.out_file, "w", encoding="utf-8") as f:
            for r in predictions:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # Also dump metrics summary
        with open(args.out_file + ".metrics.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    print("Evaluation summary:")
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {len(predictions)} predictions" + (f" to {args.out_file}" if args.out_file else ""))

if __name__ == "__main__":
    main()
