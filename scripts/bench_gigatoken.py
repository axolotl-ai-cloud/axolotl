"""Benchmark gigatoken vs the HF tokenizer on the pretraining encode hot path.

Mirrors the exact call in axolotl.utils.data.streaming.encode_streaming so the
numbers reflect real preprocessing throughput, not a synthetic micro-benchmark.

Compares three configurations:
  hf-1proc        HF tokenizer, single process (baseline for the STREAMING
                  pretraining path, whose .map has no num_proc).
  gigatoken       gigatoken .as_hf(), single process (multithreaded internally).
  hf-Nproc        HF tokenizer via datasets.map(num_proc=N) (baseline for the
                  NON-streaming completion path).

Usage:
    pip install gigatoken
    python scripts/bench_gigatoken.py --model gpt2 --corpus data.txt --num-proc 8
    # optional: stream the summary to wandb
    python scripts/bench_gigatoken.py --model gpt2 --corpus data.txt --wandb-project gigatoken-bench
"""

import argparse
import time

from transformers import AutoTokenizer


def load_texts(path, limit):
    with open(path, encoding="utf-8") as f:
        texts = [line for line in (l.strip() for l in f) if line]
    return texts[:limit] if limit else texts


def encode_all(callable_tok, texts, max_tokens, batch_size):
    """Replays encode_streaming's exact tokenizer call; returns total token count."""
    n = 0
    for i in range(0, len(texts), batch_size):
        res = callable_tok(
            texts[i : i + batch_size],
            truncation=True,
            max_length=max_tokens - 2,
            add_special_tokens=True,
        )
        n += sum(len(ids) for ids in res["input_ids"])
    return n


def bench_hf_map(model, texts, max_tokens, num_proc, batch_size):
    """HF tokenizer through datasets.map(num_proc=N) — the real multiprocess baseline."""
    from datasets import Dataset

    ds = Dataset.from_dict({"text": texts})
    tok = AutoTokenizer.from_pretrained(model)

    def _enc(batch):
        return tok(
            batch["text"],
            truncation=True,
            max_length=max_tokens - 2,
            add_special_tokens=True,
        )

    t0 = time.perf_counter()
    out = ds.map(
        _enc,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        remove_columns=["text"],
    )
    dt = time.perf_counter() - t0
    n = sum(len(x) for x in out["input_ids"])
    return dt, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument(
        "--num-proc", type=int, default=8, help="workers for the hf-Nproc baseline"
    )
    ap.add_argument(
        "--limit", type=int, default=0, help="cap number of lines (0 = all)"
    )
    ap.add_argument("--wandb-project", default=None)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)
    texts = load_texts(args.corpus, args.limit)
    print(f"loaded {len(texts):,} lines from {args.corpus}\n")

    import gigatoken as gt

    fast = gt.Tokenizer(tok).as_hf()

    results = {}

    for name, callable_tok in [("hf-1proc", tok), ("gigatoken", fast)]:
        t0 = time.perf_counter()
        n = encode_all(callable_tok, texts, args.max_tokens, args.batch_size)
        dt = time.perf_counter() - t0
        results[name] = n / dt / 1e6
        print(f"{name:>12}: {dt:7.2f}s  {n / dt / 1e6:7.2f} Mtok/s  ({n:,} tokens)")

    dt, n = bench_hf_map(
        args.model, texts, args.max_tokens, args.num_proc, args.batch_size
    )
    name = f"hf-{args.num_proc}proc"
    results[name] = n / dt / 1e6
    print(f"{name:>12}: {dt:7.2f}s  {n / dt / 1e6:7.2f} Mtok/s  ({n:,} tokens)")

    speedup_1p = results["gigatoken"] / results["hf-1proc"]
    speedup_np = results["gigatoken"] / results[name]
    print(f"\ngigatoken vs hf-1proc:  {speedup_1p:.1f}x")
    print(f"gigatoken vs {name}: {speedup_np:.1f}x")

    if args.wandb_project:
        import wandb

        run = wandb.init(
            project=args.wandb_project, name="tokenization-bench", config=vars(args)
        )
        run.summary.update({f"mtok_s/{k}": v for k, v in results.items()})
        run.summary.update(
            {"speedup_vs_1proc": speedup_1p, f"speedup_vs_{name}": speedup_np}
        )
        run.finish()


if __name__ == "__main__":
    main()
