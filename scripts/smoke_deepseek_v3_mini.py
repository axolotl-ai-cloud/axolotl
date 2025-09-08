#!/usr/bin/env python
"""Smoke test for Deepseek-V3-mini checkpoints saved by make_deepseek_v3_mini.py.

- Verifies tied weights between `lm_head` and `embed_tokens`
- Runs a dummy forward pass and prints logits shape and loss
- Optionally runs a short generation if a compatible tokenizer is provided

Usage examples:
  python scripts/smoke_deepseek_v3_mini.py --model_dir ./deepseek-v3-mini-1.3b
  python scripts/smoke_deepseek_v3_mini.py --model_dir ./deepseek-v3-mini-1.3b \
      --tokenizer_dir /path/to/32k-tokenizer --prompt "Write a haiku about oceans."
"""

import argparse
import sys

import torch
from transformers import AutoTokenizer

from axolotl.models.deepseek_v3_mini import DeepseekV3MiniForCausalLM


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True, help="Path to saved model directory")
    ap.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run the smoke test on",
    )
    ap.add_argument("--tokenizer_dir", help="Optional path to a compatible tokenizer")
    ap.add_argument("--prompt", default="Hello", help="Prompt for optional generate")
    ap.add_argument("--max_new_tokens", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    return ap.parse_args()


def pick_device(arg: str) -> torch.device:
    if arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(arg)


def main() -> int:
    args = parse_args()
    device = pick_device(args.device)

    print(f"Loading model from: {args.model_dir}")
    model = DeepseekV3MiniForCausalLM.from_pretrained(
        args.model_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=False,
    )
    model.to(device)
    model.eval()

    # Check tied weights
    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    print(f"Tied weights (lm_head <-> embed_tokens): {tied}")

    # Dummy forward pass
    B, T = 2, 16
    vocab = model.config.vocab_size
    x = torch.randint(0, vocab, (B, T), device=device)
    with torch.no_grad():
        out = model(input_ids=x, labels=x)
    logits = out["logits"]
    loss = out["loss"].item() if out["loss"] is not None else None
    print(f"Logits shape: {tuple(logits.shape)} | Loss: {loss}")

    # Optional generation
    if args.tokenizer_dir:
        print(f"Loading tokenizer from: {args.tokenizer_dir}")
        tok = AutoTokenizer.from_pretrained(args.tokenizer_dir, use_fast=True)
        if tok.pad_token_id is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        inputs = tok(args.prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        text = tok.decode(gen[0], skip_special_tokens=True)
        print("\n=== Generation ===")
        print(text)

    print("\nSmoke test complete ✅")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:  # pragma: no cover - simple CLI
        print(f"Smoke test failed: {e}", file=sys.stderr)
        raise
