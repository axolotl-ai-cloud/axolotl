#!/usr/bin/env python
"""Minimal smoke test training loop for weighted prompted dataset.

This avoids full Axolotl training stack; just verifies that we can:
1. Instantiate WeightedPromptedIterableDataset
2. Tokenize messages (simple join) for causal LM
3. Perform a few optimization steps and see loss decrease

Usage:
  python scripts/smoke_weighted_training.py \
    --data data/bethpage_black/training_bethpage_multitask.weighted.prompted.dedup.train.jsonl \
    --model distilgpt2 --steps 20 --epoch-size 256 --lr 5e-5

Note: For real training use the full Axolotl pipeline.
"""
from __future__ import annotations
import argparse
import math
import os
import sys
from typing import List

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local import (add src to path if needed)
if 'axolotl' not in sys.modules:
    sys.path.insert(0, os.path.abspath('src'))
from axolotl.data.weighted_prompted_dataset import WeightedPromptedIterableDataset  # type: ignore

SYSTEM_PREFIX = "System: "
USER_PREFIX = "User: "
ASSIST_PREFIX = "Assistant: "
SEP = "\n"

def format_messages(messages):
    parts: List[str] = []
    for msg in messages:
        role = msg.get('role')
        if role == 'system':
            parts.append(SYSTEM_PREFIX + msg.get('content','').strip())
        elif role == 'user':
            parts.append(USER_PREFIX + msg.get('content','').strip())
        elif role == 'assistant':
            parts.append(ASSIST_PREFIX + msg.get('content','').strip())
    return SEP.join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--manifest', help='Optional manifest file to verify before training')
    ap.add_argument('--model', default='distilgpt2')
    ap.add_argument('--steps', type=int, default=30)
    ap.add_argument('--epoch-size', type=int, default=512)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--lr', type=float, default=5e-5)
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--quota', action='store_true')
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    if args.manifest:
        try:
            from axolotl.data.weighted_prompted_dataset import verify_prompted_manifest  # type: ignore
            verify_prompted_manifest(args.data, args.manifest)
            print("[verify] Manifest OK")
        except Exception as e:  # pylint: disable=broad-except
            print(f"[verify] Manifest verification FAILED: {e}")
            return

    ds = WeightedPromptedIterableDataset(
        path=args.data,
        epoch_size=args.epoch_size,
        temperature=args.temperature,
        seed=args.seed,
        enforce_quota=args.quota,
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(args.device)
    model.train()

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    losses = []
    step_iter = iter(ds)
    for step in range(1, args.steps+1):
        try:
            rec = next(step_iter)
        except StopIteration:
            # new epoch
            ds.set_epoch(ds._epoch + 1)
            step_iter = iter(ds)
            rec = next(step_iter)
        text = format_messages(rec['messages'])
        enc = tok(text, return_tensors='pt')
        input_ids = enc['input_ids'].to(args.device)
        attn = enc['attention_mask'].to(args.device)
        outputs = model(input_ids=input_ids, attention_mask=attn, labels=input_ids)
        loss = outputs.loss
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)
        losses.append(loss.item())
        if step % 5 == 0:
            avg_last5 = sum(losses[-5:]) / min(5, len(losses))
            print(f"Step {step} loss={loss.item():.4f} avg_last5={avg_last5:.4f}")

    if len(losses) >= 5:
        trend = 'decrease' if losses[-1] < losses[0] else 'flat_or_increase'
    else:
        trend = 'n/a'
    print('Initial loss:', f"{losses[0]:.4f}")
    print('Final loss:', f"{losses[-1]:.4f}")
    print('Trend:', trend)

if __name__ == '__main__':
    main()
