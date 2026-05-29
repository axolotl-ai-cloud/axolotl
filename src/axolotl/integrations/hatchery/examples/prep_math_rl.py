# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Prepare hendrycks_math for RL training with Hatchery/Tinker.

Creates a dataset with chat-formatted prompts that include
a hidden gold answer tag for the reward function.

Run:
  python src/axolotl/integrations/hatchery/examples/prep_math_rl.py
"""

import os
import re

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer


def extract_boxed(text: str) -> str:
    match = re.search(r"\\boxed\{", text)
    if not match:
        return ""
    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1] if depth == 0 else ""


def main():
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)

    ds = load_dataset("EleutherAI/hendrycks_math", "algebra", split="test")
    level = os.environ.get("MATH_LEVEL", "Level 1")
    filtered_rows = [x for x in ds if x["level"] == level]
    print(f"{level} algebra: {len(filtered_rows)} problems")

    rows = []
    for prob in filtered_rows:
        gold = extract_boxed(prob["solution"])
        if not gold:
            continue

        # Format as chat prompt with hidden gold tag
        prompt = (
            f"Solve the following math problem. "
            f"Show your work and put your final answer in \\boxed{{}}.\n\n"
            f"{prob['problem']}"
            f"<|gold|>{gold}<|/gold|>"
        )

        # Tokenize the prompt
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = tokenizer.encode(text, add_special_tokens=False)

        rows.append(
            {
                "input_ids": prompt_ids,
                "labels": [-100] * len(prompt_ids),
                "attention_mask": [1] * len(prompt_ids),
            }
        )

    out = Dataset.from_list(rows)
    out_dir = f"./data/math_rl_{level.lower().replace(' ', '')}"
    out.save_to_disk(out_dir)
    print(f"Saved {len(out)} examples to {out_dir}")
    if rows:
        print(
            f"Prompt length range: {min(len(r['input_ids']) for r in rows)}"
            f"-{max(len(r['input_ids']) for r in rows)}"
        )


if __name__ == "__main__":
    main()
