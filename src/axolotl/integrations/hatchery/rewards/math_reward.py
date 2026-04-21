# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Axolotl AI
# Licensed under the Apache License, Version 2.0

"""Math reward function for hendrycks_math GRPO training.

Uses math_verify for robust answer comparison. Falls back to
exact string match of \\boxed{} content.
"""

from __future__ import annotations

import re


def extract_boxed(text: str) -> str | None:
    """Extract \\boxed{...} answer handling nested braces."""
    match = re.search(r"\\boxed\{", text)
    if not match:
        return None
    start = match.end()
    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return text[start : i - 1] if depth == 0 else None


def math_reward(prompts: list[str], completions: list[str], **kwargs) -> list[float]:
    """Score completions by checking if \\boxed{} answer matches the gold answer.

    The gold answer is extracted from the prompt (appended as a hidden
    tag by the dataset preprocessing). Format:
      ... <|gold|>ANSWER<|/gold|>
    """
    rewards = []
    for prompt, completion in zip(prompts, completions, strict=True):
        gold_match = re.search(r"<\|gold\|>(.*?)<\|/gold\|>", prompt)
        if not gold_match:
            rewards.append(0.0)
            continue

        gold_answer = gold_match.group(1).strip()
        pred_answer = extract_boxed(completion)

        if pred_answer is None:
            rewards.append(0.0)
            continue

        try:
            from math_verify import parse, verify

            gold_parsed = parse(gold_answer)
            pred_parsed = parse(pred_answer)
            if verify(gold_parsed, pred_parsed):
                rewards.append(1.0)
                continue
        except Exception:
            pass

        if pred_answer.strip() == gold_answer.strip():
            rewards.append(1.0)
        else:
            rewards.append(0.0)

    return rewards
