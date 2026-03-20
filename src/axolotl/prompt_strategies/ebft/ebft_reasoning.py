"""
Dataset transform for reasoning/thinking datasets with EBFT.

Handles datasets where assistant responses contain <think>...</think> reasoning
traces (e.g., TeichAI/Claude-Opus-4.6-Reasoning, Qwen3.5 thinking mode outputs).

Two variants:

1. `transform` — For structured EBFT (vLLM mode):
   Returns prompt + ground_truth with thinking tags preserved.
   Feature matching compares full responses (thinking + answer).

2. `transform_answer_only` — For structured EBFT (vLLM mode):
   Strips <think>...</think> from ground_truth, so feature matching
   only scores the final answer portion. Use when reasoning chains
   can vary but the answer should match.

3. `transform_strided` — For strided EBFT:
   Tokenizes the full conversation with thinking traces.
   Optionally masks thinking tokens from CE loss (labels=-100 for think spans)
   while still placing anchors in thinking regions for feature matching.

All variants work with OpenAI chat format:
  {"messages": [{"role": "...", "content": "<think>...</think>Answer"}]}
"""

import re


def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()


def _extract_thinking(text: str) -> tuple[str, str]:
    """Split text into (thinking, answer) parts."""
    match = re.search(r"<think>(.*?)</think>\s*(.*)", text, flags=re.DOTALL)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", text.strip()


def transform(cfg, **kwargs):
    """Full response including thinking traces for feature matching."""

    def transform_fn(example, tokenizer=None):
        messages = example.get("messages", example.get("conversations", []))

        # Find last assistant turn
        prompt_msgs = []
        ground_truth = ""
        for msg in messages:
            if msg["role"] == "assistant":
                prompt_msgs_snapshot = list(prompt_msgs)
                ground_truth = msg["content"]
            prompt_msgs.append(msg)

        return {
            "prompt": prompt_msgs_snapshot if "prompt_msgs_snapshot" in dir() else messages[:-1],
            "ground_truth": ground_truth,
        }

    return transform_fn, {"remove_columns": "__all__"}


def transform_answer_only(cfg, **kwargs):
    """Strip thinking from ground_truth — match features on answer only."""

    def transform_fn(example, tokenizer=None):
        messages = example.get("messages", example.get("conversations", []))

        prompt_msgs = []
        ground_truth = ""
        for msg in messages:
            if msg["role"] == "assistant":
                prompt_msgs_snapshot = list(prompt_msgs)
                ground_truth = _strip_thinking(msg["content"])
            prompt_msgs.append(msg)

        return {
            "prompt": prompt_msgs_snapshot if "prompt_msgs_snapshot" in dir() else messages[:-1],
            "ground_truth": ground_truth,
        }

    return transform_fn, {"remove_columns": "__all__"}


def transform_strided(cfg, **kwargs):
    """For strided EBFT: tokenize with thinking, optionally mask think tokens from CE loss.

    Config options (via cfg):
        - ebft.mask_thinking_ce: bool (default False)
          If True, set labels=-100 for tokens inside <think>...</think> blocks.
          Feature matching still uses these positions (anchors are placed everywhere
          in the completion span). Only CE auxiliary loss is affected.
    """
    seq_len = cfg.sequence_len
    mask_thinking = False
    if cfg.ebft and hasattr(cfg.ebft, "mask_thinking_ce"):
        mask_thinking = cfg.ebft.mask_thinking_ce

    def transform_fn(example, tokenizer=None):
        messages = example.get("messages", example.get("conversations", []))

        if tokenizer is None:
            for m in messages:
                if m.get("role") == "user":
                    return {"prompt": m["content"]}
            return {"prompt": str(messages)}

        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        # Tokenize the full conversation with the chat template
        full_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False,
        )
        full_enc = tokenizer(
            full_text, truncation=True, max_length=seq_len,
            add_special_tokens=False, return_tensors=None,
        )
        input_ids = full_enc["input_ids"]

        # Build labels: -100 for non-assistant tokens
        labels = [-100] * len(input_ids)

        # Find assistant turn boundaries using incremental tokenization
        prefix_messages = []
        for msg in messages:
            if msg["role"] == "assistant":
                prefix_text = tokenizer.apply_chat_template(
                    prefix_messages, tokenize=False, add_generation_prompt=True,
                )
                prefix_ids = tokenizer(
                    prefix_text, truncation=True, max_length=seq_len,
                    add_special_tokens=False, return_tensors=None,
                )["input_ids"]
                start = len(prefix_ids)

                prefix_messages.append(msg)
                with_turn_text = tokenizer.apply_chat_template(
                    prefix_messages, tokenize=False, add_generation_prompt=False,
                )
                with_turn_ids = tokenizer(
                    with_turn_text, truncation=True, max_length=seq_len,
                    add_special_tokens=False, return_tensors=None,
                )["input_ids"]
                end = len(with_turn_ids)

                # Mark assistant tokens as trainable
                for i in range(start, min(end, len(labels))):
                    labels[i] = input_ids[i]

                # Optionally mask <think>...</think> tokens within this turn.
                # Find think spans by scanning for <think> and </think> token IDs
                # directly in the input_ids (robust to tokenization alignment).
                if mask_thinking:
                    think_open_id = tokenizer.convert_tokens_to_ids("<think>")
                    think_close_id = tokenizer.convert_tokens_to_ids("</think>")
                    if think_open_id != tokenizer.unk_token_id:
                        # Scan from before the assistant turn start to catch
                        # <think> tags that are part of the template prefix
                        scan_start = max(0, start - 5)
                        in_think = False
                        for i in range(scan_start, min(end, len(labels))):
                            if input_ids[i] == think_open_id:
                                in_think = True
                            if in_think and i >= start:
                                labels[i] = -100
                            if input_ids[i] == think_close_id:
                                in_think = False
                                if i >= start:
                                    labels[i] = -100
            else:
                prefix_messages.append(msg)

        # Derive prompt_length
        prompt_length = len(input_ids)
        for i, lbl in enumerate(labels):
            if lbl != -100:
                prompt_length = i
                break

        # Pad
        pad_len = seq_len - len(input_ids)
        attention_mask = [1] * len(input_ids) + [0] * pad_len
        labels = labels + [-100] * pad_len
        input_ids = input_ids + [pad_id] * pad_len

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_length": prompt_length,
        }

    return transform_fn, {"remove_columns": "__all__"}
