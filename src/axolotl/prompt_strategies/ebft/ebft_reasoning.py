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
    """Full response including thinking traces for feature matching.

    For datasets where assistant content has <think>...</think> tags in the
    content field. The ground_truth includes the full content (thinking + answer).
    """

    def transform_fn(example, tokenizer=None):
        messages = example.get("messages", example.get("conversations", []))

        prompt_msgs_snapshot = None
        ground_truth = ""
        for msg_idx, msg in enumerate(messages):
            if msg["role"] == "assistant":
                prompt_msgs_snapshot = list(messages[:msg_idx])
                ground_truth = msg["content"]

        return {
            "prompt": prompt_msgs_snapshot
            if prompt_msgs_snapshot is not None
            else messages[:-1],
            "ground_truth": ground_truth,
        }

    return transform_fn, {"remove_columns": "__all__"}


def transform_split_thinking(cfg, **kwargs):
    """Split <think> tags into reasoning_content field for native chat template handling.

    For datasets where thinking is embedded in the content field as <think>...</think>.
    Splits it into separate reasoning_content and content fields so the model's
    chat template can format it natively (e.g., Qwen3.5's reasoning_content support).

    The prompt messages are passed through with reasoning_content properly split,
    so vLLM generation with enable_thinking=true produces comparable outputs.
    The ground_truth is the full assistant response (thinking + answer) for
    feature matching.

    Also works for:
    - <reasoning>...</reasoning> tags
    - <|begin_of_thought|>...<|end_of_thought|> tags
    """
    _THINKING_PAIRS = [
        ("<think>", "</think>"),
        ("<reasoning>", "</reasoning>"),
        ("<|begin_of_thought|>", "<|end_of_thought|>"),
    ]

    def _split_msg_thinking(msg):
        """Split thinking from assistant message content into reasoning_content.

        Always includes reasoning_content key on assistant messages (empty string
        if no thinking tags found) to ensure consistent HF dataset schema across
        all examples in a batch.
        """
        if msg["role"] != "assistant":
            return msg
        content = msg.get("content", "")
        # Already has reasoning_content — pass through
        if "reasoning_content" in msg:
            return msg
        for open_tag, close_tag in _THINKING_PAIRS:
            if open_tag in content and close_tag in content:
                start = content.find(open_tag)
                end = content.find(close_tag)
                thinking = content[start + len(open_tag) : end].strip()
                answer = content[end + len(close_tag) :].strip()
                return {
                    **msg,
                    "reasoning_content": thinking,
                    "content": answer,
                }
        # No thinking tags — still add reasoning_content for schema consistency
        return {**msg, "reasoning_content": ""}

    def _normalize_msg(msg):
        """Ensure every message has {role, content, reasoning_content} for HF schema consistency."""
        return {
            "role": msg.get("role", ""),
            "content": msg.get("content", ""),
            "reasoning_content": msg.get("reasoning_content", ""),
        }

    def transform_fn(example, tokenizer=None):
        messages = example.get("messages", example.get("conversations", []))

        # Split thinking in all assistant messages, then normalize schema
        split_messages = [_normalize_msg(_split_msg_thinking(m)) for m in messages]

        # Build prompt (all messages except last assistant) and ground_truth
        prompt_msgs = []
        prompt_msgs_snapshot = None
        ground_truth = ""
        for msg in split_messages:
            if msg["role"] == "assistant":
                prompt_msgs_snapshot = list(prompt_msgs)
                # ground_truth is the FULL content for feature matching
                thinking = msg.get("reasoning_content", "")
                answer = msg.get("content", "")
                if thinking:
                    ground_truth = f"<think>\n{thinking}\n</think>\n\n{answer}"
                else:
                    ground_truth = answer
            prompt_msgs.append(msg)

        return {
            "prompt": prompt_msgs_snapshot
            if prompt_msgs_snapshot is not None
            else split_messages[:-1],
            "ground_truth": ground_truth,
        }

    return transform_fn, {"remove_columns": "__all__"}


def transform_answer_only(cfg, **kwargs):
    """Strip thinking from ground_truth — match features on answer only."""

    def transform_fn(example, tokenizer=None):
        messages = example.get("messages", example.get("conversations", []))

        prompt_msgs = []
        prompt_msgs_snapshot = None
        ground_truth = ""
        for msg in messages:
            if msg["role"] == "assistant":
                prompt_msgs_snapshot = list(prompt_msgs)
                ground_truth = _strip_thinking(msg["content"])
            prompt_msgs.append(msg)

        return {
            "prompt": prompt_msgs_snapshot
            if prompt_msgs_snapshot is not None
            else messages[:-1],
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

        pad_id = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

        # Tokenize the full conversation with the chat template
        full_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_enc = tokenizer(
            full_text,
            truncation=True,
            max_length=seq_len,
            add_special_tokens=False,
            return_tensors=None,
        )
        input_ids = full_enc["input_ids"]

        # Build labels: -100 for non-assistant tokens
        labels = [-100] * len(input_ids)

        # Find assistant turn boundaries using incremental tokenization.
        # Only the FINAL assistant turn is marked as trainable.
        prefix_messages = []
        final_start = None
        final_end = None
        for msg in messages:
            if msg["role"] == "assistant":
                prefix_text = tokenizer.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                prefix_ids = tokenizer(
                    prefix_text,
                    truncation=True,
                    max_length=seq_len,
                    add_special_tokens=False,
                    return_tensors=None,
                )["input_ids"]
                start = len(prefix_ids)

                prefix_messages.append(msg)
                with_turn_text = tokenizer.apply_chat_template(
                    prefix_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
                with_turn_ids = tokenizer(
                    with_turn_text,
                    truncation=True,
                    max_length=seq_len,
                    add_special_tokens=False,
                    return_tensors=None,
                )["input_ids"]
                end = len(with_turn_ids)

                # Record this turn's boundaries; only the last one will be used
                final_start = start
                final_end = end
            else:
                prefix_messages.append(msg)

        # Mark only the final assistant turn as trainable
        if final_start is not None and final_end is not None:
            for i in range(final_start, min(final_end, len(labels))):
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
                    scan_start = max(0, final_start - 5)
                    in_think = False
                    for i in range(scan_start, min(final_end, len(labels))):
                        if input_ids[i] == think_open_id:
                            in_think = True
                        if in_think and i >= final_start:
                            labels[i] = -100
                        if input_ids[i] == think_close_id:
                            in_think = False
                            if i >= final_start:
                                labels[i] = -100

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
