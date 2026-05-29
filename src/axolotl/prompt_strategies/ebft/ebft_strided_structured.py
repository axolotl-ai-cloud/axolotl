"""
Dataset transform for structured (prompt, completion) data with strided EBFT.

Tokenizes prompt and completion separately, concatenates into a single
input_ids sequence, and marks prompt tokens with labels=-100 so the
strided trainer knows where to place anchors (completion span only).

Works with datasets that have chat-style fields (e.g., nvidia/OpenCodeInstruct).
"""


def transform(cfg, **kwargs):
    seq_len = cfg.sequence_len

    def transform_fn(example, tokenizer=None):
        # Extract prompt and completion from the example
        prompt_text = example.get(
            "input", example.get("prompt", example.get("question", ""))
        )
        completion_text = example.get(
            "output", example.get("completion", example.get("answer", ""))
        )

        if tokenizer is None:
            return {"prompt": prompt_text}

        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

        # Tokenize prompt and completion separately
        prompt_enc = tokenizer(
            prompt_text,
            truncation=False,
            add_special_tokens=True,
            return_tensors=None,
        )
        completion_enc = tokenizer(
            completion_text,
            truncation=False,
            add_special_tokens=False,
            return_tensors=None,
        )

        prompt_ids = prompt_enc["input_ids"]
        completion_ids = completion_enc["input_ids"]

        # Truncate to fit within seq_len (prioritize keeping prompt + some completion)
        total_len = len(prompt_ids) + len(completion_ids)
        if total_len > seq_len:
            # Truncate completion first, then prompt if needed
            max_completion = seq_len - len(prompt_ids)
            if max_completion < 1:
                # Prompt alone exceeds seq_len — truncate prompt, keep at least 1 completion token
                prompt_ids = prompt_ids[: seq_len - 1]
                completion_ids = completion_ids[:1]
            else:
                completion_ids = completion_ids[:max_completion]

        input_ids = prompt_ids + completion_ids
        prompt_length = len(prompt_ids)

        # Labels: -100 for prompt tokens, input_ids for completion tokens
        labels = [-100] * prompt_length + completion_ids

        # Pad to seq_len
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

    # Signal to remove all original columns (filtered to existing ones at map time)
    return transform_fn, {
        "remove_columns": "__all__",
    }
