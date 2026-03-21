"""
Dataset transform for multi-turn chat data with strided EBFT.

Tokenizes conversations using the model's chat template, producing input_ids
with labels=-100 for system/user turns and real labels for assistant turns.
The strided trainer places anchors only within assistant completion spans.

Works with datasets in OpenAI chat format:
  [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
"""


def transform(cfg, **kwargs):
    seq_len = cfg.sequence_len

    def transform_fn(example, tokenizer=None):
        messages = example.get("messages", example.get("conversations", []))

        if tokenizer is None:
            # For preview: just return the first user message
            for m in messages:
                if m.get("role") == "user":
                    return {"prompt": m["content"]}
            return {"prompt": str(messages)}

        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id

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

        # Build labels: -100 for everything except assistant turns.
        # Strategy: tokenize incrementally to find assistant turn boundaries.
        labels = [-100] * len(input_ids)

        # Tokenize prefix up to each assistant turn to find boundaries
        prefix_messages = []
        for msg in messages:
            if msg["role"] == "assistant":
                # Tokenize prefix (everything before this assistant turn + generation prompt)
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

                # Tokenize prefix + this assistant turn
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

                # Mark assistant tokens as trainable
                for i in range(start, min(end, len(labels))):
                    labels[i] = input_ids[i]
            else:
                prefix_messages.append(msg)

        # Derive prompt_length as the position of the first non-masked label
        prompt_length = len(input_ids)  # default: all masked
        for i, lbl in enumerate(labels):
            if lbl != -100:
                prompt_length = i
                break

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

    return transform_fn, {
        "remove_columns": "__all__",
    }
