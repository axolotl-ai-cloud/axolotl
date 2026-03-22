"""
Dataset transform for unstructured text data with strided EBFT.

Tokenizes raw text into fixed-length input_ids for the strided trainer.
Sequences are padded to sequence_len for uniform batching.
"""


def transform(cfg, *args, **kwargs):
    seq_len = cfg.sequence_len

    def transform_fn(example, tokenizer=None):
        text = example.get("question", example.get("text", ""))
        if tokenizer is None:
            return {"prompt": text}

        encoded = tokenizer(
            text,
            truncation=True,
            max_length=seq_len,
            padding="max_length",
            add_special_tokens=True,
            return_tensors=None,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
            "labels": list(encoded["input_ids"]),
        }

    return transform_fn, {"remove_columns": ["question", "answer"]}
