"""
Dataset transform for nvidia/OpenCodeInstruct with EBFT structured mode.

Maps the dataset's `input` (prompt) and `output` (code solution) fields
to the format expected by the EBFT trainer (prompt + ground_truth).
"""


def transform(cfg, **kwargs):
    def transform_fn(example, tokenizer=None):
        return {
            "prompt": [
                {"role": "user", "content": example["input"]},
            ],
            "ground_truth": example["output"],
        }

    return transform_fn, {
        "remove_columns": "__all__",
    }
