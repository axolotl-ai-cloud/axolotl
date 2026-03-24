"""
Dataset transform for nvidia/OpenCodeInstruct with EBFT.

Maps the dataset's `input` (prompt) and `output` (code solution) fields
to the format expected by the EBFT trainer.
"""


def transform(cfg, *args, **kwargs):
    def transform_fn(example, tokenizer=None):
        return {
            "prompt": [
                {"role": "user", "content": example["input"]},
            ],
            "ground_truth": example["output"],
        }

    return transform_fn, {
        "remove_columns": [
            "id",
            "domain",
            "generation_algorithm",
            "llm_judgement",
            "unit_tests",
            "tests_execution_status",
            "average_test_score",
        ]
    }
