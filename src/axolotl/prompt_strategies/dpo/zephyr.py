"""
DPO strategies for zephyr
"""


def nectar(cfg, **kwargs):
    def transform_fn(sample):
        data = {}
        data["prompt"] = (
            f"<|system|>\n</s>\n<|user|>\n{sample['prompt']}</s>\n<|assistant|>\n"
        )
        answers = sorted(sample["answers"], key=lambda x: x["rank"])
        data["chosen"] = answers[-1]["answer"]
        data["rejected"] = answers[-2]["answer"]

        return data

    return transform_fn
