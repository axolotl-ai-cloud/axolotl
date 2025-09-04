"""
chatml transforms for datasets with system, input, chosen, rejected to match llama3 chat template
"""


def icr(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    chatml transforms for datasets with system, input, chosen, rejected
    ex. https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            prompt = (
                f"<|start_header_id|>system<|end_header_id|>\n\n{sample['system']}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            prompt = f"<|start_header_id|>user<|end_header_id|>\n\n{sample['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        sample["chosen"] = prompt + f"{sample['chosen']}<|eot_id|>"
        sample["rejected"] = prompt + f"{sample['rejected']}<|eot_id|>"
        return sample

    return transform_fn
