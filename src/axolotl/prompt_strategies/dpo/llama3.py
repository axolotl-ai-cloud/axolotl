"""
DPO strategies for chatml
"""


def argilla(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sample['system']}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{sample['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        sample["chosen"] = f"{sample['chosen_response']}<|eot_id|><|end_of_text|>"
        sample["rejected"] = f"{sample['rejected_response']}<|eot_id|><|end_of_text|>"
        return sample

    return transform_fn


def argilla_chat(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    for argilla/dpo-mix-7k conversations
    """

    def transform_fn(sample):
        sample[
            "prompt"
        ] = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{sample['chosen'][0]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        sample["chosen"] = f"{sample['chosen'][1]['content']}<|eot_id|><|end_of_text|>"
        sample["rejected"] = f"{sample['rejected'][1]['content']}<|eot_id|><|end_of_text|>"
        return sample

    return transform_fn


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
            sample["prompt"] = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sample['system']}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{sample['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        sample["chosen"] = f"{sample['chosen']}<|eot_id|><|end_of_text|>"
        sample["rejected"] = f"{sample['rejected']}<|eot_id|><|end_of_text|>"
        return sample

    return transform_fn


def intel(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    For Intel Orca DPO Pairs
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sample['system']}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{sample['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        sample["chosen"] = f"{sample['chosen']}<|eot_id|><|end_of_text|>"
        sample["rejected"] = f"{sample['rejected']}<|eot_id|><|end_of_text|>"
        return sample

    return transform_fn


def prompt_pairs(
    cfg, **kwargs
):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sample['system']}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{sample['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        sample["chosen"] = f"{sample['chosen']}<|eot_id|><|end_of_text|>"
        sample["rejected"] = f"{sample['rejected']}<|eot_id|><|end_of_text|>"
        return sample

    return transform_fn


def ultra(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    for ultrafeedback binarized conversations
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{sample['system']}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{sample['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        sample["chosen"] = f"{sample['chosen'][1]['content']}<|eot_id|><|end_of_text|>"
        sample["rejected"] = f"{sample['rejected'][1]['content']}<|eot_id|><|end_of_text|>"
        return sample

    return transform_fn
