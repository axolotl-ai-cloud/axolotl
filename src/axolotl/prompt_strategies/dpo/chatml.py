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
                f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
                f"<|im_start|>user\n{sample['instruction']}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|im_start|>user\n{sample['instruction']}<|im_end|>\n<|im_start|>assistant\n"
        sample["chosen"] = f"{sample['chosen_response']}<|im_end|>"
        sample["rejected"] = f"{sample['rejected_response']}<|im_end|>"
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
        ] = f"<|im_start|>user\n{sample['chosen'][0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        sample["chosen"] = f"{sample['chosen'][1]['content']}<|im_end|>"
        sample["rejected"] = f"{sample['rejected'][1]['content']}<|im_end|>"
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
                f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
                f"<|im_start|>user\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|im_start|>user\n{sample['input']}<|im_end|>\n<|im_start|>assistant\n"
        sample["chosen"] = f"{sample['chosen']}<|im_end|>"
        sample["rejected"] = f"{sample['rejected']}<|im_end|>"
        return sample

    return transform_fn


def intel(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    For Intel Orca DPO Pairs
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
                f"<|im_start|>user\n{sample['question']}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|im_start|>user\n{sample['question']}<|im_end|>\n<|im_start|>assistant\n"
        sample["chosen"] = f"{sample['chosen']}<|im_end|>"
        sample["rejected"] = f"{sample['rejected']}<|im_end|>"
        return sample

    return transform_fn


def prompt_pairs(
    cfg, **kwargs
):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
                f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        sample["chosen"] = f"{sample['chosen']}<|im_end|>"
        sample["rejected"] = f"{sample['rejected']}<|im_end|>"
        return sample

    return transform_fn


def ultra(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    for ultrafeedback binarized conversations
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
                f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            sample[
                "prompt"
            ] = f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
        sample["chosen"] = f"{sample['chosen'][1]['content']}<|im_end|>"
        sample["rejected"] = f"{sample['rejected'][1]['content']}<|im_end|>"
        return sample

    return transform_fn
