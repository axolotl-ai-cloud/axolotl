"""
KTO strategies for chatml
"""

# pylint: disable=duplicate-code


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
            sample["prompt"] = (
                f"<|im_start|>user\n{sample['instruction']}<|im_end|>\n<|im_start|>assistant\n"
            )
        sample["completion"] = f"{sample['completion']}<|im_end|>"
        return sample

    return transform_fn


def argilla_chat(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    for argilla/kto-mix-15k conversations
    """

    def transform_fn(sample):
        sample["prompt"] = (
            f"<|im_start|>user\n{sample['chosen'][0]['content']}<|im_end|>\n<|im_start|>assistant\n"
        )
        sample["completion"] = f"{sample['completion'][1]['content']}<|im_end|>"
        return sample

    return transform_fn


def intel(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    For Intel Orca KTO
    ex: argilla/distilabel-intel-orca-kto
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
                f"<|im_start|>user\n{sample['question']}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            sample["prompt"] = (
                f"<|im_start|>user\n{sample['question']}<|im_end|>\n<|im_start|>assistant\n"
            )
        sample["completion"] = f"{sample['completion']}<|im_end|>"
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
            sample["prompt"] = (
                f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            )
        sample["completion"] = f"{sample['completion']}<|im_end|>"
        return sample

    return transform_fn


def ultra(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    for ultrafeedback binarized conversations
    ex: argilla/ultrafeedback-binarized-preferences-cleaned-kto
    """

    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|im_start|>system\n{sample['system']}<|im_end|>\n"
                f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            )
        else:
            sample["prompt"] = (
                f"<|im_start|>user\n{sample['prompt']}<|im_end|>\n<|im_start|>assistant\n"
            )
        sample["completion"] = f"{sample['completion']}<|im_end|>"
        return sample

    return transform_fn
