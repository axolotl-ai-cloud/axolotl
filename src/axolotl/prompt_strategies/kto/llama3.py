"""
KTO strategies for llama-3 chat template
"""

# pylint: disable=duplicate-code


def argilla(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|start_header_id|>system<|end_header_id|>\n\n{sample['system']}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            sample["prompt"] = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['instruction']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        sample["completion"] = f"{sample['completion']}<|eot_id|>"
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
            f"<|start_header_id|>user<|end_header_id|>\n\n{sample['completion'][0]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        sample["completion"] = f"{sample['completion'][1]['content']}<|eot_id|>"
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
                f"<|start_header_id|>system<|end_header_id|>\n\n{sample['system']}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            sample["prompt"] = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        sample["completion"] = f"{sample['completion']}<|eot_id|>"
        return sample

    return transform_fn


def prompt_pairs(
    cfg, **kwargs
):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|start_header_id|>system<|end_header_id|>\n\n{sample['system']}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            sample["prompt"] = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        sample["completion"] = f"{sample['completion']}<|eot_id|>"
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
                f"<|start_header_id|>system<|end_header_id|>\n\n{sample['system']}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            sample["prompt"] = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['prompt']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        sample["completion"] = f"{sample['completion']}<|eot_id|>"
        return sample

    return transform_fn
