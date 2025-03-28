"""
DPO strategies for llama-3 chat template
"""


def default(
    cfg,
    **kwargs,
):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        # pylint: disable=duplicate-code
        if "prompt" in sample.keys():
            prompt_key = "prompt"
        elif "input" in sample.keys():
            prompt_key = "input"
        elif "question" in sample.keys():
            prompt_key = "question"
        else:
            prompt_key = "instruction"

        if "chosen" in sample.keys():
            chosen_key = "chosen"
        else:
            chosen_key = "chosen_response"

        if "rejected" in sample.keys():
            rejected_key = "rejected"
        else:
            rejected_key = "rejected_response"

        if "system" in sample and sample["system"]:
            sample["prompt"] = (
                f"<|start_header_id|>system<|end_header_id|>\n\n{sample['system']}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample[prompt_key]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            sample["prompt"] = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample[prompt_key]}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        sample["chosen"] = f"{sample[chosen_key]}<|eot_id|>"
        sample["rejected"] = f"{sample[rejected_key]}<|eot_id|>"
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
        sample["prompt"] = (
            f"<|start_header_id|>user<|end_header_id|>\n\n{sample['chosen'][0]['content']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        sample["chosen"] = f"{sample['chosen'][1]['content']}<|eot_id|>"
        sample["rejected"] = f"{sample['rejected'][1]['content']}<|eot_id|>"
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
                f"<|start_header_id|>system<|end_header_id|>\n\n{sample['system']}<|eot_id|>"
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        else:
            sample["prompt"] = (
                f"<|start_header_id|>user<|end_header_id|>\n\n{sample['input']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )
        sample["chosen"] = f"{sample['chosen']}<|eot_id|>"
        sample["rejected"] = f"{sample['rejected']}<|eot_id|>"
        return sample

    return transform_fn


def intel(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    For Intel Orca DPO Pairs
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
        sample["chosen"] = f"{sample['chosen']}<|eot_id|>"
        sample["rejected"] = f"{sample['rejected']}<|eot_id|>"
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
        sample["chosen"] = f"{sample['chosen']}<|eot_id|>"
        sample["rejected"] = f"{sample['rejected']}<|eot_id|>"
        return sample

    return transform_fn


def ultra(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    for ultrafeedback binarized conversations
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
        sample["chosen"] = f"{sample['chosen'][1]['content']}<|eot_id|>"
        sample["rejected"] = f"{sample['rejected'][1]['content']}<|eot_id|>"
        return sample

    return transform_fn
