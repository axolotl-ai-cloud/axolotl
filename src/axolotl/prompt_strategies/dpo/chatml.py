"""
DPO strategies for chatml
"""


def argilla_apply_chatml(
    cfg,
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


def intel_apply_chatml(cfg):  # pylint: disable=possibly-unused-variable,unused-argument
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


def apply_chatml(cfg):  # pylint: disable=possibly-unused-variable,unused-argument
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


def ultra_apply_chatml(cfg):  # pylint: disable=possibly-unused-variable,unused-argument
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
