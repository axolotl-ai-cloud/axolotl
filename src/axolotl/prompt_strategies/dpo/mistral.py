"""
DPO strategies for mistral instruct
"""


def prompt_pairs(cfg):  # pylint: disable=possibly-unused-variable,unused-argument
    def transform_fn(sample):
        sample["prompt"] = f"[INST]{sample['prompt']}[/INST]"
        sample["chosen"] = f"{sample['chosen']}"
        sample["rejected"] = f"{sample['rejected']}"
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
        sample["prompt"] = f"[INST] {sample['chosen'][0]['content']} [/INST]"
        sample["chosen"] = f"{sample['chosen'][1]['content']}</s>"
        sample["rejected"] = f"{sample['rejected'][1]['content']}</s>"
        return sample

    return transform_fn
