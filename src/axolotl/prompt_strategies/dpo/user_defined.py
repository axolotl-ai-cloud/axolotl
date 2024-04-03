"""
User-defined DPO strategies
"""


def default(cfg, dataset_idx=0, **kwargs):  # pylint: disable=unused-argument
    ds_cfg = cfg["datasets"][dataset_idx]["type"]
    if not isinstance(ds_cfg, dict):
        raise ValueError(
            f"User-defined dataset type must be a dictionary. Got: {ds_cfg}"
        )
    field_prompt = ds_cfg.get("field_prompt", "prompt")
    field_system = ds_cfg.get("field_system", "system")
    field_chosen = ds_cfg.get("field_chosen", "chosen")
    field_rejected = ds_cfg.get("field_rejected", "rejected")
    prompt_format = ds_cfg.get("prompt_format")
    if not prompt_format:
        prompt_format = "{" + field_prompt + "}"
    chosen_format = ds_cfg.get("chosen_format")
    if not chosen_format:
        chosen_format = "{" + field_chosen + "}"
    rejected_format = ds_cfg.get("rejected_format")
    if not rejected_format:
        rejected_format = "{" + field_rejected + "}"

    def transform_fn(sample):
        if (
            "{" + field_system + "}" in prompt_format
            and field_system in sample
            and sample[field_system]
        ):
            sample["prompt"] = prompt_format.format(
                system=sample[field_system], prompt=sample[field_prompt]
            )
        else:
            sample["prompt"] = prompt_format.format(prompt=sample["prompt"])
        sample["chosen"] = chosen_format.format(chosen=sample[field_chosen])
        sample["rejected"] = rejected_format.format(rejected=sample[field_rejected])
        return sample

    return transform_fn
