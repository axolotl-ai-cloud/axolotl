"""
User-defined KTO strategies
"""

# pylint: disable=duplicate-code


def default(cfg, dataset_idx=0, **kwargs):  # pylint: disable=unused-argument
    ds_cfg = cfg["datasets"][dataset_idx]["type"]
    if not isinstance(ds_cfg, dict):
        raise ValueError(
            f"User-defined dataset type must be a dictionary. Got: {ds_cfg}"
        )
    field_prompt = ds_cfg.get("field_prompt", "prompt")
    field_system = ds_cfg.get("field_system", "system")
    field_completion = ds_cfg.get("field_completion", "completion")
    field_label = ds_cfg.get("field_label", "label")
    prompt_format = ds_cfg.get("prompt_format")
    if not prompt_format:
        prompt_format = "{" + field_prompt + "}"
    completion_format = ds_cfg.get("completion_format")
    if not completion_format:
        chosen_format = "{" + field_completion + "}"

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
        sample["completion"] = chosen_format.format(chosen=sample[field_completion])
        sample["label"] = sample[field_label]
        return sample

    return transform_fn
