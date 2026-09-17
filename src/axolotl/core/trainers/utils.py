"""Utils for Axolotl trainers"""


def trainable_tokens_per_sec_per_gpu(
    prev_trainable: float | None,
    curr_trainable: float,
    world_size: int,
    elapsed: float,
) -> float | None:
    """Effective per-GPU trainable-token throughput over a logging window.

    ``curr_trainable``/``prev_trainable`` are the cumulative trainable-token
    counter (SUM-reduced across all ranks) at this log and the previous one, so
    the delta covers every gradient-accumulation microbatch and the elapsed wall
    time captures in-window overhead. Returns None when there is no prior window.
    """
    if prev_trainable is None or elapsed <= 0:
        return None
    return (curr_trainable - prev_trainable) / max(1, world_size) / elapsed


def sanitize_kwargs_for_tagging(tag_names, kwargs=None):
    if isinstance(tag_names, str):
        tag_names = [tag_names]

    if kwargs is not None:
        if "tags" not in kwargs:
            kwargs["tags"] = tag_names
        elif "tags" in kwargs and isinstance(kwargs["tags"], list):
            kwargs["tags"].extend(tag_names)
        elif "tags" in kwargs and isinstance(kwargs["tags"], str):
            tag_names.append(kwargs["tags"])
            kwargs["tags"] = tag_names

    return kwargs


def sanitize_kwargs_for_ds_tagging(dataset_tags, kwargs=None):
    if isinstance(dataset_tags, str):
        dataset_tags = [dataset_tags]

    if (dataset_tags is not None) and (kwargs is not None):
        if "dataset_tags" not in kwargs:
            kwargs["dataset_tags"] = dataset_tags
        elif "dataset_tags" in kwargs and isinstance(kwargs["dataset_tags"], list):
            kwargs["dataset_tags"].extend(dataset_tags)
        elif "dataset_tags" in kwargs and isinstance(kwargs["dataset_tags"], str):
            dataset_tags.append(kwargs["dataset_tags"])
            kwargs["dataset_tags"] = dataset_tags

    return kwargs
