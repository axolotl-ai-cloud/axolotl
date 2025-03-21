"""Utils for Axolotl trainers"""


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
