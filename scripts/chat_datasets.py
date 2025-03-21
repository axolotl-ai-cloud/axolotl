"""
helper script to parse chat datasets into a usable yaml
"""

import click
import yaml
from datasets import load_dataset


@click.command()
@click.argument("dataset", type=str)
@click.option("--split", type=str, default="train")
def parse_dataset(dataset=None, split="train"):
    ds_cfg = {}
    ds_cfg["path"] = dataset
    ds_cfg["split"] = split
    ds_cfg["type"] = "chat_template"
    ds_cfg["chat_template"] = "<<<Replace based on your model>>>"

    dataset = load_dataset(dataset, split=split)
    features = dataset.features
    feature_keys = features.keys()
    field_messages = None
    for key in ["conversation", "conversations", "messages"]:
        if key in feature_keys:
            field_messages = key
            break
    if not field_messages:
        raise ValueError(
            f'No conversation field found in dataset: {", ".join(feature_keys)}'
        )
    ds_cfg["field_messages"] = field_messages

    message_fields = features[field_messages][0].keys()

    message_property_mappings = {"role": None, "content": None}
    for key in ["from", "role"]:
        if key in message_fields:
            message_property_mappings["role"] = key
            break
    if not message_property_mappings["role"]:
        raise ValueError(
            f'No role field found in messages: {", ".join(message_fields)}'
        )

    for key in ["content", "text", "value"]:
        if key in message_fields:
            message_property_mappings["content"] = key
            break
    if not message_property_mappings["content"]:
        raise ValueError(
            f'No content field found in messages: {", ".join(message_fields)}'
        )
    ds_cfg["message_property_mappings"] = message_property_mappings

    print(yaml.dump({"datasets": [ds_cfg]}))


if __name__ == "__main__":
    parse_dataset()
