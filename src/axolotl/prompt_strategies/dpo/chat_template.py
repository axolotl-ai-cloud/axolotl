"""
DPO prompt strategies for using tokenizer chat templates.
"""

from axolotl.utils.chat_templates import extract_chat_template_args, get_chat_template
from axolotl.utils.schemas.utils import handle_legacy_message_fields_logic


def default(
    cfg, dataset_idx=0, **kwargs
):  # pylint: disable=possibly-unused-variable,unused-argument
    ds_cfg = cfg["datasets"][dataset_idx]
    ds_cfg = handle_legacy_message_fields_logic(ds_cfg)

    chat_template_choice, chat_template_jinja = extract_chat_template_args(
        cfg=cfg, ds_cfg=ds_cfg
    )
    field_messages = ds_cfg.get("field_messages", "messages")
    field_chosen = ds_cfg.get("field_chosen", "chosen")
    field_rejected = ds_cfg.get("field_rejected", "rejected")
    message_property_mappings = ds_cfg.get(
        "message_property_mappings",
        {
            "role": "role",
            "content": "content",
        },
    )
    role_map_inv = ds_cfg.get(
        "roles",
        {
            "user": ["user"],
            "assistant": ["assistant"],
            "system": ["system"],
        },
    )
    role_map = {}
    for target, sources in role_map_inv.items():
        for source in sources:
            role_map[source] = target

    def transform_fn(sample, tokenizer=None):
        chat_template_string = get_chat_template(
            user_choice=chat_template_choice,
            jinja_template=chat_template_jinja,
            tokenizer=tokenizer,
        )

        messages = sample[field_messages]
        if isinstance(messages, str):
            messages = [
                {
                    message_property_mappings["role"]: "user",
                    message_property_mappings["content"]: messages,
                }
            ]

        messages = [
            {
                "role": role_map[m[message_property_mappings["role"]]],
                "content": m[message_property_mappings["content"]],
            }
            for m in messages
        ]
        chosen_msg = (
            sample[field_chosen]
            if isinstance(sample[field_chosen], dict)
            else sample[field_chosen][-1]
        )
        chosen = {
            "role": role_map[chosen_msg[message_property_mappings["role"]]],
            "content": chosen_msg[message_property_mappings["content"]],
        }
        rejected_msg = (
            sample[field_rejected]
            if isinstance(sample[field_rejected], dict)
            else sample[field_rejected][-1]
        )
        rejected = {
            "role": role_map[rejected_msg[message_property_mappings["role"]]],
            "content": rejected_msg[message_property_mappings["content"]],
        }
        dummy_user_message = {"role": "user", "content": "[[dummy_message]]"}

        result = {}
        result["prompt"] = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            chat_template=chat_template_string,
            tokenize=False,
        )

        result["chosen"] = tokenizer.apply_chat_template(
            [dummy_user_message, chosen],
            add_generation_prompt=False,
            chat_template=chat_template_string,
            tokenize=False,
        )
        chosen_strip_index = result["chosen"].find(chosen["content"])
        result["chosen"] = result["chosen"][chosen_strip_index:].rstrip()

        result["rejected"] = tokenizer.apply_chat_template(
            [dummy_user_message, rejected],
            add_generation_prompt=False,
            chat_template=chat_template_string,
            tokenize=False,
        )
        rejected_strip_index = result["rejected"].find(rejected["content"])
        result["rejected"] = result["rejected"][rejected_strip_index:].rstrip()

        return result

    return transform_fn, {"remove_columns": [field_messages]}
