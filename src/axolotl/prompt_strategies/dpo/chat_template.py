"""
DPO prompt strategies for using tokenizer chat templates.
"""

from axolotl.utils.chat_templates import chat_templates


def default(
    cfg, dataset_idx=0, **kwargs
):  # pylint: disable=possibly-unused-variable,unused-argument
    ds_cfg = cfg["datasets"][dataset_idx]
    chat_template_str = chat_templates(cfg.chat_template)

    field_messages = ds_cfg.get("field_messages", "messages")
    field_chosen = ds_cfg.get("field_chosen", "chosen")
    field_rejected = ds_cfg.get("field_rejected", "rejected")
    field_message_role = ds_cfg.get("message_field_role", "role")
    field_message_content = ds_cfg.get("message_field_content", "content")
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
        messages = sample[field_messages]
        messages = [
            {
                "role": role_map[m[field_message_role]],
                "content": m[field_message_content],
            }
            for m in messages
        ]
        chosen = {
            "role": role_map[sample[field_chosen][field_message_role]],
            "content": sample[field_chosen][field_message_content],
        }
        rejected = {
            "role": role_map[sample[field_rejected][field_message_role]],
            "content": sample[field_rejected][field_message_content],
        }

        result = {}
        result["prompt"] = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            chat_template=chat_template_str,
            tokenize=False,
        )

        result["chosen"] = tokenizer.apply_chat_template(
            [chosen],
            add_generation_prompt=False,
            chat_template=chat_template_str,
            tokenize=False,
        )
        chosen_strip_index = result["chosen"].find(chosen["content"])
        result["chosen"] = result["chosen"][chosen_strip_index:]

        result["rejected"] = tokenizer.apply_chat_template(
            [rejected],
            add_generation_prompt=False,
            chat_template=chat_template_str,
            tokenize=False,
        )
        rejected_strip_index = result["rejected"].find(rejected["content"])
        result["rejected"] = result["rejected"][rejected_strip_index:]

        return result

    return transform_fn
