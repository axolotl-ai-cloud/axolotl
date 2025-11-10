"""
DPO prompt strategies for using tokenizer chat templates.
"""

from axolotl.utils.chat_templates import extract_chat_template_args, get_chat_template
from axolotl.utils.schemas.utils import handle_legacy_message_fields_logic


def default(cfg, dataset_idx=0, **kwargs):
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

        chosen_raw = sample[field_chosen]
        if isinstance(chosen_raw, str):
            chosen_msg = {
                message_property_mappings["role"]: "assistant",
                message_property_mappings["content"]: chosen_raw,
            }
        elif isinstance(chosen_raw, dict):
            chosen_msg = chosen_raw
        else:
            chosen_msg = chosen_raw[-1]
        chosen = {
            "role": role_map[chosen_msg[message_property_mappings["role"]]],
            "content": chosen_msg[message_property_mappings["content"]],
        }

        rejected_raw = sample[field_rejected]
        if isinstance(rejected_raw, str):
            rejected_msg = {
                message_property_mappings["role"]: "assistant",
                message_property_mappings["content"]: rejected_raw,
            }
        elif isinstance(rejected_raw, dict):
            rejected_msg = rejected_raw
        else:
            rejected_msg = rejected_raw[-1]
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


def argilla_chat(cfg, dataset_idx=0, **kwargs):
    """
    DPO chat template strategy for argilla-style datasets.

    For argilla-style datasets where chosen/rejected contain full conversations
    instead of single response messages. Extracts the conversation history from
    the chosen field and formats both chosen/rejected responses using the
    configured chat template.

    Args:
        cfg: Configuration object containing chat_template and dataset settings
        dataset_idx: Index of the dataset in the config (default: 0)
        **kwargs: Additional keyword arguments (unused)

    Returns:
        tuple: (transform_fn, dataset_kwargs) where:
            - transform_fn: Function to transform dataset samples
            - dataset_kwargs: Dict with 'remove_columns' specifying columns to drop

    Dataset format:
        {
            "chosen": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ],
            "rejected": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
    """
    ds_cfg = cfg["datasets"][dataset_idx]
    ds_cfg = handle_legacy_message_fields_logic(ds_cfg)

    chat_template_choice, chat_template_jinja = extract_chat_template_args(
        cfg=cfg, ds_cfg=ds_cfg
    )
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

        chosen_raw = sample[field_chosen]
        rejected_raw = sample[field_rejected]

        # Extract messages (all but last) and responses (last message)
        chosen_messages = [
            {
                "role": role_map[m[message_property_mappings["role"]]],
                "content": m[message_property_mappings["content"]],
            }
            for m in chosen_raw[:-1]
        ]
        chosen_response = {
            "role": role_map[chosen_raw[-1][message_property_mappings["role"]]],
            "content": chosen_raw[-1][message_property_mappings["content"]],
        }

        rejected_response = {
            "role": role_map[rejected_raw[-1][message_property_mappings["role"]]],
            "content": rejected_raw[-1][message_property_mappings["content"]],
        }

        dummy_user_message = {"role": "user", "content": "[[dummy_message]]"}

        result = {}
        result["prompt"] = tokenizer.apply_chat_template(
            chosen_messages,
            add_generation_prompt=True,
            chat_template=chat_template_string,
            tokenize=False,
        )

        result["chosen"] = tokenizer.apply_chat_template(
            [dummy_user_message, chosen_response],
            add_generation_prompt=False,
            chat_template=chat_template_string,
            tokenize=False,
        )
        chosen_strip_index = result["chosen"].find(chosen_response["content"])
        result["chosen"] = result["chosen"][chosen_strip_index:].rstrip()

        result["rejected"] = tokenizer.apply_chat_template(
            [dummy_user_message, rejected_response],
            add_generation_prompt=False,
            chat_template=chat_template_string,
            tokenize=False,
        )
        rejected_strip_index = result["rejected"].find(rejected_response["content"])
        result["rejected"] = result["rejected"][rejected_strip_index:].rstrip()

        return result

    return transform_fn, {"remove_columns": [field_chosen, field_rejected]}
