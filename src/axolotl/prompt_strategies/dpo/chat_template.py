"""
DPO prompt strategies for using tokenizer chat templates.
"""

from axolotl.prompt_strategies.preference_chat_template import (
    build_message_transform,
    make_msg_variables_getter,
    parse_tools,
    render_preference_sample,
)
from axolotl.utils.chat_templates import extract_chat_template_args, get_chat_template
from axolotl.utils.schemas.utils import handle_legacy_message_fields_logic


def default(cfg, dataset_idx=0, **kwargs):
    """DPO chat template strategy for OpenAI-format datasets.

    Renders `field_messages` (with tools from `field_tools`) into the prompt
    and extracts the chosen/rejected response strings via the chat template.
    """
    ds_cfg = cfg["datasets"][dataset_idx]
    ds_cfg = handle_legacy_message_fields_logic(ds_cfg)

    chat_template_choice, chat_template_jinja = extract_chat_template_args(
        cfg=cfg, ds_cfg=ds_cfg
    )
    field_messages = ds_cfg.get("field_messages", "messages")
    field_chosen = ds_cfg.get("field_chosen", "chosen")
    field_rejected = ds_cfg.get("field_rejected", "rejected")
    field_tools = ds_cfg.get("field_tools", "tools")
    chat_template_kwargs = cfg.get("chat_template_kwargs") or {}
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
            "tool": ["tool"],
        },
    )
    role_map = {}
    for target, sources in role_map_inv.items():
        for source in sources:
            role_map[source] = target

    transform_message = build_message_transform(message_property_mappings, role_map)
    get_msg_variables = make_msg_variables_getter()

    def transform_fn(sample, tokenizer=None):
        """Map a dataset sample to prompt/chosen/rejected strings."""
        chat_template_string = get_chat_template(
            user_choice=chat_template_choice,
            jinja_template=chat_template_jinja,
            tokenizer=tokenizer,
        )
        msg_variables = get_msg_variables(chat_template_string)

        messages = sample[field_messages]
        if isinstance(messages, str):
            messages = [
                {
                    message_property_mappings["role"]: "user",
                    message_property_mappings["content"]: messages,
                }
            ]

        messages = [transform_message(m, msg_variables) for m in messages]

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
        chosen = transform_message(chosen_msg, msg_variables)

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
        rejected = transform_message(rejected_msg, msg_variables)

        return render_preference_sample(
            tokenizer,
            messages,
            chosen,
            rejected,
            chat_template_string,
            chat_template_kwargs,
            parse_tools(sample.get(field_tools)),
        )

    return transform_fn, {"remove_columns": [field_messages, field_tools]}


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
    field_tools = ds_cfg.get("field_tools", "tools")
    chat_template_kwargs = cfg.get("chat_template_kwargs") or {}
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
            "tool": ["tool"],
        },
    )
    role_map = {}
    for target, sources in role_map_inv.items():
        for source in sources:
            role_map[source] = target

    transform_message = build_message_transform(message_property_mappings, role_map)
    get_msg_variables = make_msg_variables_getter()

    def transform_fn(sample, tokenizer=None):
        """Map a dataset sample to prompt/chosen/rejected strings."""
        chat_template_string = get_chat_template(
            user_choice=chat_template_choice,
            jinja_template=chat_template_jinja,
            tokenizer=tokenizer,
        )
        msg_variables = get_msg_variables(chat_template_string)

        chosen_raw = sample[field_chosen]
        rejected_raw = sample[field_rejected]

        # Extract messages (all but last) and responses (last message)
        chosen_messages = [transform_message(m, msg_variables) for m in chosen_raw[:-1]]
        chosen_response = transform_message(chosen_raw[-1], msg_variables)
        rejected_response = transform_message(rejected_raw[-1], msg_variables)

        return render_preference_sample(
            tokenizer,
            chosen_messages,
            chosen_response,
            rejected_response,
            chat_template_string,
            chat_template_kwargs,
            parse_tools(sample.get(field_tools)),
        )

    return transform_fn, {"remove_columns": [field_chosen, field_rejected, field_tools]}
