"""
DPO prompt strategies for using tokenizer chat templates.
"""

import json

from axolotl.prompt_strategies.jinja_template_analyzer import JinjaTemplateAnalyzer
from axolotl.utils.chat_templates import extract_chat_template_args, get_chat_template
from axolotl.utils.logging import get_logger
from axolotl.utils.schemas.utils import handle_legacy_message_fields_logic

LOG = get_logger(__name__)


def _parse_tools(tools):
    """Parse tools into a list of dicts, decoding JSON-encoded strings."""
    if tools is None:
        return None

    if isinstance(tools, str):
        try:
            tools = json.loads(tools)
        except json.JSONDecodeError as e:
            LOG.error(f"Error parsing tools as JSON. Error: {e}")
            raise

    if isinstance(tools, list):
        parsed_tools = []
        for tool in tools:
            # some datasets store each tool as a JSON-encoded string
            if isinstance(tool, str):
                try:
                    tool = json.loads(tool)
                except json.JSONDecodeError as e:
                    LOG.error(f"Error parsing tool as JSON. Tool: {tool!r}, Error: {e}")
                    raise
            if isinstance(tool, dict) and "function" in tool:
                function = tool["function"]
                params = function.get("parameters")
                if isinstance(params, str):
                    try:
                        function["parameters"] = json.loads(params)
                    except json.JSONDecodeError as e:
                        LOG.error(
                            f"Error parsing tool parameters as JSON. "
                            f"Function: {function.get('name', 'unknown')}, "
                            f"Parameters string: {params!r}, "
                            f"Error: {e}"
                        )
                        raise
            parsed_tools.append(tool)
        return parsed_tools

    raise ValueError(
        "Unknown tools format. Please convert it into a list[dict].\n"
        f"Current format: {type(tools)}"
    )


def _parse_tool_call_arguments(message):
    """Decode JSON-encoded tool call arguments so templates receive dicts."""
    for tool_call in message.get("tool_calls") or []:
        if "function" in tool_call and "arguments" in tool_call["function"]:
            args = tool_call["function"]["arguments"]
            if isinstance(args, str):
                try:
                    tool_call["function"]["arguments"] = json.loads(args)
                except json.JSONDecodeError as e:
                    LOG.error(
                        f"Error parsing tool_calls arguments as JSON. "
                        f"Function: {tool_call.get('function', {}).get('name', 'unknown')}, "
                        f"Arguments string: {args!r}, "
                        f"Error: {e}"
                    )
                    raise


def _build_message_transform(message_property_mappings, role_map):
    """Build a function that maps a raw dataset message to a chat template message,
    preserving any extra properties the chat template uses (e.g. tool_calls)."""

    def transform_message(message, msg_variables):
        """Map a raw dataset message to a chat template message."""
        transformed = {}
        for target, source in message_property_mappings.items():
            value = message.get(source)
            if value is not None:
                transformed[target] = value

        if "role" in transformed:
            transformed["role"] = role_map.get(transformed["role"], transformed["role"])

        mapped_sources = set(message_property_mappings.values())
        for key in msg_variables - mapped_sources:
            value = message.get(key)
            if value is not None:
                transformed[key] = value

        _parse_tool_call_arguments(transformed)
        return transformed

    return transform_message


# always preserved: template analysis misses properties accessed via
# `message.get(...)` (e.g. gemma4), so OpenAI message keys are unioned in
_BASE_MSG_VARIABLES = frozenset(
    ["tool_calls", "tool_call_id", "name", "reasoning_content", "reasoning"]
)


def _make_msg_variables_getter():
    """Cache chat template message variable analysis per template string."""
    cache = {}

    def get_msg_variables(chat_template_string):
        """Return the message properties used by the chat template."""
        if chat_template_string not in cache:
            cache[chat_template_string] = (
                JinjaTemplateAnalyzer(chat_template_string).get_message_vars("messages")
                | _BASE_MSG_VARIABLES
            )
        return cache[chat_template_string]

    return get_msg_variables


DUMMY_USER_MESSAGE_CONTENT = "[[dummy_message]]"


def _extract_response(full, prompt_prefix, content):
    """Strip the rendered dummy-user prompt from a response rendering.

    Strips the longest common prefix rather than requiring an exact prefix
    match, since a generation prompt can diverge slightly from the completed
    message rendering (e.g. thinking templates open `<think>` with different
    whitespace than a rendered `reasoning_content` block).
    """
    common = 0
    for prefix_char, full_char in zip(prompt_prefix, full, strict=False):
        if prefix_char != full_char:
            break
        common += 1
    response = full[common:]
    if DUMMY_USER_MESSAGE_CONTENT not in response:
        return response.rstrip()
    # Fallback: locate the response content directly
    if content:
        strip_index = full.find(content)
        if strip_index != -1:
            return full[strip_index:].rstrip()
    return full.rstrip()


def _render_dpo_sample(
    tokenizer,
    messages,
    chosen,
    rejected,
    chat_template_string,
    chat_template_kwargs,
    tools,
):
    """Render the prompt and extract the chosen/rejected response strings."""
    template_kwargs = {
        "chat_template": chat_template_string,
        "tokenize": False,
        **chat_template_kwargs,
    }
    if tools:
        template_kwargs["tools"] = tools

    dummy_user_message = {"role": "user", "content": DUMMY_USER_MESSAGE_CONTENT}

    result = {}
    result["prompt"] = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        **template_kwargs,
    )

    dummy_prompt = tokenizer.apply_chat_template(
        [dummy_user_message],
        add_generation_prompt=True,
        **template_kwargs,
    )

    for key, response in (("chosen", chosen), ("rejected", rejected)):
        full = tokenizer.apply_chat_template(
            [dummy_user_message, response],
            add_generation_prompt=False,
            **template_kwargs,
        )
        result[key] = _extract_response(full, dummy_prompt, response.get("content"))

    return result


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

    transform_message = _build_message_transform(message_property_mappings, role_map)
    get_msg_variables = _make_msg_variables_getter()

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

        return _render_dpo_sample(
            tokenizer,
            messages,
            chosen,
            rejected,
            chat_template_string,
            chat_template_kwargs,
            _parse_tools(sample.get(field_tools)),
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

    transform_message = _build_message_transform(message_property_mappings, role_map)
    get_msg_variables = _make_msg_variables_getter()

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

        return _render_dpo_sample(
            tokenizer,
            chosen_messages,
            chosen_response,
            rejected_response,
            chat_template_string,
            chat_template_kwargs,
            _parse_tools(sample.get(field_tools)),
        )

    return transform_fn, {"remove_columns": [field_chosen, field_rejected, field_tools]}
