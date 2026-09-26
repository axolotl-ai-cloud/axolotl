"""
Shared helpers for rendering OpenAI-format preference (chosen/rejected) messages
via tokenizer chat templates.

Reused across RL prompt strategies (DPO, ORPO, Bradley-Terry) so tool_calls,
tools, and reasoning_content handling only needs to be implemented once. The
tools/tool_calls JSON decoding helpers are also reused by the SFT
`chat_template` strategy for the same reason.
"""

import json

from axolotl.prompt_strategies.jinja_template_analyzer import JinjaTemplateAnalyzer
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


def parse_tools(tools):
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


def parse_tool_call_arguments(message):
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


def build_message_transform(message_property_mappings, role_map):
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

        parse_tool_call_arguments(transformed)
        return transformed

    return transform_message


# always preserved: template analysis misses properties accessed via
# `message.get(...)` (e.g. gemma4), so OpenAI message keys are unioned in
BASE_MSG_VARIABLES = frozenset(
    ["tool_calls", "tool_call_id", "name", "reasoning_content", "reasoning"]
)


def make_msg_variables_getter():
    """Cache chat template message variable analysis per template string."""
    cache = {}

    def get_msg_variables(chat_template_string):
        """Return the message properties used by the chat template."""
        if chat_template_string not in cache:
            cache[chat_template_string] = (
                JinjaTemplateAnalyzer(chat_template_string).get_message_vars("messages")
                | BASE_MSG_VARIABLES
            )
        return cache[chat_template_string]

    return get_msg_variables


DUMMY_USER_MESSAGE_CONTENT = "[[dummy_message]]"


def extract_response(full, prompt_prefix, content):
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


def render_preference_sample(
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
        result[key] = extract_response(full, dummy_prompt, response.get("content"))

    return result
