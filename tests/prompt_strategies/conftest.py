"""
shared fixtures for prompt strategies tests
"""

import pytest
from datasets import Dataset
from transformers import AutoTokenizer

from axolotl.prompt_strategies.jinja_template_analyzer import JinjaTemplateAnalyzer
from axolotl.utils.chat_templates import _CHAT_TEMPLATES

from tests.hf_offline_utils import enable_hf_offline


@pytest.fixture(name="assistant_dataset")
def fixture_assistant_dataset():
    return Dataset.from_list(
        [
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hello"},
                    {"role": "user", "content": "goodbye"},
                    {"role": "assistant", "content": "goodbye"},
                ]
            }
        ]
    )


@pytest.fixture(name="sharegpt_dataset")
def fixture_sharegpt_dataset():
    # pylint: disable=duplicate-code
    return Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "human", "value": "hello"},
                    {"from": "gpt", "value": "hello"},
                    {"from": "human", "value": "goodbye"},
                    {"from": "gpt", "value": "goodbye"},
                ]
            }
        ]
    )


@pytest.fixture(name="basic_dataset")
def fixture_basic_dataset():
    # pylint: disable=duplicate-code
    return Dataset.from_list(
        [
            {
                "conversations": [
                    {"from": "system", "value": "You are an AI assistant."},
                    {"from": "human", "value": "Hello"},
                    {"from": "assistant", "value": "Hi there!"},
                    {"from": "human", "value": "How are you?"},
                    {"from": "assistant", "value": "I'm doing well, thank you!"},
                ]
            }
        ]
    )


@pytest.fixture(name="toolcalling_dataset")
def fixture_toolcalling_dataset():
    # pylint: disable=duplicate-code
    return Dataset.from_list(
        [
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a bot that responds to weather queries. You should reply with the unit used in the queried location.",
                    },
                    {
                        "role": "user",
                        "content": "Hey, what's the temperature in Paris right now?",
                    },
                    {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "get_current_temperature",
                                    "arguments": {
                                        "location": "Paris, France",
                                        "unit": "celsius",
                                    },
                                },
                            }
                        ],
                    },
                    {
                        "role": "tool",
                        "name": "get_current_temperature",
                        "content": "22.0",
                    },
                    {
                        "role": "assistant",
                        "content": "The temperature in Paris is 22.0 degrees Celsius.",
                    },
                ]
            }
        ]
    )


@pytest.fixture(name="llama3_tokenizer", scope="session", autouse=True)
@enable_hf_offline
def fixture_llama3_tokenizer(
    download_llama3_8b_instruct_model_fixture,
):  # pylint: disable=unused-argument,redefined-outer-name
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Meta-Llama-3-8B-Instruct")

    return tokenizer


@pytest.fixture(name="smollm2_tokenizer", scope="session", autouse=True)
@enable_hf_offline
def fixture_smollm2_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")
    return tokenizer


@pytest.fixture(name="mistralv03_tokenizer", scope="session", autouse=True)
@enable_hf_offline
def fixture_mistralv03_tokenizer(
    download_mlx_mistral_7b_model_fixture,
):  # pylint: disable=unused-argument,redefined-outer-name
    tokenizer = AutoTokenizer.from_pretrained(
        "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
    )
    return tokenizer


@pytest.fixture(name="phi35_tokenizer", scope="session", autouse=True)
@enable_hf_offline
def fixture_phi35_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3.5-mini-instruct")
    return tokenizer


@pytest.fixture(name="gemma2_tokenizer", scope="session", autouse=True)
def fixture_gemma2_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("mlx-community/gemma-2-9b-it-4bit")

    return tokenizer


@pytest.fixture(name="mistralv03_tokenizer_chat_template_jinja")
def fixture_mistralv03_chat_template_jinja_w_system() -> str:
    return '{%- if messages[0]["role"] == "system" %}\n    {%- set system_message = messages[0]["content"] %}\n    {%- set loop_messages = messages[1:] %}\n{%- else %}\n    {%- set loop_messages = messages %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n{%- set user_messages = loop_messages | selectattr("role", "equalto", "user") | list %}\n\n{#- This block checks for alternating user/assistant messages, skipping tool calling messages #}\n{%- set ns = namespace() %}\n{%- set ns.index = 0 %}\n{%- for message in loop_messages %}\n    {%- if not (message.role == "tool" or message.role == "tool_results" or (message.tool_calls is defined and message.tool_calls is not none)) %}\n        {%- if (message["role"] == "user") != (ns.index % 2 == 0) %}\n            {{- raise_exception("After the optional system message, conversation roles must alternate user/assistant/user/assistant/...") }}\n        {%- endif %}\n        {%- set ns.index = ns.index + 1 %}\n    {%- endif %}\n{%- endfor %}\n\n{{- bos_token }}\n{%- for message in loop_messages %}\n    {%- if message["role"] == "user" %}\n        {%- if tools is not none and (message == user_messages[-1]) %}\n            {{- "[AVAILABLE_TOOLS] [" }}\n            {%- for tool in tools %}\n                {%- set tool = tool.function %}\n                {{- \'{"type": "function", "function": {\' }}\n                {%- for key, val in tool.items() if key != "return" %}\n                    {%- if val is string %}\n                        {{- \'"\' + key + \'": "\' + val + \'"\' }}\n                    {%- else %}\n                        {{- \'"\' + key + \'": \' + val|tojson }}\n                    {%- endif %}\n                    {%- if not loop.last %}\n                        {{- ", " }}\n                    {%- endif %}\n                {%- endfor %}\n                {{- "}}" }}\n                {%- if not loop.last %}\n                    {{- ", " }}\n                {%- else %}\n                    {{- "]" }}\n                {%- endif %}\n            {%- endfor %}\n            {{- "[/AVAILABLE_TOOLS]" }}\n            {%- endif %}\n        {%- if loop.first and system_message is defined %}\n            {{- "[INST] " + system_message + "\\n\\n" + message["content"] + "[/INST]" }}\n        {%- else %}\n            {{- "[INST] " + message["content"] + "[/INST]" }}\n        {%- endif %}\n    {%- elif message.tool_calls is defined and message.tool_calls is not none %}\n        {{- "[TOOL_CALLS] [" }}\n        {%- for tool_call in message.tool_calls %}\n            {%- set out = tool_call.function|tojson %}\n            {{- out[:-1] }}\n            {%- if not tool_call.id is defined or tool_call.id|length != 9 %}\n                {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}\n            {%- endif %}\n            {{- \', "id": "\' + tool_call.id + \'"}\' }}\n            {%- if not loop.last %}\n                {{- ", " }}\n            {%- else %}\n                {{- "]" + eos_token }}\n            {%- endif %}\n        {%- endfor %}\n    {%- elif message["role"] == "assistant" %}\n        {{- " " + message["content"]|trim + eos_token}}\n    {%- elif message["role"] == "tool_results" or message["role"] == "tool" %}\n        {%- if message.content is defined and message.content.content is defined %}\n            {%- set content = message.content.content %}\n        {%- else %}\n            {%- set content = message.content %}\n        {%- endif %}\n        {{- \'[TOOL_RESULTS] {"content": \' + content|string + ", " }}\n        {%- if not message.tool_call_id is defined or message.tool_call_id|length != 9 %}\n            {{- raise_exception("Tool call IDs should be alphanumeric strings with length 9!") }}\n        {%- endif %}\n        {{- \'"call_id": "\' + message.tool_call_id + \'"}[/TOOL_RESULTS]\' }}\n    {%- else %}\n        {{- raise_exception("Only user and assistant roles are supported, with the exception of an initial optional system message!") }}\n    {%- endif %}\n{%- endfor %}\n'


@pytest.fixture(name="gemma2_tokenizer_chat_template_jinja")
def fixture_gemma2_chat_template_jinja_w_system() -> str:
    return "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'assistant') %}{% set role = 'model' %}{% else %}{% set role = message['role'] %}{% endif %}{{ '<start_of_turn>' + role + '\n' + message['content'] | trim + '<end_of_turn>\n' }}{% endfor %}{% if add_generation_prompt %}{{'<start_of_turn>model\n'}}{% endif %}"


@pytest.fixture(name="llama3_2_vision_chat_template_jinja")
def fixture_llama3_2_vision_with_hardcoded_date() -> str:
    """Hardcodes the date in the template to avoid the need for date logic in the prompt"""

    template = _CHAT_TEMPLATES["llama3_2_vision"]

    old_date_logic = """{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}"""

    new_date_logic = """{%- set date_string = "17 Dec 2024" %}"""

    modified_template = template.replace(old_date_logic, new_date_logic)

    return modified_template


@pytest.fixture(name="chat_template_jinja_with_optional_fields")
def fixture_chat_template_jinja_with_optional_fields() -> str:
    return """{% for message in messages %}
{{'<|im_start|>'}}{{ message['role'] }}
{% if message['thoughts'] is defined %}[Thoughts: {{ message['thoughts'] }}]{% endif %}
{% if message['tool_calls'] is defined %}[Tool: {{ message['tool_calls'][0]['type'] }}]{% endif %}
{{ message['content'] }}{{'<|im_end|>'}}
{% endfor %}"""


@pytest.fixture(name="basic_jinja_template_analyzer")
def basic_jinja_template_analyzer():
    return JinjaTemplateAnalyzer(
        """{% for message in messages %}{% if message['role'] == 'system' and message['content'] %}{{'<|system|>
' + message['content'] + '<|end|>
'}}{% elif message['role'] == 'user' %}{{'<|user|>
' + message['content'] + '<|end|>
'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>
' + message['content'] + '<|end|>
'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>
' }}{% else %}{{ eos_token }}{% endif %}"""
    )


@pytest.fixture(name="mistral_jinja_template_analyzer")
def mistral_jinja_template_analyzer(mistralv03_tokenizer_chat_template_jinja):
    return JinjaTemplateAnalyzer(mistralv03_tokenizer_chat_template_jinja)
