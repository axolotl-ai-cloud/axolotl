"""
Golden-output snapshot tests for the bundled chat templates.

Templates are whitespace-sensitive: a stray newline or space shifts token
boundaries and corrupts label masking. These tests render each template through
transformers' own jinja compiler (the exact path used at training time) and pin
the output byte-for-byte, so any future edit that changes rendering is caught.

To refresh a snapshot after an intentional template change, render the case and
paste the new expected string (see ``EXPECTED`` below).
"""

import pytest
from transformers.utils.chat_template_utils import _compile_jinja_template

from axolotl.utils.chat_templates import _CHAT_TEMPLATES


def render(template_name, messages, **kwargs):
    compiled = _compile_jinja_template(_CHAT_TEMPLATES[template_name])
    kwargs.setdefault("bos_token", "<s>")
    kwargs.setdefault("eos_token", "</s>")
    kwargs.setdefault("add_generation_prompt", False)
    return compiled.render(messages=messages, **kwargs)


# (case_id, template_name, messages, render_kwargs)
CASES: list[tuple[str, str, list, dict]] = [
    (
        "deepseek_v3_basic",
        "deepseek_v3",
        [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        {},
    ),
    (
        "deepseek_v3_tool_flow",
        "deepseek_v3",
        [
            {"role": "user", "content": "Weather in Paris?"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": '{"city": "Paris"}',
                        },
                    }
                ],
            },
            {"role": "tool", "content": "sunny"},
            {"role": "assistant", "content": "It's sunny in Paris."},
        ],
        {},
    ),
    # assistant closes a tool output with null content (regression: used to crash
    # on ``str + None``)
    (
        "deepseek_v3_null_close",
        "deepseek_v3",
        [
            {"role": "user", "content": "q"},
            {"role": "tool", "content": "TOUT"},
            {"role": "assistant", "content": None},
        ],
        {},
    ),
    (
        "qwen3_basic",
        "qwen3",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ],
        {},
    ),
    # assistant tool call with null content (regression: used to crash iterating
    # ``None`` in the non-string branch)
    (
        "qwen3_null_toolcall",
        "qwen3",
        [
            {"role": "user", "content": "q"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [{"function": {"name": "f", "arguments": {"a": 1}}}],
            },
        ],
        {},
    ),
    (
        "qwen3_multimodal_user",
        "qwen3",
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Describe"}, {"type": "image"}],
            },
            {"role": "assistant", "content": "An image."},
        ],
        {},
    ),
    (
        "mistral_v1_sys",
        "mistral_v1",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ],
        {},
    ),
    (
        "mistral_v2v3_sys",
        "mistral_v2v3",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ],
        {},
    ),
    (
        "mistral_v3_tekken_sys",
        "mistral_v3_tekken",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "user", "content": "Bye"},
        ],
        {},
    ),
    (
        "exaone4_sys",
        "exaone4",
        [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        {},
    ),
    (
        "falcon_h1_basic",
        "falcon_h1",
        [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ],
        {},
    ),
    (
        "pixtral_multimodal",
        "pixtral",
        [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {"type": "image"},
                ],
            },
            {"role": "assistant", "content": "A cat."},
        ],
        {},
    ),
    (
        "llama3_multimodal",
        "llama3",
        [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hi"}, {"type": "image"}],
            },
            {"role": "assistant", "content": "Hello"},
        ],
        {},
    ),
    (
        "gemma3_sys_list",
        "gemma3",
        [
            {"role": "system", "content": [{"type": "text", "text": "sys"}]},
            {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hello"}]},
        ],
        {},
    ),
]

EXPECTED = {
    "deepseek_v3_basic": "<s>You are helpful.<｜User｜>Hi<｜Assistant｜>Hello!<｜end▁of▁sentence｜>",
    "deepseek_v3_tool_flow": '<s><｜User｜>Weather in Paris?<｜Assistant｜><｜tool▁calls▁begin｜><｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather\n```json\n{"city": "Paris"}\n```<｜tool▁call▁end｜><｜tool▁calls▁end｜><｜end▁of▁sentence｜><｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>sunny<｜tool▁output▁end｜><｜tool▁outputs▁end｜>It\'s sunny in Paris.<｜end▁of▁sentence｜>',
    "deepseek_v3_null_close": "<s><｜User｜>q<｜tool▁outputs▁begin｜><｜tool▁output▁begin｜>TOUT<｜tool▁output▁end｜><｜tool▁outputs▁end｜><｜end▁of▁sentence｜>",
    "qwen3_basic": "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nHello!<|im_end|>\n",
    "qwen3_null_toolcall": '<|im_start|>user\nq<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n<tool_call>\n{"name": "f", "arguments": {"a": 1}}\n</tool_call><|im_end|>\n',
    "qwen3_multimodal_user": "<|im_start|>user\nDescribe<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nAn image.<|im_end|>\n",
    "mistral_v1_sys": "<s> [INST] sys\n\nHi [/INST] Hello</s> [INST] Bye [/INST]",
    "mistral_v2v3_sys": "<s>[INST] Hi[/INST] Hello</s>[INST] sys\n\nBye[/INST]",
    "mistral_v3_tekken_sys": "<s>[INST]Hi[/INST]Hello</s>[INST]sys\n\nBye[/INST]",
    "exaone4_sys": "[|system|]\nsys[|endofturn|]\n[|user|]\nHi[|endofturn|]\n[|assistant|]\n<think>\n\n</think>\n\nHello[|endofturn|]\n",
    "falcon_h1_basic": "<s><|im_start|>user\nHi<|im_end|>\n<|im_start|>assistant\nHello<|im_end|>\n",
    "pixtral_multimodal": "<s>[INST]What is this?[IMG][/INST]A cat.</s>",
    "llama3_multimodal": "<s><|start_header_id|>user<|end_header_id|>\n\nHi<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\nHello<|eot_id|>",
    "gemma3_sys_list": "<s><start_of_turn>user\nsys\n\nHi<end_of_turn>\n<start_of_turn>model\nHello<end_of_turn>\n",
}


@pytest.mark.parametrize(
    "case_id, template_name, messages, render_kwargs",
    CASES,
    ids=[c[0] for c in CASES],
)
def test_chat_template_snapshot(case_id, template_name, messages, render_kwargs):
    assert render(template_name, messages, **render_kwargs) == EXPECTED[case_id]
