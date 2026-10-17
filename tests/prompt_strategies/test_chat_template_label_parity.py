"""find_turn keeps per-turn diff labels without re-tokenizing each prefix."""

from axolotl.prompt_strategies.chat_template import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
)
from axolotl.utils.chat_templates import get_chat_template


def _strategy(tokenizer, chat_template):
    return ChatTemplateStrategy(
        ChatTemplatePrompter(
            tokenizer,
            chat_template=chat_template,
            message_property_mappings={"role": "role", "content": "content"},
        ),
        tokenizer=tokenizer,
        train_on_inputs=False,
        sequence_len=2048,
        roles_to_train=["assistant", "tool"],
        train_on_eos="turn",
    )


def _forbid_token_fallback(strategy):
    def _boom(*args, **kwargs):
        raise AssertionError("per-turn prefix re-tokenization was used")

    strategy._find_turn_from_tokens = _boom


def test_plain_turns_do_not_re_tokenize(llama3_tokenizer):
    prompt = {
        "messages": [
            {"role": "user", "content": "alpha-user-question"},
            {"role": "assistant", "content": "alpha-assistant-answer"},
            {"role": "user", "content": "beta-user-question"},
            {"role": "assistant", "content": "beta-assistant-answer"},
        ]
    }
    strategy = _strategy(llama3_tokenizer, get_chat_template("llama3"))
    _forbid_token_fallback(strategy)
    tokenized = strategy.tokenize_prompt(prompt)
    labels = tokenized["labels"]
    assert any(label != -100 for label in labels)
    assert tokenized["input_ids"]


def test_tool_call_labels_match_the_text_diff(
    llama3_tokenizer, toolcalling_dataset, llama3_2_vision_chat_template_jinja
):
    strategy = _strategy(
        llama3_tokenizer,
        get_chat_template("jinja", jinja_template=llama3_2_vision_chat_template_jinja),
    )
    prompt = toolcalling_dataset[0]
    turns = strategy.get_conversation_thread(prompt)
    tools = strategy._get_tools(prompt)
    result = strategy.prompter.build_prompt(turns, tools=tools)
    input_ids = result["input_ids"] if isinstance(result, dict) else result
    locator = strategy._build_turn_locator(turns, tools, list(input_ids))
    assert locator is not None
    tool_turn = next(i for i, turn in enumerate(turns) if "tool_calls" in turn)
    span = strategy.find_turn(
        turns=turns, turn_idx=tool_turn, tools=tools, locator=locator
    )
    assert span != (-1, -1)
    _forbid_token_fallback(strategy)
    strategy.tokenize_prompt(prompt)


def test_reasoning_turn_uses_the_text_diff(llama3_tokenizer):
    template = (
        "{% for message in messages %}"
        "{{ message['role'] }}\n"
        "{% if message['reasoning_content'] is defined %}"
        "{{ message['reasoning_content'] }}\n{% endif %}"
        "{{ message['content'] }}\n"
        "{% endfor %}"
    )
    prompt = {
        "messages": [
            {"role": "user", "content": "reason-user-question"},
            {
                "role": "assistant",
                "content": "reason-assistant-answer",
                "reasoning_content": "reason-private-trace",
            },
        ]
    }
    strategy = _strategy(
        llama3_tokenizer, get_chat_template("jinja", jinja_template=template)
    )
    _forbid_token_fallback(strategy)
    tokenized = strategy.tokenize_prompt(prompt)
    trained = tokenizer_decode_trained(llama3_tokenizer, tokenized)
    assert "reason-private-trace" in trained
    assert "reason-assistant-answer" in trained


def test_inserted_special_token_still_uses_the_text_diff(llama3_tokenizer):
    prompt = {
        "messages": [
            {"role": "user", "content": "shift-user-question"},
            {"role": "assistant", "content": "shift-assistant-answer"},
        ]
    }
    strategy = _strategy(llama3_tokenizer, get_chat_template("llama3"))
    turns = strategy.get_conversation_thread(prompt)
    result = strategy.prompter.build_prompt(turns)
    input_ids = result["input_ids"] if isinstance(result, dict) else list(result)
    shifted = [input_ids[0], *input_ids]
    locator = strategy._build_turn_locator(turns, None, shifted)
    assert locator is not None
    _forbid_token_fallback(strategy)
    span = strategy.find_turn(turns=turns, turn_idx=1, locator=locator)
    plain = strategy._build_turn_locator(turns, None, list(input_ids))
    plain_span = strategy.find_turn(turns=turns, turn_idx=1, locator=plain)
    assert span[0] == plain_span[0] + 1
    assert span[1] == plain_span[1] + 1


def tokenizer_decode_trained(tokenizer, tokenized):
    ids = [
        token
        for token, label in zip(
            tokenized["input_ids"], tokenized["labels"], strict=True
        )
        if label != -100
    ]
    return tokenizer.decode(ids)
