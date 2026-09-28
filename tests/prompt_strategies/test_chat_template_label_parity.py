"""Label parity between the one-pass turn locator and find_turn."""

from axolotl.prompt_strategies.chat_template import (
    ChatTemplatePrompter,
    ChatTemplateStrategy,
)
from axolotl.utils.chat_templates import get_chat_template


def _strategy(tokenizer, chat_template, roles_to_train=None):
    return ChatTemplateStrategy(
        ChatTemplatePrompter(
            tokenizer,
            chat_template=chat_template,
            message_property_mappings={"role": "role", "content": "content"},
        ),
        tokenizer=tokenizer,
        train_on_inputs=False,
        sequence_len=2048,
        roles_to_train=roles_to_train or ["assistant"],
        train_on_eos="turn",
    )


def _spans(strategy, prompt):
    turns = strategy.get_conversation_thread(prompt)
    tools = strategy._get_tools(prompt)
    result = strategy.prompter.build_prompt(turns, tools=tools)
    input_ids = result["input_ids"] if isinstance(result, dict) else result
    locator = strategy._build_turn_locator(turns, tools, input_ids)
    return strategy._locate_turns_from_content(turns, tools, input_ids, locator=locator)


def _assert_label_parity(strategy, prompt):
    fast = strategy.tokenize_prompt(prompt)
    original = strategy._locate_turns_from_content
    strategy._locate_turns_from_content = lambda *args, **kwargs: None
    try:
        slow = strategy.tokenize_prompt(prompt)
    finally:
        strategy._locate_turns_from_content = original
    assert fast["input_ids"] == slow["input_ids"]
    assert fast["labels"] == slow["labels"]


def test_unique_role_content_turns_match_find_turn(llama3_tokenizer):
    prompt = {
        "messages": [
            {"role": "user", "content": "alpha-user-question"},
            {"role": "assistant", "content": "alpha-assistant-answer"},
            {"role": "user", "content": "beta-user-question"},
            {"role": "assistant", "content": "beta-assistant-answer"},
        ]
    }
    strategy = _strategy(llama3_tokenizer, get_chat_template("llama3"))
    spans = _spans(strategy, prompt)
    assert spans is not None
    assert 1 in spans and 3 in spans
    _assert_label_parity(strategy, prompt)


def test_tool_call_turn_is_not_cached(
    llama3_tokenizer, toolcalling_dataset, llama3_2_vision_chat_template_jinja
):
    strategy = _strategy(
        llama3_tokenizer,
        get_chat_template("jinja", jinja_template=llama3_2_vision_chat_template_jinja),
    )
    prompt = toolcalling_dataset[0]
    turns = strategy.get_conversation_thread(prompt)
    tool_indexes = [
        i
        for i, turn in enumerate(turns)
        if set(turn) - {"role", "content", "training", "training_detail"}
    ]
    assert tool_indexes
    spans = _spans(strategy, prompt)
    if spans is not None:
        assert all(i not in spans for i in tool_indexes)
    _assert_label_parity(strategy, prompt)


def test_reasoning_turn_is_not_cached(llama3_tokenizer):
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
    turns = strategy.get_conversation_thread(prompt)
    spans = _spans(strategy, prompt)
    reasoning_indexes = [
        i for i, turn in enumerate(turns) if "reasoning_content" in turn
    ]
    assert reasoning_indexes
    if spans is not None:
        assert all(i not in spans for i in reasoning_indexes)
    _assert_label_parity(strategy, prompt)


def test_missing_content_disables_the_fast_path(llama3_tokenizer):
    template = "{% for message in messages %}{{ message['role'] }}\n{% endfor %}"
    prompt = {
        "messages": [
            {"role": "user", "content": "dropped-user-text"},
            {"role": "assistant", "content": "dropped-assistant-text"},
        ]
    }
    strategy = _strategy(
        llama3_tokenizer, get_chat_template("jinja", jinja_template=template)
    )
    assert _spans(strategy, prompt) is None
    _assert_label_parity(strategy, prompt)
