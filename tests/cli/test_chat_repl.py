"""pytest tests for the interactive chat REPL (no model required)."""

import io
import json

import pytest
from rich.console import Console

from axolotl.cli.chat import (
    CausalTurnGenerator,
    ChatRepl,
    ChatSession,
    TurnResult,
    default_gen_params,
    longest_common_prefix_len,
    parse_gen_param_value,
    resolve_command,
    resolve_gen_param,
)


class FakeCache:
    """Stands in for DynamicCache in cache-planning tests."""

    def __init__(self, length=0, croppable=True):
        self.length = length
        self.croppable = croppable

    def crop(self, max_length):
        if not self.croppable:
            raise NotImplementedError("cannot crop")
        self.length = max_length

    def get_seq_length(self):
        return self.length


class FakeGenerator:
    """Records conversations passed in and returns canned replies."""

    def __init__(self, replies=None):
        self.replies = replies or ["canned reply"]
        self.calls = []

    def generate_turn(self, conversation, params, on_text):
        self.calls.append(([dict(m) for m in conversation], dict(params)))
        content = self.replies[min(len(self.calls) - 1, len(self.replies) - 1)]
        on_text(content)
        return TurnResult(content=content, prompt_tokens=10, new_tokens=3)


def make_repl(inputs, generator=None, session=None):
    lines = iter(inputs)

    def input_fn(_prompt):
        try:
            return next(lines)
        except StopIteration as err:
            raise EOFError from err

    generator = generator or FakeGenerator()
    repl = ChatRepl(
        generator=generator,
        session=session,
        console=Console(file=io.StringIO(), force_terminal=False),
        input_fn=input_fn,
    )
    return repl, generator


def cache_planner(cached_ids, cache):
    generator = CausalTurnGenerator.__new__(CausalTurnGenerator)
    generator._cache = cache  # pylint: disable=protected-access
    generator._cached_ids = cached_ids  # pylint: disable=protected-access
    generator._new_cache = FakeCache  # pylint: disable=protected-access
    return generator


class TestGenParams:
    def test_alias_resolution(self):
        assert resolve_gen_param("temp").key == "temperature"
        assert resolve_gen_param("max").key == "max_new_tokens"
        assert resolve_gen_param("rep").key == "repetition_penalty"
        assert resolve_gen_param("bogus") is None

    def test_value_validation(self):
        spec = resolve_gen_param("temperature")
        assert parse_gen_param_value(spec, "0.7") == 0.7
        with pytest.raises(ValueError):
            parse_gen_param_value(spec, "100")
        with pytest.raises(ValueError):
            parse_gen_param_value(spec, "abc")

    def test_nullable_params(self):
        assert parse_gen_param_value(resolve_gen_param("seed"), "none") is None
        assert parse_gen_param_value(resolve_gen_param("min_p"), "off") is None
        with pytest.raises(ValueError):
            parse_gen_param_value(resolve_gen_param("temperature"), "none")


class TestChatSession:
    def test_system_prompt_prepended(self):
        session = ChatSession()
        session.system = "be brief"
        session.add_user("hi")
        conversation = session.conversation()
        assert conversation[0] == {"role": "system", "content": "be brief"}
        assert conversation[1]["role"] == "user"

    def test_undo_removes_exchange(self):
        session = ChatSession()
        session.add_user("q1")
        session.add_assistant("a1")
        session.add_user("q2")
        session.add_assistant("a2")
        assert session.undo()
        assert [m["content"] for m in session.messages] == ["q1", "a1"]
        assert session.undo()
        assert not session.messages
        assert not session.undo()

    def test_drop_last_assistant_for_retry(self):
        session = ChatSession()
        session.add_user("q1")
        session.add_assistant("a1")
        assert session.drop_last_assistant()
        assert session.messages[-1]["role"] == "user"
        session.clear()
        assert not session.drop_last_assistant()

    def test_save_jsonl(self, tmp_path):
        session = ChatSession()
        session.system = "sys"
        session.add_user("q")
        session.add_assistant("a")
        path = tmp_path / "chat.jsonl"
        session.save_jsonl(str(path))
        session.save_jsonl(str(path))
        lines = path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        sample = json.loads(lines[0])
        assert [m["role"] for m in sample["messages"]] == [
            "system",
            "user",
            "assistant",
        ]


class TestCachePlanning:
    def test_prefix_extension_reuses_cache(self):
        cache = FakeCache(length=5)
        generator = cache_planner([1, 2, 3, 4, 5], cache)
        assert generator._prepare_cache([1, 2, 3, 4, 5, 6, 7]) == 5
        assert generator._cache is cache

    def test_divergence_crops_to_common_prefix(self):
        cache = FakeCache(length=5)
        generator = cache_planner([1, 2, 3, 4, 5], cache)
        assert generator._prepare_cache([1, 2, 3, 9, 9, 9]) == 3
        assert cache.length == 3
        assert generator._cached_ids == [1, 2, 3]

    def test_no_overlap_resets_cache(self):
        cache = FakeCache(length=3)
        generator = cache_planner([1, 2, 3], cache)
        assert generator._prepare_cache([7, 8, 9]) == 0
        assert generator._cache is not cache
        assert generator._cached_ids == []

    def test_uncroppable_cache_resets(self):
        cache = FakeCache(length=5, croppable=False)
        generator = cache_planner([1, 2, 3, 4, 5], cache)
        assert generator._prepare_cache([1, 2, 3, 9, 9]) == 0
        assert generator._cache is not cache

    def test_render_fully_cached_leaves_one_input_token(self):
        # cache covering all input tokens would give generate() nothing to process
        cache = FakeCache(length=5)
        generator = cache_planner([1, 2, 3, 4, 5], cache)
        assert generator._prepare_cache([1, 2, 3, 4, 5]) == 4
        assert cache.length == 4


class TestChatRepl:
    def test_message_generates_turn_with_history(self):
        repl, generator = make_repl(["hi", "again", "/quit"])
        repl.run()
        assert len(generator.calls) == 2
        second_conversation = generator.calls[1][0]
        assert [m["role"] for m in second_conversation] == [
            "user",
            "assistant",
            "user",
        ]
        assert repl.session.messages[-1]["content"] == "canned reply"

    def test_command_aliases(self):
        assert resolve_command("clear").name == "new"
        assert resolve_command("reset").name == "new"
        assert resolve_command("regen").name == "retry"
        assert resolve_command("q").name == "quit"
        assert resolve_command("?").name == "help"

    def test_new_clears_history_keeps_system_and_params(self):
        repl, generator = make_repl(
            ["/system be brief", "/temp 0.5", "hi", "/new", "next", "/quit"]
        )
        repl.run()
        assert repl.session.system == "be brief"
        assert repl.params["temperature"] == 0.5
        last_conversation = generator.calls[-1][0]
        assert [m["role"] for m in last_conversation] == ["system", "user"]
        assert last_conversation[1]["content"] == "next"

    def test_param_shortcut_and_set_forms(self):
        repl, _ = make_repl(
            ["/temp 0.3", "/set top_k 10", "/set max_tokens=64", "/quit"]
        )
        repl.run()
        assert repl.params["temperature"] == 0.3
        assert repl.params["top_k"] == 10
        assert repl.params["max_new_tokens"] == 64

    def test_invalid_param_value_not_applied(self):
        repl, _ = make_repl(["/temp 100", "/quit"])
        repl.run()
        assert repl.params["temperature"] == default_gen_params()["temperature"]

    def test_retry_regenerates_last_turn(self):
        repl, generator = make_repl(
            ["hi", "/retry", "/quit"], generator=FakeGenerator(["first", "second"])
        )
        repl.run()
        assert len(generator.calls) == 2
        retry_conversation = generator.calls[1][0]
        assert retry_conversation[-1] == {"role": "user", "content": "hi"}
        assert repl.session.messages[-1]["content"] == "second"

    def test_undo_command(self):
        repl, _ = make_repl(["hi", "/undo", "/quit"])
        repl.run()
        assert not repl.session.messages

    def test_multiline_input(self):
        repl, generator = make_repl(["first line\\", "second line", "/quit"])
        repl.run()
        assert generator.calls[0][0][0]["content"] == "first line\nsecond line"

    def test_thinking_block_kept_verbatim_in_history(self):
        reply = "<think>step by step</think>\nThe answer is 4."
        repl, _ = make_repl(["what is 2+2?", "/quit"], generator=FakeGenerator([reply]))
        repl.run()
        assert repl.session.messages[-1]["content"] == reply

    def test_unknown_command_does_not_generate(self):
        repl, generator = make_repl(["/bogus", "/quit"])
        repl.run()
        assert not generator.calls


def test_longest_common_prefix_len():
    assert longest_common_prefix_len([], [1, 2]) == 0
    assert longest_common_prefix_len([1, 2], [1, 2]) == 2
    assert longest_common_prefix_len([1, 2, 3], [1, 2]) == 2
    assert longest_common_prefix_len([1, 9], [1, 2, 3]) == 1
