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

    def __init__(self, replies=None, messages=None):
        self.replies = replies or ["canned reply"]
        self.messages = messages
        self.calls = []
        self.render_kwargs_seen = []

    def generate_turn(self, conversation, params, on_text, render_kwargs=None):
        self.calls.append(([dict(m) for m in conversation], dict(params)))
        self.render_kwargs_seen.append(
            dict(render_kwargs) if render_kwargs is not None else None
        )
        index = min(len(self.calls) - 1, len(self.replies) - 1)
        content = self.replies[index]
        message = dict(self.messages[index]) if self.messages else None
        on_text(content)
        return TurnResult(
            content=content, message=message, prompt_tokens=10, new_tokens=3
        )


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

    def test_add_user_merges_consecutive_user_messages(self):
        # a failed generation leaves a trailing user message; typing again must
        # not create consecutive user turns (strict templates reject them)
        session = ChatSession()
        session.add_user("first try")
        session.add_user("second try")
        assert [m["role"] for m in session.messages] == ["user"]
        assert session.messages[0]["content"] == "first try\nsecond try"

    def test_save_jsonl_keeps_reasoning_content(self, tmp_path):
        session = ChatSession()
        session.add_user("q")
        session.add_assistant_message(
            {"role": "assistant", "content": "a", "reasoning_content": "hmm"}
        )
        path = tmp_path / "chat.jsonl"
        session.save_jsonl(str(path))
        sample = json.loads(path.read_text(encoding="utf-8"))
        assistant = sample["messages"][1]
        assert assistant["content"] == [{"type": "text", "text": "a"}]
        assert assistant["reasoning_content"] == "hmm"

    def test_save_jsonl_multimodal_parts_format(self, tmp_path):
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
        assert sample["messages"][1]["content"] == [{"type": "text", "text": "q"}]
        assert sample["messages"][2]["content"] == [{"type": "text", "text": "a"}]


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

    def test_generator_message_stored_in_history(self):
        message = {
            "role": "assistant",
            "content": "The answer is 4.",
            "reasoning_content": "step by step",
        }
        repl, _ = make_repl(
            ["what is 2+2?", "/quit"],
            generator=FakeGenerator(["The answer is 4."], messages=[message]),
        )
        repl.run()
        assert repl.session.messages[-1] == message
        # renderer saw no think markers, so /expand falls back to the message
        assert repl.last_think_text == "step by step"

    def test_legacy_content_fallback_kept_verbatim(self):
        # generators that return no message dict store their content as-is
        reply = "<think>step by step</think>\nThe answer is 4."
        repl, _ = make_repl(["what is 2+2?", "/quit"], generator=FakeGenerator([reply]))
        repl.run()
        assert repl.session.messages[-1]["content"] == reply

    def test_unknown_command_does_not_generate(self):
        repl, generator = make_repl(["/bogus", "/quit"])
        repl.run()
        assert not generator.calls

    def test_command_handler_error_does_not_crash_repl(self):
        # unclosed quote makes shlex raise inside /save
        repl, generator = make_repl(["hi", '/save "unclosed', "again", "/quit"])
        repl.run()
        assert len(generator.calls) == 2

    def test_keyboard_interrupt_keeps_session_alive(self):
        class InterruptingGenerator(FakeGenerator):
            def generate_turn(self, conversation, params, on_text, render_kwargs=None):
                if not self.calls:
                    self.calls.append(None)
                    raise KeyboardInterrupt
                return super().generate_turn(
                    conversation, params, on_text, render_kwargs
                )

        repl, generator = make_repl(
            ["hi", "again", "/quit"], generator=InterruptingGenerator()
        )
        repl.run()
        # interrupted turn keeps the user message; the next one merges into it
        assert [m["role"] for m in repl.session.messages] == ["user", "assistant"]
        assert repl.session.messages[0]["content"] == "hi\nagain"
        assert len(generator.calls) == 2

    def test_generation_failure_keeps_session_alive(self):
        class FailingGenerator(FakeGenerator):
            def generate_turn(self, conversation, params, on_text, render_kwargs=None):
                if not self.calls:
                    self.calls.append(None)
                    raise RuntimeError("boom")
                return super().generate_turn(
                    conversation, params, on_text, render_kwargs
                )

        repl, _ = make_repl(["hi", "/retry", "/quit"], generator=FailingGenerator())
        repl.run()
        assert [m["role"] for m in repl.session.messages] == ["user", "assistant"]
        assert repl.session.messages[-1]["content"] == "canned reply"


def test_longest_common_prefix_len():
    assert longest_common_prefix_len([], [1, 2]) == 0
    assert longest_common_prefix_len([1, 2], [1, 2]) == 2
    assert longest_common_prefix_len([1, 2, 3], [1, 2]) == 2
    assert longest_common_prefix_len([1, 9], [1, 2, 3]) == 1


class TestDiffusionChat:
    def test_diffusion_param_specs(self):
        from axolotl.cli.chat import DIFFUSION_GEN_PARAMS

        lines = iter(["/steps 32", "/tokens 64", "/top_p 0.9", "/quit"])

        def input_fn(_prompt):
            try:
                return next(lines)
            except StopIteration as err:
                raise EOFError from err

        repl = ChatRepl(
            generator=FakeGenerator(),
            param_specs=DIFFUSION_GEN_PARAMS,
            console=Console(file=io.StringIO(), force_terminal=False),
            input_fn=input_fn,
        )
        repl.run()
        assert repl.params["steps"] == 32
        assert repl.params["max_new_tokens"] == 64
        assert "top_p" not in repl.params

    def test_diffusion_turn_cuts_at_eos(self, monkeypatch):
        from types import SimpleNamespace

        import axolotl.integrations.diffusion as diffusion_module
        from axolotl.cli.chat import (
            DIFFUSION_GEN_PARAMS,
            DiffusionTurnGenerator,
            default_gen_params,
        )

        class FakeTokenizer:
            eos_token_id = 2

            def apply_chat_template(self, conversation, **kwargs):
                return {"input_ids": [1, 5, 6]}

            def decode(self, ids, **kwargs):
                return ",".join(str(i) for i in ids)

        fake_model = SimpleNamespace(
            generation_config=SimpleNamespace(eos_token_id=None)
        )

        def fake_generate(model, tokenizer, **kwargs):
            assert kwargs["mode"] == "completion"
            assert kwargs["completion_tokens"] == 256
            return {"generated_ids": [1, 5, 6, 7, 8, 2, 4]}

        monkeypatch.setattr(diffusion_module, "generate", fake_generate)

        generator = DiffusionTurnGenerator(
            fake_model, FakeTokenizer(), None, "cpu", mask_token_id=9
        )
        chunks = []
        result = generator.generate_turn(
            [{"role": "user", "content": "hi"}],
            default_gen_params(DIFFUSION_GEN_PARAMS),
            chunks.append,
        )
        assert result.content == "7,8"
        assert result.new_tokens == 2
        assert result.prompt_tokens == 3
        assert chunks == ["7,8"]


def test_unknown_command_suggests_alias():
    buf = io.StringIO()
    repl = ChatRepl(
        generator=FakeGenerator(),
        console=Console(file=buf, force_terminal=False, width=200),
        input_fn=lambda _p: "/quit",
    )
    repl._dispatch("/clea")
    assert "Did you mean /clear?" in buf.getvalue()
    repl._dispatch("/tem")
    assert "Did you mean /temp?" in buf.getvalue()


class TestThinkStreamRenderer:
    def make_renderer(self, collapse=True, markers=("<think>", "</think>")):
        from axolotl.cli.chat import ThinkStreamRenderer

        buf = io.StringIO()
        console = Console(file=buf, force_terminal=False, width=200)
        return ThinkStreamRenderer(console, collapse=collapse, markers=markers), buf

    def test_collapse_splits_thinking_from_reply(self, capsys):
        renderer, buf = self.make_renderer()
        for chunk in ["<thi", "nk>\nreasoning he", "re</th", "ink>\n\nAnswer!"]:
            renderer.feed(chunk)
        renderer.finish()
        assert renderer.think_text.strip() == "reasoning here"
        assert capsys.readouterr().out == "Answer!"
        assert "thought for" in buf.getvalue()

    def test_no_thinking_passthrough(self, capsys):
        renderer, buf = self.make_renderer()
        renderer.feed("Just a plain reply")
        renderer.finish()
        assert renderer.think_text == ""
        assert capsys.readouterr().out == "Just a plain reply"
        assert "thought for" not in buf.getvalue()

    def test_unterminated_thinking(self, capsys):
        renderer, buf = self.make_renderer()
        renderer.feed("<think>partial reasoning")
        renderer.finish()
        assert renderer.think_text == "partial reasoning"
        assert capsys.readouterr().out == ""
        assert "no </think>" in buf.getvalue()

    def test_collapse_off_is_passthrough(self, capsys):
        renderer, buf = self.make_renderer(collapse=False)
        renderer.feed("<think>abc</think>reply")
        renderer.finish()
        assert capsys.readouterr().out == "<think>abc</think>reply"
        assert buf.getvalue() == ""

    def test_custom_markers(self, capsys):
        renderer, _ = self.make_renderer(
            markers=("<|START_THINKING|>", "<|END_THINKING|>")
        )
        renderer.feed("<|START_THINKING|>hmm<|END_THINKING|>ok")
        renderer.finish()
        assert renderer.think_text == "hmm"
        assert capsys.readouterr().out == "ok"


class TestThinkTokenSplit:
    def make_generator(self, vocab):
        from types import SimpleNamespace

        from axolotl.cli.chat import TurnGenerator

        class FakeTokenizer:
            eos_token_id = 0
            chat_template = None

            def encode(self, text, **kwargs):
                return vocab[text]

        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=None))
        return TurnGenerator(model, FakeTokenizer(), None, "cpu")

    def test_split_counts(self):
        generator = self.make_generator({"<think>": [100], "</think>": [101]})
        assert generator.split_think_token_counts([100, 1, 2, 3, 101, 7, 8]) == (3, 2)
        assert generator.split_think_token_counts([100, 1, 2]) == (2, 0)
        assert generator.split_think_token_counts([5, 6]) == (0, 2)
        assert generator.split_think_token_counts([]) == (0, 0)


class TestBuildAssistantMessage:
    VOCAB = {
        1: "step by step",
        2: "The answer is 4.",
        50: "<eos>",
        100: "<think>",
        101: "</think>",
    }
    SPECIAL = {50, 100, 101}

    def make_generator(self, response_schema=None, parse_response=None):
        from types import SimpleNamespace

        from axolotl.cli.chat import TurnGenerator

        vocab, special = self.VOCAB, self.SPECIAL

        class FakeTokenizer:
            eos_token_id = 50
            chat_template = None

            def encode(self, text, **kwargs):
                return [token_id for token_id, t in vocab.items() if t == text]

            def decode(self, ids, skip_special_tokens=False):
                return "".join(
                    vocab[i] for i in ids if not (skip_special_tokens and i in special)
                )

        tokenizer = FakeTokenizer()
        if response_schema is not None:
            tokenizer.response_schema = response_schema
            tokenizer.parse_response = parse_response
        model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=None))
        return TurnGenerator(model, tokenizer, None, "cpu")

    def test_thinking_split_into_reasoning_content(self):
        generator = self.make_generator()
        message = generator.build_assistant_message([100, 1, 101, 2])
        assert message == {
            "role": "assistant",
            "content": "The answer is 4.",
            "reasoning_content": "step by step",
        }

    def test_no_thinking_omits_reasoning_key(self):
        generator = self.make_generator()
        message = generator.build_assistant_message([2])
        assert message == {"role": "assistant", "content": "The answer is 4."}

    def test_special_tokens_stripped_from_content(self):
        generator = self.make_generator()
        message = generator.build_assistant_message([2, 50])
        assert message["content"] == "The answer is 4."

    def test_parse_response_schema_preferred(self):
        generator = self.make_generator(
            response_schema={"x": "regex"},
            parse_response=lambda text: {"content": "parsed", "thinking": "hmm"},
        )
        message = generator.build_assistant_message([2])
        assert message == {
            "role": "assistant",
            "content": "parsed",
            "thinking": "hmm",
        }

    def test_parse_response_failure_falls_back_to_markers(self):
        def boom(text):
            raise ValueError("bad schema")

        generator = self.make_generator(
            response_schema={"x": "regex"}, parse_response=boom
        )
        message = generator.build_assistant_message([100, 1, 101, 2])
        assert message["content"] == "The answer is 4."
        assert message["reasoning_content"] == "step by step"


class TestEosTextTrimmer:
    def make_trimmer(self, eos_strings=("<|im_end|>",)):
        from axolotl.cli.chat import EosTextTrimmer

        chunks = []
        return EosTextTrimmer(eos_strings, chunks.append), chunks

    def test_eos_marker_never_emitted(self):
        trimmer, chunks = self.make_trimmer()
        trimmer.feed("Hello")
        trimmer.feed(" world<|im_end|>")
        trimmer.finish()
        assert "".join(chunks) == "Hello world"

    def test_eos_split_across_chunks(self):
        trimmer, chunks = self.make_trimmer()
        trimmer.feed("Hi<|im_")
        trimmer.feed("end|>")
        trimmer.finish()
        assert "".join(chunks) == "Hi"

    def test_false_partial_released(self):
        trimmer, chunks = self.make_trimmer()
        trimmer.feed("a<")
        trimmer.feed("b")
        trimmer.finish()
        assert "".join(chunks) == "a<b"

    def test_plain_text_passthrough(self):
        trimmer, chunks = self.make_trimmer()
        trimmer.feed("no special tokens here")
        trimmer.finish()
        assert "".join(chunks) == "no special tokens here"


class TestThinkCommands:
    def test_think_toggle_sets_render_kwargs(self):
        repl, generator = make_repl(
            ["/think off", "hi", "/think default", "again", "/quit"]
        )
        repl.think_toggle_key = "enable_thinking"
        repl.run()
        assert generator.render_kwargs_seen[0] == {"enable_thinking": False}
        assert generator.render_kwargs_seen[1] is None

    def test_think_without_toggle_key(self):
        repl, generator = make_repl(["/think off", "hi", "/quit"])
        repl.run()
        assert generator.render_kwargs_seen[0] is None

    def test_collapse_toggle_and_expand(self):
        reply = "<think>secret reasoning</think>\nAnswer."
        repl, _ = make_repl(
            ["hi", "/expand", "/collapse off", "/quit"],
            generator=FakeGenerator([reply]),
        )
        repl.run()
        assert repl.last_think_text == "secret reasoning"
        assert repl.collapse_thinking is False
        # raw content with thinking still stored in history
        assert repl.session.messages[-1]["content"] == reply


def test_detect_think_markers_and_toggle_key():
    from axolotl.cli.chat import detect_think_markers, detect_think_toggle_key

    qwen_like = "{% if enable_thinking %}...{{ '</think>' }}{% endif %}"
    assert detect_think_toggle_key(qwen_like) == "enable_thinking"
    assert detect_think_markers(qwen_like) == ("<think>", "</think>")

    command_a_like = "...<|START_THINKING|>...<|END_THINKING|>..."
    assert detect_think_markers(command_a_like) == (
        "<|START_THINKING|>",
        "<|END_THINKING|>",
    )
    assert detect_think_toggle_key(command_a_like) is None
    assert detect_think_toggle_key("{% if thinking %}x{% endif %}") == "thinking"
    assert detect_think_toggle_key(None) is None
