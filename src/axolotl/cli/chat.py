"""Interactive multi-turn chat CLI for a trained model."""

import difflib
import json
import shlex
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable

import torch
from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.text import Text
from transformers import (
    DynamicCache,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)

from axolotl.cli.args import InferenceCliArgs
from axolotl.cli.utils import load_model_and_tokenizer, resolve_chat_template_str
from axolotl.integrations.base import PluginManager
from axolotl.telemetry.errors import send_errors
from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

USER_PROMPT = ">>> "
CONTINUATION_PROMPT = "... "

# marker pairs used by the bundled chat templates (qwen3/exaone4/phi_4 vs command_a)
THINK_MARKER_PAIRS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<|START_THINKING|>", "<|END_THINKING|>"),
)
DEFAULT_THINK_MARKERS = THINK_MARKER_PAIRS[0]


def detect_think_markers(chat_template_str: str | None) -> tuple[str, str]:
    """Picks the thinking marker pair the template works with. Called once at startup."""
    if chat_template_str:
        for pair in THINK_MARKER_PAIRS:
            if pair[1] in chat_template_str:
                return pair
    return DEFAULT_THINK_MARKERS


def detect_think_toggle_key(chat_template_str: str | None) -> str | None:
    """
    Finds the jinja variable the template uses to toggle thinking at render time
    (`enable_thinking` in our bundled gemma4/qwen3_5 templates; `thinking` on some
    hub templates). Called once at startup.
    """
    if not chat_template_str:
        return None
    if "enable_thinking" in chat_template_str:
        return "enable_thinking"
    if "thinking" in chat_template_str:
        return "thinking"
    return None


@dataclass(frozen=True)
class GenParamSpec:
    """Specification of a runtime-adjustable generation parameter."""

    key: str
    cast: Callable
    lo: float
    hi: float
    default: Any
    aliases: tuple[str, ...] = ()
    nullable: bool = False
    help: str = ""


GEN_PARAMS: tuple[GenParamSpec, ...] = (
    GenParamSpec(
        "temperature", float, 0.0, 5.0, 0.9, ("temp",), help="0 = greedy decoding"
    ),
    GenParamSpec("top_p", float, 0.0, 1.0, 0.95),
    GenParamSpec("top_k", int, 0, 1000, 40),
    GenParamSpec("min_p", float, 0.0, 1.0, None, nullable=True),
    GenParamSpec("max_new_tokens", int, 1, 1_000_000, 1024, ("max_tokens", "max")),
    GenParamSpec("repetition_penalty", float, 0.5, 3.0, 1.1, ("rep",)),
    GenParamSpec(
        "seed", int, 0, 2**32 - 1, None, nullable=True, help="`/set seed none` clears"
    ),
)


DIFFUSION_GEN_PARAMS: tuple[GenParamSpec, ...] = (
    GenParamSpec(
        "temperature", float, 0.0, 5.0, 0.0, ("temp",), help="0 = greedy denoising"
    ),
    GenParamSpec(
        "max_new_tokens",
        int,
        1,
        100_000,
        256,
        ("tokens", "max_tokens", "max"),
        help="size of the denoised completion block",
    ),
    GenParamSpec("steps", int, 1, 10_000, 128, help="number of denoising steps"),
    GenParamSpec(
        "seed", int, 0, 2**32 - 1, None, nullable=True, help="`/set seed none` clears"
    ),
)


def default_gen_params(
    specs: tuple[GenParamSpec, ...] = GEN_PARAMS,
) -> dict[str, Any]:
    return {spec.key: spec.default for spec in specs}


def resolve_gen_param(
    name: str, specs: tuple[GenParamSpec, ...] = GEN_PARAMS
) -> GenParamSpec | None:
    for spec in specs:
        if name == spec.key or name in spec.aliases:
            return spec
    return None


def parse_gen_param_value(spec: GenParamSpec, raw: str) -> Any:
    if spec.nullable and raw.lower() in ("none", "null", "off"):
        return None
    try:
        value = spec.cast(raw)
    except ValueError as err:
        raise ValueError(f"{spec.key} expects a {spec.cast.__name__}") from err
    if not spec.lo <= value <= spec.hi:
        raise ValueError(f"{spec.key} must be in [{spec.lo}, {spec.hi}]")
    return value


def longest_common_prefix_len(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n


def find_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        return -1
    n = len(needle)
    for i in range(start, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return -1


def partial_suffix_len(text: str, marker: str) -> int:
    """Length of the longest suffix of `text` that is a proper prefix of `marker`."""
    max_len = min(len(text), len(marker) - 1)
    for k in range(max_len, 0, -1):
        if text.endswith(marker[:k]):
            return k
    return 0


def content_as_text(content: Any) -> str:
    # parse_response may return content as a list of parts rather than a string
    if isinstance(content, list):
        return "".join(
            part.get("text", "") for part in content if isinstance(part, dict)
        )
    return content or ""


@dataclass
class ChatSession:
    """Holds the conversation state for a chat session."""

    messages: list[dict] = field(default_factory=list)
    system: str | None = None

    def conversation(self) -> list[dict]:
        prefix = [{"role": "system", "content": self.system}] if self.system else []
        return prefix + self.messages

    def add_user(self, content: str):
        # merge into a trailing unanswered user message (e.g. after a failed
        # generation) so strict templates never see consecutive user turns
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages[-1]["content"] += "\n" + content
            return
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.add_assistant_message({"role": "assistant", "content": content})

    def add_assistant_message(self, message: dict):
        self.messages.append(message)

    def clear(self):
        self.messages = []

    def undo(self) -> bool:
        """Removes the last user/assistant exchange. Returns False if empty."""
        if not self.messages:
            return False
        if self.messages[-1]["role"] == "assistant":
            self.messages.pop()
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()
        return True

    def drop_last_assistant(self) -> bool:
        """Removes the trailing assistant message so the turn can be retried."""
        if self.messages and self.messages[-1]["role"] == "assistant":
            self.messages.pop()
        return bool(self.messages) and self.messages[-1]["role"] == "user"

    def save_jsonl(self, path: str):
        # content-parts format: text-only today, but matches the multimodal
        # dataset format so saved sessions stay usable as training data
        messages = []
        for message in self.conversation():
            content = message.get("content")
            parts = (
                content
                if isinstance(content, list)
                else [{"type": "text", "text": content or ""}]
            )
            out = {
                "role": message["role"],
                "content": parts,
            }
            for key in ("reasoning_content", "thinking", "tool_calls"):
                if message.get(key):
                    out[key] = message[key]
            messages.append(out)
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")


@dataclass
class TurnResult:
    """Result of generating a single assistant turn."""

    content: str
    message: dict | None = None
    interrupted: bool = False
    prompt_tokens: int = 0
    reused_tokens: int = 0
    new_tokens: int = 0
    thinking_tokens: int = 0
    response_tokens: int = 0
    seconds: float = 0.0


class _StopOnEvent(StoppingCriteria):
    """Stops generation when the given event is set (e.g. on Ctrl+C)."""

    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.event.is_set()


class TurnGenerator:
    """Base for assistant-turn generators: template rendering and EOS handling."""

    def __init__(self, model, tokenizer, chat_template_str: str | None, device):
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template_str = chat_template_str
        self.device = device
        self.think_markers = detect_think_markers(
            chat_template_str or getattr(tokenizer, "chat_template", None)
        )
        self._think_marker_ids: tuple[list[int], list[int]] | None = None
        self._eos_strings: tuple[str, ...] | None = None

        self.eos_token_ids: set[int] = set()
        if tokenizer.eos_token_id is not None:
            self.eos_token_ids.add(tokenizer.eos_token_id)
        config_eos = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(config_eos, int):
            self.eos_token_ids.add(config_eos)
        elif isinstance(config_eos, (list, tuple)):
            self.eos_token_ids.update(config_eos)

    def render(
        self, conversation: list[dict], render_kwargs: dict | None = None
    ) -> list[int]:
        kwargs = dict(render_kwargs or {})
        if self.chat_template_str:
            kwargs["chat_template"] = self.chat_template_str
        batch = self.tokenizer.apply_chat_template(
            conversation,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            **kwargs,
        )
        return list(batch["input_ids"])

    def _split_think_token_ids(
        self, generated: list[int]
    ) -> tuple[list[int], list[int]]:
        """Splits a generated sequence into (thinking, response) token ids."""
        if self._think_marker_ids is None:
            try:
                self._think_marker_ids = (
                    self.tokenizer.encode(
                        self.think_markers[0], add_special_tokens=False
                    ),
                    self.tokenizer.encode(
                        self.think_markers[1], add_special_tokens=False
                    ),
                )
            except Exception:  # pylint: disable=broad-exception-caught
                self._think_marker_ids = ([], [])

        open_ids, close_ids = self._think_marker_ids
        i = find_subsequence(generated, open_ids)
        if i < 0:
            return [], generated
        j = find_subsequence(generated, close_ids, i + len(open_ids))
        if j < 0:
            return generated[i + len(open_ids) :], generated[:i]
        thinking = generated[i + len(open_ids) : j]
        response = generated[:i] + generated[j + len(close_ids) :]
        return thinking, response

    def split_think_token_counts(self, generated: list[int]) -> tuple[int, int]:
        """Returns (thinking, response) token counts for a generated sequence."""
        thinking, response = self._split_think_token_ids(generated)
        return len(thinking), len(response)

    def build_assistant_message(
        self,
        generated: list[int],
        split: tuple[list[int], list[int]] | None = None,
    ) -> dict:
        """
        Parses generated token ids into an assistant message dict. Prefers the
        tokenizer's own `parse_response` schema (transformers v5); otherwise splits
        thinking out of the content by marker and stores it under
        `reasoning_content`, the key the bundled chat templates read. Special
        tokens are kept out of the stored text either way — the template re-adds
        them on render.
        """
        if getattr(self.tokenizer, "response_schema", None):
            try:
                text = self.tokenizer.decode(generated, skip_special_tokens=False)
                message = self.tokenizer.parse_response(text)
                if isinstance(message, dict):
                    message.setdefault("role", "assistant")
                    message.setdefault("content", "")
                    return message
            except Exception:  # pylint: disable=broad-exception-caught
                LOG.warning(
                    "tokenizer.parse_response failed; falling back to marker split",
                    exc_info=True,
                )

        thinking_ids, response_ids = (
            split if split is not None else self._split_think_token_ids(generated)
        )
        parsed: dict[str, Any] = {
            "role": "assistant",
            "content": self.tokenizer.decode(
                response_ids, skip_special_tokens=True
            ).strip(),
        }
        if thinking_ids:
            parsed["reasoning_content"] = self.tokenizer.decode(
                thinking_ids, skip_special_tokens=True
            ).strip()
        return parsed

    def eos_strings(self) -> tuple[str, ...]:
        if self._eos_strings is None:
            self._eos_strings = tuple(
                text
                for token_id in sorted(self.eos_token_ids)
                if (text := self.tokenizer.decode([token_id]))
            )
        return self._eos_strings

    def generate_turn(
        self,
        conversation: list[dict],
        params: dict[str, Any],
        on_text: Callable[[str], None],
        render_kwargs: dict | None = None,
    ) -> TurnResult:
        raise NotImplementedError


class EosTextTrimmer:
    """
    Filters streamed text so terminal EOS markers (e.g. `<|im_end|>`) never reach
    the display. Text that could be the start of an EOS string is held back until
    disambiguated by the next chunk.
    """

    def __init__(self, eos_strings: tuple[str, ...], emit: Callable[[str], None]):
        self.eos_strings = tuple(s for s in eos_strings if s)
        self.emit = emit
        self.pending = ""
        self.done = False

    def feed(self, text: str):
        if self.done or not text:
            return
        self.pending += text
        positions = [
            idx for s in self.eos_strings if (idx := self.pending.find(s)) >= 0
        ]
        if positions:
            if min(positions) > 0:
                self.emit(self.pending[: min(positions)])
            self.pending = ""
            self.done = True
            return
        hold = max(
            (partial_suffix_len(self.pending, s) for s in self.eos_strings),
            default=0,
        )
        if len(self.pending) > hold:
            self.emit(self.pending[: len(self.pending) - hold])
            self.pending = self.pending[len(self.pending) - hold :]

    def finish(self):
        if not self.done and self.pending:
            self.emit(self.pending)
        self.pending = ""


class CausalTurnGenerator(TurnGenerator):
    """
    Generates assistant turns with `model.generate`, re-using the KV cache across
    turns when the rendered conversation extends the previously cached tokens.
    """

    def __init__(self, model, tokenizer, chat_template_str: str | None, device):
        super().__init__(model, tokenizer, chat_template_str, device)
        self._cache: DynamicCache | None = None
        self._cached_ids: list[int] = []

    def reset_cache(self):
        self._cache = None
        self._cached_ids = []

    def _new_cache(self) -> DynamicCache:
        return DynamicCache(config=self.model.config)

    def _prepare_cache(self, ids: list[int]) -> int:
        """
        Crops or resets the cross-turn cache so it holds a strict prefix of `ids`.
        Chat templates may rewrite earlier turns when re-rendering (e.g. stripping
        prior-turn thinking blocks), so reuse is gated on a token-level prefix
        check rather than assumed.

        Returns the number of re-used prefix tokens.
        """
        common = longest_common_prefix_len(self._cached_ids, ids)
        # generate() needs at least one uncached input token
        keep = min(common, len(ids) - 1)

        if self._cache is None or keep <= 0:
            self._cache = self._new_cache()
            self._cached_ids = []
            return 0

        if keep < len(self._cached_ids):
            try:
                self._cache.crop(keep)
                self._cached_ids = self._cached_ids[:keep]
            except Exception:  # pylint: disable=broad-exception-caught
                # some cache layer types (e.g. sliding window) cannot crop
                self._cache = self._new_cache()
                self._cached_ids = []
                return 0

        return len(self._cached_ids)

    def _build_generation_config(self, params: dict[str, Any]) -> GenerationConfig:
        do_sample = params["temperature"] > 0
        kwargs: dict[str, Any] = {
            "max_new_tokens": params["max_new_tokens"],
            "repetition_penalty": params["repetition_penalty"],
            "do_sample": do_sample,
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": sorted(self.eos_token_ids) or None,
            "pad_token_id": self.tokenizer.pad_token_id,
            "use_cache": True,
            "return_dict_in_generate": True,
        }
        if do_sample:
            kwargs["temperature"] = params["temperature"]
            kwargs["top_p"] = params["top_p"]
            kwargs["top_k"] = params["top_k"]
            if params["min_p"] is not None:
                kwargs["min_p"] = params["min_p"]
        return GenerationConfig(**kwargs)

    def generate_turn(
        self,
        conversation: list[dict],
        params: dict[str, Any],
        on_text: Callable[[str], None],
        render_kwargs: dict | None = None,
    ) -> TurnResult:
        ids = self.render(conversation, render_kwargs)
        reused = self._prepare_cache(ids)
        cache = self._cache
        assert cache is not None

        if params["seed"] is not None:
            torch.manual_seed(params["seed"])

        input_ids = torch.tensor([ids], dtype=torch.long, device=self.device)
        attention_mask = torch.ones_like(input_ids)
        generation_config = self._build_generation_config(params)

        stop_event = threading.Event()
        streamer = TextIteratorStreamer(
            self.tokenizer, skip_prompt=True, skip_special_tokens=False
        )
        holder: dict[str, Any] = {}

        def _worker():
            try:
                with torch.no_grad():
                    holder["output"] = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        generation_config=generation_config,
                        streamer=streamer,
                        stopping_criteria=StoppingCriteriaList(
                            [_StopOnEvent(stop_event)]
                        ),
                        past_key_values=cache,
                    )
            except Exception as err:  # pylint: disable=broad-exception-caught
                holder["error"] = err
                streamer.end()

        start = time.monotonic()
        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        trimmer = EosTextTrimmer(self.eos_strings(), on_text)
        interrupted = False
        try:
            for text in streamer:
                trimmer.feed(text)
        except KeyboardInterrupt:
            interrupted = True
            stop_event.set()
            for text in streamer:
                trimmer.feed(text)
        finally:
            # a 2nd Ctrl+C can escape the drain; join so the worker stops writing the cache
            stop_event.set()
            thread.join()
            trimmer.finish()
        seconds = time.monotonic() - start

        if "error" in holder:
            self.reset_cache()
            raise holder["error"]

        sequence = holder["output"].sequences[0].tolist()
        self._cached_ids = sequence[: cache.get_seq_length()]

        generated = sequence[len(ids) :]
        while generated and generated[-1] in self.eos_token_ids:
            generated.pop()
        thinking_ids, response_ids = self._split_think_token_ids(generated)
        message = self.build_assistant_message(generated, (thinking_ids, response_ids))

        return TurnResult(
            content=message.get("content") or "",
            message=message,
            interrupted=interrupted,
            prompt_tokens=len(ids),
            reused_tokens=reused,
            new_tokens=len(generated),
            thinking_tokens=len(thinking_ids),
            response_tokens=len(response_ids),
            seconds=seconds,
        )


class DiffusionTurnGenerator(TurnGenerator):
    """
    Generates assistant turns for diffusion LMs by appending a masked completion
    block to the rendered conversation and denoising it. The whole block resolves
    at once, so the reply is emitted in one piece rather than streamed.
    """

    def __init__(
        self, model, tokenizer, chat_template_str: str | None, device, mask_token_id
    ):
        super().__init__(model, tokenizer, chat_template_str, device)
        self.mask_token_id = int(mask_token_id)

    def generate_turn(
        self,
        conversation: list[dict],
        params: dict[str, Any],
        on_text: Callable[[str], None],
        render_kwargs: dict | None = None,
    ) -> TurnResult:
        from axolotl.integrations.diffusion import generate as diffusion_generate

        ids = self.render(conversation, render_kwargs)
        if params["seed"] is not None:
            torch.manual_seed(params["seed"])

        sequence = torch.tensor([ids], dtype=torch.long, device=self.device)

        start = time.monotonic()
        with torch.no_grad():
            result = diffusion_generate(
                self.model,
                self.tokenizer,
                original_sequence=sequence,
                num_diffusion_steps=params["steps"],
                temperature=params["temperature"],
                mask_token_id=self.mask_token_id,
                mode="completion",
                completion_tokens=params["max_new_tokens"],
            )
        seconds = time.monotonic() - start

        generated = result["generated_ids"][len(ids) :]
        for i, token_id in enumerate(generated):
            if token_id in self.eos_token_ids:
                generated = generated[:i]
                break
        content = self.tokenizer.decode(generated, skip_special_tokens=False)
        on_text(content)
        thinking_ids, response_ids = self._split_think_token_ids(generated)

        return TurnResult(
            content=content,
            message=self.build_assistant_message(
                generated, (thinking_ids, response_ids)
            ),
            prompt_tokens=len(ids),
            new_tokens=len(generated),
            thinking_tokens=len(thinking_ids),
            response_tokens=len(response_ids),
            seconds=seconds,
        )


class ThinkStreamRenderer:
    """
    Renders one streamed turn. When collapsing, thinking is shown as a rolling
    dim tail in a live region (so it never enters scrollback) and replaced by a
    one-line summary when the block closes; the reply streams normally. When
    collapse is off, this is a plain passthrough print.
    """

    LIVE_FPS = 12

    def __init__(
        self,
        console: Console,
        collapse: bool,
        markers: tuple[str, str] = DEFAULT_THINK_MARKERS,
        tail_lines: int = 6,
    ):
        self.console = console
        self.collapse = collapse
        self.open_marker, self.close_marker = markers
        self.tail_lines = tail_lines
        self.think_text = ""
        self._mode = "detect"
        self._pending = ""
        self._start = time.monotonic()
        self._live: Live | None = None
        self._last_live_update = 0.0

    def feed(self, text: str):
        if not self.collapse:
            print(text, end="", flush=True)
            return
        self._pending += text
        self._process()

    def finish(self, interrupted: bool = False):
        if not self.collapse:
            return
        if self._mode == "think":
            self.think_text += self._pending
            self._pending = ""
            reason = "interrupted" if interrupted else f"no {self.close_marker}"
            self._end_think(f" ({escape(reason)})")
        elif self._pending:
            print(self._pending, end="", flush=True)
            self._pending = ""

    def _process(self):
        while True:
            if self._mode == "detect":
                stripped = self._pending.lstrip()
                if stripped.startswith(self.open_marker):
                    idx = self._pending.find(self.open_marker)
                    self._pending = self._pending[idx + len(self.open_marker) :]
                    self._mode = "think"
                    self._live = Live(
                        Text(""),
                        console=self.console,
                        refresh_per_second=self.LIVE_FPS,
                        transient=True,
                    )
                    self._live.start()
                    continue
                if not stripped or self.open_marker.startswith(stripped):
                    return  # could still be a marker prefix; wait for more text
                self._mode = "reply"
                continue

            if self._mode == "think":
                idx = self._pending.find(self.close_marker)
                if idx >= 0:
                    self.think_text += self._pending[:idx]
                    self._pending = self._pending[
                        idx + len(self.close_marker) :
                    ].lstrip("\n")
                    self._end_think()
                    self._mode = "reply"
                    continue
                keep = partial_suffix_len(self._pending, self.close_marker)
                emit_until = len(self._pending) - keep
                self.think_text += self._pending[:emit_until]
                self._pending = self._pending[emit_until:]
                self._update_live()
                return

            # reply mode
            if self._pending:
                print(self._pending, end="", flush=True)
                self._pending = ""
            return

    def _update_live(self):
        if self._live is None:
            return
        # Live repaints at LIVE_FPS; building renderables faster than that is wasted
        now = time.monotonic()
        if now - self._last_live_update < 1 / self.LIVE_FPS:
            return
        self._last_live_update = now
        lines = self.think_text.splitlines()[-self.tail_lines :]
        self._live.update(Text("\n".join(lines), style="dim"))

    def _end_think(self, note: str = ""):
        if self._live is not None:
            self._live.stop()
            self._live = None
        seconds = time.monotonic() - self._start
        self.console.print(
            f"[dim]▸ thought for {seconds:.1f}s{note} · /expand to view[/dim]"
        )


@dataclass(frozen=True)
class Command:
    """A slash command with its aliases and handler."""

    name: str
    handler: str
    help: str
    aliases: tuple[str, ...] = ()
    usage: str = ""


COMMANDS: tuple[Command, ...] = (
    Command("help", "cmd_help", "show this help", ("?",)),
    Command(
        "new",
        "cmd_new",
        "clear the conversation (keeps system prompt and params)",
        ("clear", "reset"),
    ),
    Command(
        "system",
        "cmd_system",
        "show, set, or clear the system prompt",
        usage="/system [text|clear]",
    ),
    Command(
        "set",
        "cmd_set",
        "set a generation parameter",
        usage="/set <param> <value>",
    ),
    Command("status", "cmd_status", "show model and generation settings", ("params",)),
    Command("history", "cmd_history", "show the conversation so far"),
    Command("retry", "cmd_retry", "regenerate the last assistant reply", ("regen",)),
    Command("undo", "cmd_undo", "remove the last exchange"),
    Command(
        "save",
        "cmd_save",
        "append conversation as a chat_template-format JSONL sample",
        usage="/save [path]",
    ),
    Command(
        "think",
        "cmd_think",
        "toggle template-level thinking, if the template supports it",
        usage="/think [on|off|default]",
    ),
    Command(
        "collapse",
        "cmd_collapse",
        "collapse thinking blocks in the display",
        usage="/collapse [on|off]",
    ),
    Command("expand", "cmd_expand", "show the hidden thinking from the last reply"),
    Command("quit", "cmd_quit", "exit chat", ("exit", "q")),
)


def resolve_command(name: str) -> Command | None:
    for command in COMMANDS:
        if name == command.name or name in command.aliases:
            return command
    return None


class ChatRepl:
    """Interactive chat loop: slash commands plus streamed model turns."""

    def __init__(
        self,
        *,
        generator: TurnGenerator,
        session: ChatSession | None = None,
        params: dict[str, Any] | None = None,
        param_specs: tuple[GenParamSpec, ...] = GEN_PARAMS,
        console: Console | None = None,
        banner: dict[str, str] | None = None,
        input_fn: Callable[[str], str] | None = None,
        think_toggle_key: str | None = None,
        collapse_thinking: bool = True,
    ):
        self.generator = generator
        self.session = session or ChatSession()
        self.param_specs = param_specs
        self.params = params or default_gen_params(param_specs)
        self.console = console or Console()
        self.banner = banner or {}
        self.input_fn = input_fn or input
        self.think_toggle_key = think_toggle_key
        self.collapse_thinking = collapse_thinking
        self.render_kwargs: dict[str, Any] = {}
        self.last_think_text: str | None = None

    def run(self):
        self._print_banner()
        while True:
            try:
                line = self._read_line()
            except EOFError:
                break
            except KeyboardInterrupt:
                self.console.print("\n[dim]Use /quit to exit.[/dim]")
                continue

            line = line.strip()
            if not line:
                continue

            if line.startswith("/"):
                try:
                    action = self._dispatch(line)
                except Exception as err:  # pylint: disable=broad-exception-caught
                    self.console.print(f"[red]Command failed: {escape(str(err))}[/red]")
                    continue
                if action == "quit":
                    break
                if action == "regenerate":
                    self._generate_turn()
                continue

            self.session.add_user(line)
            self._generate_turn()

    def _read_line(self) -> str:
        parts = []
        prompt = USER_PROMPT
        while True:
            line = self.input_fn(prompt)
            if line.endswith("\\"):
                parts.append(line[:-1])
                prompt = CONTINUATION_PROMPT
                continue
            parts.append(line)
            break
        return "\n".join(parts)

    def _dispatch(self, line: str) -> str | None:
        name, _, args = line[1:].partition(" ")
        name = name.lower()
        args = args.strip()

        command = resolve_command(name)
        if command:
            return getattr(self, command.handler)(args)

        # bare parameter shortcuts: /temp 0.7, /top_p 0.9, ...
        spec = resolve_gen_param(name, self.param_specs)
        if spec:
            return self.cmd_set(f"{spec.key} {args}" if args else spec.key)

        candidates = [
            alias for command in COMMANDS for alias in (command.name, *command.aliases)
        ] + [alias for spec in self.param_specs for alias in (spec.key, *spec.aliases)]
        close = difflib.get_close_matches(name, candidates, n=1)
        hint = f" Did you mean /{close[0]}?" if close else ""
        self.console.print(f"[red]Unknown command /{name}.[/red]{hint}")
        return None

    def _generate_turn(self):
        renderer = ThinkStreamRenderer(
            self.console,
            collapse=self.collapse_thinking,
            markers=getattr(self.generator, "think_markers", DEFAULT_THINK_MARKERS),
        )
        try:
            result = self.generator.generate_turn(
                self.session.conversation(),
                self.params,
                renderer.feed,
                render_kwargs=self.render_kwargs or None,
            )
        except KeyboardInterrupt:
            # an escaped interrupt leaves the cache out of sync with _cached_ids
            reset_cache = getattr(self.generator, "reset_cache", None)
            if callable(reset_cache):
                reset_cache()
            renderer.finish(interrupted=True)
            print()
            self.console.print(
                "[dim]Interrupted; reply discarded. Your message is kept —"
                " /retry regenerates, /undo removes it.[/dim]"
            )
            return
        except Exception as err:  # pylint: disable=broad-exception-caught
            renderer.finish(interrupted=True)
            self.console.print(f"\n[red]Generation failed: {escape(str(err))}[/red]")
            self.console.print(
                "[dim]Your message is kept — /retry regenerates, /undo removes it.[/dim]"
            )
            return

        renderer.finish(interrupted=result.interrupted)
        message = result.message or {"role": "assistant", "content": result.content}
        self.last_think_text = (
            renderer.think_text.strip()
            or message.get("reasoning_content")
            or message.get("thinking")
            or None
        )

        print()
        self.session.add_assistant_message(message)
        token_summary = f"{result.new_tokens} tokens"
        if result.thinking_tokens:
            token_summary += (
                f" ({result.thinking_tokens} thinking · {result.response_tokens} reply)"
            )
        stats = (
            f"{token_summary} · {result.seconds:.1f}s · "
            f"{result.prompt_tokens} prompt ({result.reused_tokens} cached)"
        )
        if result.interrupted:
            stats += " · interrupted (partial reply kept; /retry to regenerate)"
        self.console.print(f"[dim]{stats}[/dim]")

    def _print_banner(self):
        self.console.print("[bold]axolotl chat[/bold]")
        for key, value in self.banner.items():
            self.console.print(f"[dim]{key}:[/dim] {escape(value)}")
        self.console.print(
            "[dim]Type a message to chat, /help for commands, \\ at line end to"
            " continue on the next line.[/dim]"
        )

    # --- command handlers (return "quit", "regenerate", or None) ---

    def cmd_help(self, _args: str) -> None:
        for command in COMMANDS:
            names = "/" + command.name
            if command.aliases:
                names += " (" + ", ".join("/" + a for a in command.aliases) + ")"
            usage = f" — {command.usage}" if command.usage else ""
            self.console.print(f"  [bold]{names}[/bold]: {command.help}{usage}")
        params = ", ".join(
            "/" + s.key + ("".join(f" /{a}" for a in s.aliases))
            for s in self.param_specs
        )
        self.console.print(f"  parameter shortcuts: {params}")
        return None

    def cmd_new(self, _args: str) -> None:
        self.session.clear()
        self.last_think_text = None
        reset_cache = getattr(self.generator, "reset_cache", None)
        if callable(reset_cache):
            reset_cache()
        self.console.print("[dim]Conversation cleared.[/dim]")
        return None

    def cmd_system(self, args: str) -> None:
        if not args:
            if self.session.system:
                self.console.print(escape(self.session.system))
            else:
                self.console.print("[dim]No system prompt set.[/dim]")
            return None
        if args.lower() == "clear":
            self.session.system = None
            self.console.print("[dim]System prompt cleared.[/dim]")
            return None
        self.session.system = args
        self.console.print("[dim]System prompt set.[/dim]")
        return None

    def cmd_set(self, args: str) -> str | None:
        tokens = args.replace("=", " ").split()
        if len(tokens) != 2:
            self.console.print("[red]Usage: /set <param> <value>[/red]")
            return None
        spec = resolve_gen_param(tokens[0].lower(), self.param_specs)
        if not spec:
            valid = ", ".join(s.key for s in self.param_specs)
            self.console.print(f"[red]Unknown parameter. Valid: {valid}[/red]")
            return None
        try:
            self.params[spec.key] = parse_gen_param_value(spec, tokens[1])
        except ValueError as err:
            self.console.print(f"[red]{err}[/red]")
            return None
        self.console.print(f"[dim]{spec.key} = {self.params[spec.key]}[/dim]")
        return None

    def cmd_status(self, _args: str) -> None:
        for key, value in self.banner.items():
            self.console.print(f"[dim]{key}:[/dim] {escape(value)}")
        for spec in self.param_specs:
            self.console.print(f"[dim]{spec.key}:[/dim] {self.params[spec.key]}")
        n_messages = len(self.session.messages)
        self.console.print(f"[dim]messages:[/dim] {n_messages}")
        return None

    def cmd_history(self, _args: str) -> None:
        conversation = self.session.conversation()
        if not conversation:
            self.console.print("[dim]No messages yet.[/dim]")
            return None
        for message in conversation:
            self.console.print(f"[bold]{message['role']}:[/bold]")
            reasoning = message.get("reasoning_content") or message.get("thinking")
            if reasoning:
                self.console.print(escape(content_as_text(reasoning)), style="dim")
            self.console.print(escape(content_as_text(message.get("content"))))
        return None

    def cmd_retry(self, _args: str) -> str | None:
        if not self.session.drop_last_assistant():
            self.console.print("[dim]Nothing to retry yet.[/dim]")
            return None
        return "regenerate"

    def cmd_undo(self, _args: str) -> None:
        if self.session.undo():
            self.last_think_text = None
            self.console.print("[dim]Removed last exchange.[/dim]")
        else:
            self.console.print("[dim]Nothing to undo.[/dim]")
        return None

    def cmd_save(self, args: str) -> None:
        if not self.session.messages:
            self.console.print("[dim]Nothing to save yet.[/dim]")
            return None
        path = (
            shlex.split(args)[0]
            if args
            else f"chat-{datetime.now().strftime('%Y%m%d-%H%M%S')}.jsonl"
        )
        self.session.save_jsonl(path)
        self.console.print(f"[dim]Saved conversation to {path}[/dim]")
        return None

    def cmd_think(self, args: str) -> None:
        if not args:
            current = self.render_kwargs.get(self.think_toggle_key or "", "default")
            self.console.print(f"[dim]template thinking: {current}[/dim]")
            return None
        if not self.think_toggle_key:
            self.console.print(
                "[dim]This chat template has no thinking toggle; thinking is"
                " controlled by the model/template itself.[/dim]"
            )
            return None
        value = args.lower()
        if value in ("on", "true"):
            self.render_kwargs[self.think_toggle_key] = True
        elif value in ("off", "false"):
            self.render_kwargs[self.think_toggle_key] = False
        elif value == "default":
            self.render_kwargs.pop(self.think_toggle_key, None)
        else:
            self.console.print("[red]Usage: /think [on|off|default][/red]")
            return None
        current = self.render_kwargs.get(self.think_toggle_key, "default")
        self.console.print(
            f"[dim]{self.think_toggle_key} = {current} (applies from next turn)[/dim]"
        )
        return None

    def cmd_collapse(self, args: str) -> None:
        value = args.lower()
        if value in ("on", "true", ""):
            self.collapse_thinking = True
        elif value in ("off", "false"):
            self.collapse_thinking = False
        else:
            self.console.print("[red]Usage: /collapse [on|off][/red]")
            return None
        state = "collapsed" if self.collapse_thinking else "shown raw"
        self.console.print(f"[dim]Thinking blocks will be {state}.[/dim]")
        return None

    def cmd_expand(self, _args: str) -> None:
        if self.last_think_text:
            self.console.print(escape(self.last_think_text), style="dim")
        else:
            self.console.print(
                "[dim]No hidden thinking recorded for the last reply.[/dim]"
            )
        return None

    def cmd_quit(self, _args: str) -> str:
        return "quit"


def _build_banner(cfg: DictDefault) -> dict[str, str]:
    banner = {"model": str(cfg.base_model)}

    if cfg.lora_model_dir:
        banner["adapter"] = f"{cfg.adapter or 'lora'} from {cfg.lora_model_dir}"
    elif cfg.adapter:
        banner["adapter"] = str(cfg.adapter)

    quant = []
    if cfg.load_in_4bit:
        quant.append("4-bit (bnb)")
    if cfg.load_in_8bit:
        quant.append("8-bit (bnb)")
    if cfg.qat:
        quant.append("QAT fake-quant active")
    if quant:
        banner["quantization"] = ", ".join(quant)

    if cfg.chat_template:
        template_name = getattr(cfg.chat_template, "value", cfg.chat_template)
        banner["chat template"] = f"config ({template_name})"
    elif cfg.datasets and cfg.datasets[0].type == "chat_template":
        banner["chat template"] = "dataset config"
    else:
        banner["chat template"] = "tokenizer default"

    return banner


@send_errors
def do_chat(
    *,
    cfg: DictDefault,
    cli_args: InferenceCliArgs,
):
    """
    Runs an interactive multi-turn chat session on the command line. The chat
    template is applied to the full conversation each turn, and generation
    parameters can be adjusted at runtime via slash commands.

    Args:
        cfg: Dictionary mapping `axolotl` config keys to values.
        cli_args: Inference-specific CLI arguments.
    """
    if cli_args.prompter:
        raise ValueError(
            "--chat does not support --prompter; legacy prompters are single-turn."
            " Use the default inference mode instead."
        )

    plugin_manager = PluginManager.get_instance()
    is_diffusion = any(
        plugin.__class__.__name__ == "DiffusionPlugin"
        for plugin in plugin_manager.plugins.values()
    )

    if not sys.stdin.isatty():
        raise ValueError(
            "--chat requires an interactive terminal. For piped input, use the"
            " default inference mode."
        )

    try:
        import readline  # noqa: F401  pylint: disable=unused-import
    except ImportError:
        pass

    model, tokenizer, _ = load_model_and_tokenizer(cfg=cfg, inference=True)
    if cfg.is_multimodal:
        LOG.warning(
            "Multimodal attachments are not supported in chat mode yet;"
            " proceeding with text-only chat."
        )

    chat_template_str = resolve_chat_template_str(cfg, tokenizer)
    if not chat_template_str and not tokenizer.chat_template:
        raise ValueError(
            "Chat mode requires a chat template. Set `chat_template` in your config"
            " or use a tokenizer that provides one."
        )

    model = model.to(cfg.device, dtype=cfg.torch_dtype)
    model.eval()

    banner = _build_banner(cfg)
    generator: TurnGenerator
    param_specs = GEN_PARAMS

    if is_diffusion:
        from axolotl.integrations.diffusion import resolve_mask_token_id

        mask_token_id = resolve_mask_token_id(tokenizer, cfg, allow_add=False)
        generator = DiffusionTurnGenerator(
            model, tokenizer, chat_template_str, cfg.device, mask_token_id
        )
        param_specs = DIFFUSION_GEN_PARAMS
        params = default_gen_params(param_specs)
        if cfg.diffusion.num_diffusion_steps:
            params["steps"] = cfg.diffusion.num_diffusion_steps
        if cfg.diffusion.generation_temperature is not None:
            params["temperature"] = cfg.diffusion.generation_temperature
        banner["mode"] = "diffusion (completion-block denoising)"
    else:
        generator = CausalTurnGenerator(model, tokenizer, chat_template_str, cfg.device)
        params = default_gen_params(param_specs)

    think_toggle_key = detect_think_toggle_key(
        chat_template_str or tokenizer.chat_template
    )
    repl = ChatRepl(
        generator=generator,
        params=params,
        param_specs=param_specs,
        banner=banner,
        think_toggle_key=think_toggle_key,
    )
    repl.run()
