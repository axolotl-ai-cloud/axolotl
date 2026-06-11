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
from rich.markup import escape
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


def default_gen_params() -> dict[str, Any]:
    return {spec.key: spec.default for spec in GEN_PARAMS}


def resolve_gen_param(name: str) -> GenParamSpec | None:
    for spec in GEN_PARAMS:
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


@dataclass
class ChatSession:
    """Holds the conversation state for a chat session."""

    messages: list[dict] = field(default_factory=list)
    system: str | None = None

    def conversation(self) -> list[dict]:
        prefix = [{"role": "system", "content": self.system}] if self.system else []
        return prefix + self.messages

    def add_user(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

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
        with open(path, "a", encoding="utf-8") as file:
            file.write(
                json.dumps({"messages": self.conversation()}, ensure_ascii=False) + "\n"
            )


@dataclass
class TurnResult:
    """Result of generating a single assistant turn."""

    content: str
    interrupted: bool = False
    prompt_tokens: int = 0
    reused_tokens: int = 0
    new_tokens: int = 0
    seconds: float = 0.0


class _StopOnEvent(StoppingCriteria):
    """Stops generation when the given event is set (e.g. on Ctrl+C)."""

    def __init__(self, event: threading.Event):
        self.event = event

    def __call__(self, input_ids, scores, **kwargs) -> bool:
        return self.event.is_set()


class CausalTurnGenerator:
    """
    Generates assistant turns with `model.generate`, re-using the KV cache across
    turns when the rendered conversation extends the previously cached tokens.
    """

    def __init__(self, model, tokenizer, chat_template_str: str | None, device):
        self.model = model
        self.tokenizer = tokenizer
        self.chat_template_str = chat_template_str
        self.device = device
        self._cache: DynamicCache | None = None
        self._cached_ids: list[int] = []

        self.eos_token_ids: set[int] = set()
        if tokenizer.eos_token_id is not None:
            self.eos_token_ids.add(tokenizer.eos_token_id)
        config_eos = getattr(model.generation_config, "eos_token_id", None)
        if isinstance(config_eos, int):
            self.eos_token_ids.add(config_eos)
        elif isinstance(config_eos, (list, tuple)):
            self.eos_token_ids.update(config_eos)

    def render(self, conversation: list[dict]) -> list[int]:
        kwargs = {}
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
    ) -> TurnResult:
        ids = self.render(conversation)
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

        interrupted = False
        try:
            for text in streamer:
                on_text(text)
        except KeyboardInterrupt:
            interrupted = True
            stop_event.set()
            for text in streamer:
                on_text(text)
        thread.join()
        seconds = time.monotonic() - start

        if "error" in holder:
            self.reset_cache()
            raise holder["error"]

        sequence = holder["output"].sequences[0].tolist()
        self._cached_ids = sequence[: cache.get_seq_length()]

        generated = sequence[len(ids) :]
        while generated and generated[-1] in self.eos_token_ids:
            generated.pop()
        content = self.tokenizer.decode(generated, skip_special_tokens=False)

        return TurnResult(
            content=content,
            interrupted=interrupted,
            prompt_tokens=len(ids),
            reused_tokens=reused,
            new_tokens=len(generated),
            seconds=seconds,
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
        generator: CausalTurnGenerator,
        session: ChatSession | None = None,
        params: dict[str, Any] | None = None,
        console: Console | None = None,
        banner: dict[str, str] | None = None,
        input_fn: Callable[[str], str] | None = None,
    ):
        self.generator = generator
        self.session = session or ChatSession()
        self.params = params or default_gen_params()
        self.console = console or Console()
        self.banner = banner or {}
        self.input_fn = input_fn or input

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
                action = self._dispatch(line)
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
        spec = resolve_gen_param(name)
        if spec:
            return self.cmd_set(f"{spec.key} {args}" if args else spec.key)

        candidates = [c.name for c in COMMANDS] + [s.key for s in GEN_PARAMS]
        close = difflib.get_close_matches(name, candidates, n=1)
        hint = f" Did you mean /{close[0]}?" if close else ""
        self.console.print(f"[red]Unknown command /{name}.[/red]{hint}")
        return None

    def _generate_turn(self):
        def on_text(text: str):
            print(text, end="", flush=True)

        try:
            result = self.generator.generate_turn(
                self.session.conversation(), self.params, on_text
            )
        except KeyboardInterrupt:
            raise
        except Exception as err:  # pylint: disable=broad-exception-caught
            self.console.print(f"\n[red]Generation failed: {escape(str(err))}[/red]")
            self.console.print("[dim]The last message is kept; /undo removes it.[/dim]")
            return

        print()
        self.session.add_assistant(result.content)
        stats = (
            f"{result.new_tokens} tokens · {result.seconds:.1f}s · "
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
            "/" + s.key + ("".join(f" /{a}" for a in s.aliases)) for s in GEN_PARAMS
        )
        self.console.print(f"  parameter shortcuts: {params}")
        return None

    def cmd_new(self, _args: str) -> None:
        self.session.clear()
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
        spec = resolve_gen_param(tokens[0].lower())
        if not spec:
            valid = ", ".join(s.key for s in GEN_PARAMS)
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
        for spec in GEN_PARAMS:
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
            self.console.print(escape(message["content"]))
        return None

    def cmd_retry(self, _args: str) -> str | None:
        if not self.session.drop_last_assistant():
            self.console.print("[dim]Nothing to retry yet.[/dim]")
            return None
        return "regenerate"

    def cmd_undo(self, _args: str) -> None:
        if self.session.undo():
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
        banner["chat template"] = f"config ({cfg.chat_template})"
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
    if is_diffusion:
        raise NotImplementedError(
            "--chat does not support diffusion models yet."
            " Use the default inference mode instead."
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

    generator = CausalTurnGenerator(model, tokenizer, chat_template_str, cfg.device)
    repl = ChatRepl(generator=generator, banner=_build_banner(cfg))
    repl.run()
