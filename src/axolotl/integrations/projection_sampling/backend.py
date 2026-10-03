"""Inference backend contract and lazy loading for projection sampling."""

from abc import ABC, abstractmethod
from contextlib import contextmanager, nullcontext
from importlib import import_module
from typing import Any, ContextManager, Iterator, Protocol

from .args import ProjectionSamplingConfig


class SamplingTokenizer(Protocol):
    """Tokenization operations needed by the backend-independent sampler."""

    @property
    def bos_token_id(self) -> int | None: ...

    @property
    def eos_token_id(self) -> int | None: ...

    @property
    def pad_token_id(self) -> int | None: ...

    def encode(self, text: str, **kwargs: Any) -> list[int]: ...

    def decode(self, tokens: list[int], **kwargs: Any) -> str: ...

    def apply_chat_template(
        self, messages: list[dict[str, str]], **kwargs: Any
    ) -> Any: ...


class SamplingBackend(ABC):
    """Generate proposals and score exact conditional sequence log densities.

    Tokenization must match the SFT tokenizer. Scores sum over the supplied
    continuation only, including EOS, without length normalization. Proposal
    scores must use the same distribution as sample(), including temperature
    and penalties. Target scores use the unmodified base-model distribution.
    """

    tokenizer: SamplingTokenizer
    eos_token_ids: set[int]

    @classmethod
    @abstractmethod
    def from_config(
        cls, cfg: Any, config: ProjectionSamplingConfig
    ) -> "SamplingBackend":
        """Load a backend; clean up partial resources if initialization fails."""

    @classmethod
    def rng_context(cls, cfg: Any, config: ProjectionSamplingConfig) -> ContextManager:
        """Scope backend RNG state when the inference runtime needs it."""
        return nullcontext()

    @abstractmethod
    def sample(self, context: list[int], max_tokens: int) -> list[int]:
        """Extend context by at most max_tokens, stopping only on EOS."""

    @abstractmethod
    def target_logprob(self, context: list[int], tokens: list[int]) -> float:
        """Sum base-model log probabilities of tokens conditioned on context."""

    @abstractmethod
    def proposal_logprob(self, context: list[int], tokens: list[int]) -> float:
        """Sum log probabilities under the distribution used by sample()."""

    @abstractmethod
    def close(self) -> None:
        """Release owned resources; repeated calls must be safe."""


BACKENDS = {
    "transformers": "axolotl.integrations.projection_sampling.backends.transformers.TransformersBackend",
    "vllm": "axolotl.integrations.projection_sampling.backends.vllm.VLLMBackend",
}


def resolve_backend(name: str) -> type[SamplingBackend]:
    """Resolve a bundled alias or an external backend's dotted class path."""
    target = BACKENDS.get(name, name)
    module, separator, class_name = target.rpartition(".")
    if not separator:
        raise ValueError(
            f"Unknown projection sampling backend {name!r}; use {', '.join(BACKENDS)} or a dotted SamplingBackend subclass"
        )
    backend = getattr(import_module(module), class_name)
    if not isinstance(backend, type) or not issubclass(backend, SamplingBackend):
        raise TypeError(f"Backend {target!r} must subclass SamplingBackend")
    return backend


@contextmanager
def load_backend(
    cfg: Any, config: ProjectionSamplingConfig
) -> Iterator[SamplingBackend]:
    """Own the backend lifecycle and close it on successful or failed sampling."""
    backend_cls = resolve_backend(config.backend)
    with backend_cls.rng_context(cfg, config):
        backend = backend_cls.from_config(cfg, config)
        try:
            yield backend
        finally:
            backend.close()
