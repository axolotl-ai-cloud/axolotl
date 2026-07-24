"""Base interface for per-architecture model support descriptors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Generic, Mapping, TypeVar, Union, cast

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from axolotl.processing_strategies import ProcessingStrategy
    from axolotl.utils.dict import DictDefault

    from .profile import ModelHookPhase, ModelProfile

LOG = get_logger(__name__)


@dataclass(frozen=True)
class Supported:
    """The feature is known to work for this architecture."""

    note: str = ""


@dataclass(frozen=True)
class Unsupported:
    """The feature is known-broken; enabling it raises, citing `reason`."""

    reason: str = ""


@dataclass(frozen=True)
class Experimental:
    """The feature runs but is unverified; enabling it warns, citing `note`."""

    note: str = ""


Capability = Union[Supported, Unsupported, Experimental]
_T = TypeVar("_T")


class _ProfileProjection(Generic[_T]):
    """Expose profile values without blocking legacy attribute overrides."""

    def __init__(self, name: str, default: _T):
        self.name = name
        self.default = default

    def __get__(
        self,
        instance: ModelSupport | None,
        owner: type[ModelSupport] | None = None,
    ) -> _T:
        target: ModelSupport | type[ModelSupport] | None = (
            instance if instance is not None else owner
        )
        if target is None:
            return self.default

        from .profile import _resolve_declarative_model_support

        resolved = _resolve_declarative_model_support(target)
        return cast(_T, getattr(resolved, self.name))


def check_capability(
    support: "ModelSupport | None",
    name: str,
    model_type: str | None,
    *,
    feature: str | None = None,
    hint: str = "",
) -> None:
    """Enforce a declared capability: raise on `Unsupported`, warn on `Experimental`.

    A missing descriptor or capability key is a no-op — unknown means the
    feature applies its generic fallback handling.
    """
    if support is not None:
        from .profile import resolve_model_support

        resolved = resolve_model_support(support)
        cap = resolved.capabilities.get(name)
    else:
        cap = None
    feature = feature or name
    if isinstance(cap, Unsupported):
        raise ValueError(
            f"{feature} is not supported for model_type={model_type}."
            + (f" {cap.reason}" if cap.reason else "")
            + (f" {hint}" if hint else "")
        )
    if isinstance(cap, Experimental):
        LOG.warning_once(
            "%s is experimental for model_type=%s.%s",
            feature,
            model_type,
            f" {cap.note}" if cap.note else "",
        )


class ModelSupport:
    """Everything axolotl needs to know about one model architecture, in one place.

    New descriptors set a declarative ``profile`` and register with
    ``@register_model_support``. Legacy class attributes and overridden methods
    remain supported while architectures migrate. Descriptors activate
    implicitly from the Hugging Face ``config.model_type``.

    `capabilities` maps a feature name to its support state: `Unsupported`
    raises when the feature is enabled, `Experimental` warns, `Supported`
    documents verified coverage, and a missing key means unknown — the feature
    falls back to its generic handling (e.g. CCE's llama-like patch). Features
    consume the mapping via `check_capability`.

    Profile hooks are exposed through the legacy methods for compatibility.
    Keep imperative patches localized to the model-support package.
    """

    model_types: ClassVar[tuple[str, ...]] = ()
    profile: ClassVar[ModelProfile | None] = None
    is_multimodal: ClassVar[bool] = cast(
        bool,
        _ProfileProjection("is_multimodal", False),
    )
    capabilities: ClassVar[Mapping[str, Capability]] = cast(
        Mapping[str, Capability],
        _ProfileProjection("capabilities", {}),
    )

    def get_auto_model_cls(self) -> type | None:
        """AutoModel class selected by a declarative profile, if present."""
        from .profile import _resolve_declarative_model_support

        resolved = _resolve_declarative_model_support(self)
        provider = resolved.strategies.auto_model_cls
        return provider() if provider is not None else None

    def get_processing_strategy_cls(self) -> type["ProcessingStrategy"] | None:
        """ProcessingStrategy class for the multimodal collator."""
        from .profile import _resolve_declarative_model_support

        resolved = _resolve_declarative_model_support(self)
        provider = resolved.strategies.processing_strategy_cls
        return provider() if provider is not None else None

    def matches_processor(self, processor: "ProcessorMixin") -> bool:
        """Whether this descriptor owns the given multimodal processor."""
        from .profile import _resolve_declarative_model_support

        resolved = _resolve_declarative_model_support(self)
        matcher = resolved.matchers.processor
        return matcher(processor) if matcher is not None else False

    def matches_cfg(self, cfg: "DictDefault") -> bool:
        """Whether this descriptor owns the run before ``model_type`` is known.

        Needed by architectures whose config/tokenizer loading itself must be
        patched (remote-code models, or modeling code shipped in-tree) —
        typically implemented as a name match on ``cfg.base_model_config``.
        """
        from .profile import _resolve_declarative_model_support

        resolved = _resolve_declarative_model_support(self)
        matcher = resolved.matchers.cfg
        return matcher(cfg) if matcher is not None else False

    def _run_profile_hook(
        self,
        phase: "ModelHookPhase",
        cfg: "DictDefault",
        model: "PreTrainedModel | None" = None,
    ) -> None:
        from .profile import ModelHookContext, _run_model_profile_hooks

        _run_model_profile_hooks(
            self,
            phase,
            ModelHookContext(cfg=cfg, model=model),
        )

    def validate_cfg(self, cfg: "DictDefault") -> None:
        """Model-specific config validation; raise ValueError on bad combos."""
        from .profile import ModelHookPhase

        self._run_profile_hook(ModelHookPhase.CONFIGURE_RUN, cfg)

    def pre_config_load(self, cfg: "DictDefault") -> None:
        """Patch before ``AutoConfig.from_pretrained``; dispatched via
        `matches_cfg` since ``model_type`` is not yet known."""
        from .profile import ModelHookPhase

        self._run_profile_hook(ModelHookPhase.BEFORE_CONFIG_LOAD, cfg)

    def pre_tokenizer_load(self, cfg: "DictDefault") -> None:
        """Patch before ``AutoTokenizer.from_pretrained``; dispatched via
        `matches_cfg`."""
        from .profile import ModelHookPhase

        self._run_profile_hook(ModelHookPhase.BEFORE_TOKENIZER_LOAD, cfg)

    def pre_model_load(self, cfg: "DictDefault") -> None:
        """Apply model-specific patches before checkpoint load."""
        from .profile import ModelHookPhase

        self._run_profile_hook(ModelHookPhase.BEFORE_MODEL_BUILD, cfg)

    def post_model_load(self, cfg: "DictDefault", model: "PreTrainedModel") -> None:
        """Adjust the adapter-wrapped model before generic post-load patches."""
        from .profile import ModelHookPhase

        self._run_profile_hook(ModelHookPhase.AFTER_ADAPTER_LOAD, cfg, model)
