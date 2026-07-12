"""Base interface for per-architecture model support descriptors."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Union

from axolotl.utils.logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from axolotl.processing_strategies import ProcessingStrategy
    from axolotl.utils.dict import DictDefault

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
    cap = support.capabilities.get(name) if support is not None else None
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

    Subclass, set the class attributes, and register with
    ``@register_model_support``. Descriptors are looked up by
    ``cfg.model_config_type`` (the HF ``config.model_type``) and activate
    implicitly — users never opt in via YAML.

    `capabilities` maps a feature name to its support state: `Unsupported`
    raises when the feature is enabled, `Experimental` warns, `Supported`
    documents verified coverage, and a missing key means unknown — the feature
    falls back to its generic handling (e.g. CCE's llama-like patch). Features
    consume the mapping via `check_capability`.

    The hooks default to no-ops; override them for imperative model-specific
    patches instead of adding branches to ``PatchManager`` or the trainer.
    """

    model_types: ClassVar[tuple[str, ...]] = ()

    is_multimodal: ClassVar[bool] = False
    capabilities: ClassVar[dict[str, Capability]] = {}

    def get_auto_model_cls(self) -> type | None:
        """AutoModel class used to load checkpoints (multimodal models only)."""
        return None

    def get_processing_strategy_cls(self) -> type["ProcessingStrategy"] | None:
        """ProcessingStrategy class for the multimodal collator."""
        return None

    def matches_processor(self, processor: "ProcessorMixin") -> bool:
        """Whether this descriptor owns the given multimodal processor."""
        return False

    def matches_cfg(self, cfg: "DictDefault") -> bool:
        """Whether this descriptor owns the run before ``model_type`` is known.

        Needed by architectures whose config/tokenizer loading itself must be
        patched (remote-code models, or modeling code shipped in-tree) —
        typically implemented as a name match on ``cfg.base_model_config``.
        """
        return False

    def validate_cfg(self, cfg: "DictDefault") -> None:
        """Model-specific config validation; raise ValueError on bad combos."""

    def pre_config_load(self, cfg: "DictDefault") -> None:
        """Patch before ``AutoConfig.from_pretrained``; dispatched via
        `matches_cfg` since ``model_type`` is not yet known."""

    def pre_tokenizer_load(self, cfg: "DictDefault") -> None:
        """Patch before ``AutoTokenizer.from_pretrained``; dispatched via
        `matches_cfg`."""

    def pre_model_load(self, cfg: "DictDefault") -> None:
        """Apply model-specific patches before checkpoint load."""

    def post_model_load(self, cfg: "DictDefault", model: "PreTrainedModel") -> None:
        """Adjust the model instance after checkpoint load."""
