"""Base interface for per-architecture model support descriptors."""

from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from transformers import PreTrainedModel, ProcessorMixin

    from axolotl.processing_strategies import ProcessingStrategy
    from axolotl.utils.dict import DictDefault


class ModelSupport:
    """Everything axolotl needs to know about one model architecture, in one place.

    Subclass, set the class attributes, and register with
    ``@register_model_support``. Descriptors are looked up by
    ``cfg.model_config_type`` (the HF ``config.model_type``) and activate
    implicitly — users never opt in via YAML.

    Capability flags are tri-state: ``True`` means supported, ``False`` means
    the feature raises a helpful error when enabled for this model, and
    ``None`` means unknown — the feature falls back to its generic handling
    (e.g. CCE's llama-like patch).

    The hooks default to no-ops; override them for imperative model-specific
    patches instead of adding branches to ``PatchManager`` or the trainer.
    """

    model_types: ClassVar[tuple[str, ...]] = ()

    is_multimodal: ClassVar[bool] = False
    supports_cut_cross_entropy: ClassVar[bool | None] = None
    supports_liger: ClassVar[bool | None] = None

    # optional per-capability explanation appended to "not supported" errors,
    # keyed by capability name without the `supports_` prefix
    unsupported_reasons: ClassVar[dict[str, str]] = {}

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

    def unsupported_reason(self, capability: str) -> str:
        reason = self.unsupported_reasons.get(capability, "")
        return f" {reason}" if reason else ""
