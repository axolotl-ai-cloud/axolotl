"""Lightweight, serializable facts for diffusion-model support descriptors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class _SerializedEnum(str, Enum):
    """Enum whose value is stable in manifests and profile caches."""


class DiffusionNoise(_SerializedEnum):
    """Forward-process family used by a model."""

    UNIFORM = "uniform"
    ABSORBING = "absorbing"


class DiffusionLayout(_SerializedEnum):
    """Native model input layout."""

    ENCODER_CANVAS = "encoder_canvas"
    FULL_SEQUENCE = "full_sequence"


class LogitAlignment(_SerializedEnum):
    """How model logits align with clean canvas targets."""

    ALIGNED = "aligned"
    SHIFTED = "shifted"


class FirstPositionAlignment(_SerializedEnum):
    """Alignment rule for the first target in each logical sequence."""

    DUPLICATE_FIRST = "duplicate_first"
    REQUIRES_PREDECESSOR = "requires_predecessor"


class EosHandling(_SerializedEnum):
    """How absorbing corruption treats EOS-like token groups."""

    INDEPENDENT = "independent"
    TREAT_EOS_AS_ONE = "treat_eos_as_one"
    EXCLUDE_SPECIAL = "exclude_special"


class MaskTokenPolicy(_SerializedEnum):
    """How an absorbing model obtains its mask token."""

    NONE = "none"
    MODEL = "model"
    RESOLVE_OR_ADD = "resolve_or_add"


class TimeWeighting(_SerializedEnum):
    """Named weighting objective selected by the shared loss registry."""

    NONE = "none"
    INV_T = "inv_t"
    LINEAR = "linear"
    INV_ONE_MINUS_T = "inv_one_minus_t"
    CART = "cart"
    LOO = "loo"


class ObjectiveReduction(_SerializedEnum):
    """Reference denominator and support for the architecture's base objective."""

    SUPERVISED_TOKEN_MEAN = "supervised_token_mean"  # nosec B105 - serialized objective name.
    MASKED_TOKEN_MEAN = "masked_token_mean"  # nosec B105 - serialized objective name.
    EXAMPLE_MEAN = "example_mean"


class ReductionScope(_SerializedEnum):
    """Whether a token objective is normalized locally or across DDP workers."""

    MICROBATCH = "microbatch"
    GLOBAL_WINDOW = "global_window"


class GenerationAdapter(_SerializedEnum):
    """Stable key for the architecture-specific diffusion reader."""

    ENCODER_CANVAS = "encoder_canvas"
    DREAM = "dream"
    FULL_SEQUENCE = "full_sequence"


@dataclass(frozen=True)
class DiffusionSpec:
    """Immutable model facts consumed without importing training implementations."""

    noise: DiffusionNoise
    layout: DiffusionLayout
    logit_alignment: LogitAlignment
    first_position_alignment: FirstPositionAlignment
    self_conditioning: bool
    max_canvas: int | None
    max_context: int | None
    eos_handling: EosHandling
    mask_token_policy: MaskTokenPolicy
    default_time_weighting: TimeWeighting
    objective_reduction: ObjectiveReduction
    generation_adapter: GenerationAdapter
    time_floor: float = 0.0
    reduction_scope: ReductionScope = ReductionScope.MICROBATCH

    def __post_init__(self) -> None:
        if self.max_canvas is not None and self.max_canvas < 1:
            raise ValueError("max_canvas must be positive when set")
        if self.max_context is not None and self.max_context < 1:
            raise ValueError("max_context must be positive when set")
        if not 0.0 <= self.time_floor <= 1.0:
            raise ValueError("time_floor must be in [0, 1]")
        if (
            self.noise is DiffusionNoise.UNIFORM
            and self.mask_token_policy is not MaskTokenPolicy.NONE
        ):
            raise ValueError("uniform diffusion must not declare a mask-token policy")
        if (
            self.noise is DiffusionNoise.ABSORBING
            and self.mask_token_policy is MaskTokenPolicy.NONE
        ):
            raise ValueError("absorbing diffusion requires a mask-token policy")

    def to_dict(self) -> dict[str, object]:
        """Return primitives suitable for a run manifest."""

        return {
            name: value.value if isinstance(value, Enum) else value
            for name, value in asdict(self).items()
        }
