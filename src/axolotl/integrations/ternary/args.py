"""Config args for ternary (1.58-bit) conversion (nested under `ternary:`)."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import (
    BaseModel,
    Field,
    SerializerFunctionWrapHandler,
    model_serializer,
    model_validator,
)

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

WeightScaleMode = Literal["absmean", "group", "learnable"]
LambdaSchedule = Literal["linear", "sigmoid", "none"]
InitMode = Literal["absmean", "ptq_itf", "svid"]
DistillMode = Literal["kd_plugin", "inprocess"]
ExportFormat = Literal["master_bf16", "hf_bitnet", "gguf_tq2_0", "gguf_tq1_0", "i2_s"]

PER_TENSOR_SCALE_FORMATS: frozenset[str] = frozenset(
    {"hf_bitnet", "gguf_tq2_0", "gguf_tq1_0", "i2_s"}
)
GGUF_FORMATS: frozenset[str] = frozenset({"gguf_tq2_0", "gguf_tq1_0", "i2_s"})

# below this a group holds too few weights for a shared scale to quantize anything
MIN_GROUP_SIZE: int = 32
DEFAULT_EXPORT_FORMATS: tuple[ExportFormat, ...] = ("master_bf16", "hf_bitnet")

# module names a router regex would grab if it is written unanchored (`.*gate.*`)
ROUTER_CANARY_NAMES: tuple[str, ...] = (
    "model.layers.0.mlp.gate",
    "model.layers.0.block_sparse_moe.gate",
)

# keys on the merged config that cannot coexist with a ternary conversion run
CONFLICTING_KEYS: tuple[str, ...] = (
    "use_onebitllms",
    "qat",
    "quantization",
    "load_in_8bit",
    "load_in_4bit",
)


class TernaryDistillConfig(BaseModel):
    """Teacher/distillation ("healing") losses under `ternary.distill`."""

    mode: DistillMode | None = Field(
        default=None,
        description=(
            "Distillation path: null (pure QAT), 'kd_plugin' (compose with the KD "
            "integration and a served/offline teacher), or 'inprocess' (frozen teacher "
            "copy loaded beside the student)."
        ),
    )
    teacher_model: str | None = Field(
        default=None,
        description="Teacher checkpoint; defaults to `base_model` (the unquantized base).",
    )
    teacher_device_map: str | None = Field(
        default=None,
        description=(
            "`device_map` for the in-process teacher (e.g. 'auto', 'cpu') so a teacher "
            "that does not fit beside the student can be offloaded; null keeps it on "
            "the student's device."
        ),
    )
    logits_weight: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description=(
            "Mixing coefficient between CE and logits-KD: `(1 - w)·CE + w·τ²·KL`."
        ),
    )
    logits_temperature: float = Field(
        default=2.0, gt=0.0, description="Softmax temperature for the logits-KD term."
    )
    hidden_weight: float = Field(
        default=0.0,
        ge=0.0,
        description="Weight of the cosine hidden-state feature-KD term (0 disables).",
    )
    attn_relation_layer: int | None = Field(
        default=None,
        description=(
            "Experimental: layer index for single-layer MiniLM-style attention-relation "
            "KD; null disables it. Negative indexes count from the last layer. Results "
            "are sensitive to the layer choice."
        ),
    )

    @model_validator(mode="after")
    def _validate_distill_mode(self) -> "TernaryDistillConfig":
        if self.mode is None:
            if self.teacher_model:
                raise ValueError(
                    "ternary.distill.teacher_model requires ternary.distill.mode"
                )
            return self
        if self.mode != "inprocess":
            if self.hidden_weight > 0.0:
                raise ValueError(
                    "ternary.distill.hidden_weight is only supported with "
                    "ternary.distill.mode: inprocess"
                )
            if self.attn_relation_layer is not None:
                raise ValueError(
                    "ternary.distill.attn_relation_layer is only supported with "
                    "ternary.distill.mode: inprocess"
                )
        return self


class TernaryExportConfig(BaseModel):
    """Export artifacts written after training, under `ternary.export`."""

    formats: list[ExportFormat] = Field(
        default_factory=lambda: list(DEFAULT_EXPORT_FORMATS),
        description=(
            "Artifacts to emit after training. 'master_bf16' is the baked "
            "{-s, 0, +s} bf16 checkpoint; the rest are packed deployment formats."
        ),
    )
    run_parity_gate: bool = Field(
        default=True,
        description=(
            "Verify each packed export against the baked master (exact code equality "
            "plus a dequant-error bound) and fail the export when it does not match."
        ),
    )

    @model_validator(mode="after")
    def _validate_formats(self) -> "TernaryExportConfig":
        duplicates = {fmt for fmt in self.formats if self.formats.count(fmt) > 1}
        if duplicates:
            raise ValueError(
                f"duplicate ternary.export.formats entries: {sorted(duplicates)}"
            )
        return self


class TernaryConfig(BaseModel):
    """Nested ternary conversion configuration available under the `ternary` key."""

    target_modules: list[str] | None = Field(
        default=None,
        description=(
            "Regexes fullmatched against `named_modules()` names; every match is "
            "replaced by a TernaryLinear. Defaults to the architecture preset."
        ),
    )
    keep_fp_modules: list[str] | None = Field(
        default=None,
        description=(
            "Regexes for Linears that stay full precision. Embeddings, lm_head and "
            "norms are always kept regardless of this list."
        ),
    )
    strict_enumeration: bool = Field(
        default=True,
        description=(
            "Require every nn.Linear to fullmatch exactly one of the target/keep "
            "lists; unmatched or doubly-matched modules are a hard error."
        ),
    )

    weight_scale: WeightScaleMode = Field(
        default="absmean",
        description=(
            "Weight scale mode: 'absmean' (per-tensor, the only mode every deployment "
            "format can represent), 'group' (per group_size block), 'learnable' "
            "(per-tensor scale trained jointly)."
        ),
    )
    group_size: int | None = Field(
        default=None,
        ge=MIN_GROUP_SIZE,
        description=(
            f"Block size for `weight_scale: group`, at least {MIN_GROUP_SIZE} (128 or "
            "256 are the usual values); export-limited (see validators). Small groups "
            "make the quantizer a no-op — at group_size 1 every value is its own scale."
        ),
    )
    activation_bits: Literal[8] | None = Field(
        default=8,
        description="8 for per-token int8 activation quantization, null for weight-only QAT.",
    )
    lambda_schedule: LambdaSchedule = Field(
        default="linear",
        description="How the quantization-strength λ ramps from 0 to 1 during training.",
    )
    lambda_warmup_steps: int | float = Field(
        default=1000,
        description=(
            "λ warmup length. The type decides: an int is an absolute step count, a "
            "float in (0, 1] is a fraction of max_steps — so 1 is one step and 1.0 is "
            "the whole run."
        ),
    )
    weight_decay_zero_at: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of training after which weight decay is annealed to 0 so ternary "
            "code assignments can converge. 1.0 keeps weight decay for the whole run."
        ),
    )

    init: InitMode = Field(
        default="absmean",
        description="Latent-weight initialization for the swapped modules.",
    )
    smoothing: bool = Field(
        default=False,
        description="Fold a SmoothQuant-style norm→linear rescale into the weights before the swap.",
    )
    subln: bool = Field(
        default=False,
        description=(
            "Insert BitDistill-style sub-norms before o_proj/down_proj. Changes the "
            "architecture and is not representable in GGUF/I2_S exports."
        ),
    )

    distill: TernaryDistillConfig = Field(
        default_factory=TernaryDistillConfig,
        description="Distillation ('healing') losses; the teacher defaults to the FP base.",
    )

    int8_forward: Literal["auto"] | bool = Field(
        default="auto",
        description=(
            "Use the int8 tensor-core W2A8 forward once λ == 1; 'auto' enables it when "
            "the shapes and hardware qualify."
        ),
    )
    fused_fake_quant: bool = Field(
        default=True,
        description="Use the fused Triton fake-quant kernels instead of the eager oracle.",
    )
    share_act_quant: bool = Field(
        default=False,
        description=(
            "Quantize a shared input once for every linear that consumes it (q/k/v, "
            "gate/up) instead of once per linear."
        ),
    )

    log_code_flip_every: int = Field(
        default=100,
        ge=0,
        description=(
            "Log ternary code-flip rate and zero fraction every N steps; 0 disables "
            "monitoring (and its packed-code snapshot buffers)."
        ),
    )

    export: TernaryExportConfig = Field(
        default_factory=TernaryExportConfig,
        description="Artifacts written after training completes.",
    )

    @model_validator(mode="after")
    def _validate_scale_mode(self) -> "TernaryConfig":
        if self.group_size is not None and self.weight_scale != "group":
            raise ValueError(
                "ternary.group_size is only valid with ternary.weight_scale: group"
            )
        if self.weight_scale == "group" and self.group_size is None:
            raise ValueError(
                "ternary.weight_scale: group requires ternary.group_size (e.g. 128)"
            )
        return self

    @model_validator(mode="after")
    def _validate_export_compat(self) -> "TernaryConfig":
        if self.weight_scale == "group":
            unsupported = sorted(
                fmt for fmt in self.export.formats if fmt in PER_TENSOR_SCALE_FORMATS
            )
            if unsupported:
                raise ValueError(
                    f"ternary.weight_scale: group cannot be represented by {unsupported}; "
                    "those formats carry a single per-tensor scale. Export master_bf16 only."
                )
        if self.subln:
            unsupported = sorted(
                fmt for fmt in self.export.formats if fmt in GGUF_FORMATS
            )
            if unsupported:
                raise ValueError(
                    f"ternary.subln inserts sub-norms that {unsupported} have no tensor "
                    "slots for; export master_bf16 or hf_bitnet instead."
                )
        return self

    @model_validator(mode="after")
    def _validate_lambda_warmup_steps(self) -> "TernaryConfig":
        steps = self.lambda_warmup_steps
        if isinstance(steps, float):
            if not 0.0 < steps <= 1.0:
                raise ValueError(
                    "ternary.lambda_warmup_steps as a float is a fraction of max_steps "
                    "and must be in (0, 1]; use an int for an absolute step count"
                )
        elif steps < 1:
            raise ValueError("ternary.lambda_warmup_steps must be >= 1 step")
        return self

    @model_validator(mode="after")
    def _validate_module_regexes(self) -> "TernaryConfig":
        for field, patterns in (
            ("target_modules", self.target_modules),
            ("keep_fp_modules", self.keep_fp_modules),
        ):
            for pattern in patterns or []:
                try:
                    compiled = re.compile(pattern)
                except re.error as exc:
                    raise ValueError(
                        f"invalid regex in ternary.{field}: {pattern!r} ({exc})"
                    ) from exc
                if field != "target_modules":
                    continue
                for canary in ROUTER_CANARY_NAMES:
                    if compiled.fullmatch(canary):
                        LOG.warning(
                            f"ternary.target_modules pattern {pattern!r} matches the MoE "
                            f"router name {canary!r}; routers must stay full precision"
                        )
        return self

    @model_serializer(mode="wrap")
    def _serialize(self, handler: SerializerFunctionWrapHandler) -> dict[str, Any]:
        data = handler(self)
        # axolotl dumps the validated config with exclude_none, which would turn the
        # explicit `activation_bits: null` (weight-only QAT) back into the default 8
        data["activation_bits"] = self.activation_bits
        return data


class TernaryArgs(BaseModel):
    """Plugin entry that exposes the nested `ternary` block to the core config."""

    ternary: TernaryConfig = Field(
        default_factory=TernaryConfig,
        description="Ternary (1.58-bit) conversion configuration. Only nested block is supported.",
    )

    @model_validator(mode="before")
    @classmethod
    def _check_ternary_conflicts(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        if data.get("adapter"):
            raise ValueError(
                "ternary conversion is full-finetune only; remove `adapter:` "
                "(latent weights need dense gradients, there is nothing for an adapter to wrap)"
            )
        for key in CONFLICTING_KEYS:
            if data.get(key):
                raise ValueError(
                    f"`{key}:` cannot be combined with the ternary plugin; the ternary "
                    "quantizer owns the weight representation for the whole run"
                )
        return data


def resolve_ternary_config(cfg: Any) -> TernaryConfig:
    """Return the validated `TernaryConfig` for an axolotl config of any shape.

    Accepts a `DictDefault`, a plain mapping, or an object with a `ternary`
    attribute; an absent or empty block resolves to the defaults.
    """
    raw = cfg.get("ternary") if hasattr(cfg, "get") else getattr(cfg, "ternary", None)
    if isinstance(raw, TernaryConfig):
        return raw
    if not raw:
        return TernaryConfig()
    return TernaryConfig.model_validate(dict(raw))
