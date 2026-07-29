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

WeightScaleMode = Literal[
    "absmean", "group", "learnable", "learnable_row", "dual", "trit_planes"
]
LambdaSchedule = Literal["linear", "sigmoid", "none"]
InitMode = Literal["absmean", "ternary_fit", "ternary_fit_calibrated", "svid"]
DistillMode = Literal["kd_plugin", "inprocess"]
DistillSchedule = Literal["constant", "anchored"]
HiddenLoss = Literal["cosine", "mse", "huber"]
ExportFormat = Literal[
    "master_bf16",
    "hf_bitnet",
    "gguf_tq2_0",
    "gguf_tq1_0",
    "i2_s",
    "mask_sign",
    "onebitllms_bf16",
]
EmbeddingDtype = Literal["bf16", "int8"]

# one scale and one trit plane for the whole tensor — everything a packed format holds
PER_TENSOR_SCALE_FORMATS: frozenset[str] = frozenset(
    {"hf_bitnet", "gguf_tq2_0", "gguf_tq1_0", "i2_s", "mask_sign", "onebitllms_bf16"}
)
GGUF_FORMATS: frozenset[str] = frozenset({"gguf_tq2_0", "gguf_tq1_0", "i2_s"})

# formats with nowhere to put a `ternary.subln` gain. hf_bitnet is among them: the
# packer emits the source architecture plus a bitnet `quantization_config`, and
# transformers' BitLinear only offers an input norm through a model-wide
# `use_rms_norm` flag that would also normalize the projections subln leaves alone.
SUBLN_UNSUPPORTED_FORMATS: frozenset[str] = GGUF_FORMATS | {
    "hf_bitnet",
    "onebitllms_bf16",
}

# scale modes no packed format can carry: more than one scale per tensor ('group',
# 'learnable_row') or more than one plane ('dual')
NON_PER_TENSOR_SCALE_MODES: frozenset[str] = frozenset(
    {"group", "learnable_row", "dual", "trit_planes"}
)
# scale modes whose scale is an optimizer-visible parameter rather than a statistic
LEARNABLE_SCALE_MODES: frozenset[str] = frozenset(
    {"learnable", "learnable_row", "dual", "trit_planes"}
)

# scale modes carrying two trit planes per weight rather than one
TWO_PLANE_SCALE_MODES: frozenset[str] = frozenset({"dual", "trit_planes"})
# init modes that consume `ternary.init_calibration`
CALIBRATED_INIT_MODES: frozenset[str] = frozenset({"ternary_fit_calibrated"})
# init modes that solve a scale and write the solution into the latent
FITTING_INIT_MODES: frozenset[str] = frozenset(
    {"ternary_fit", "ternary_fit_calibrated", "svid"}
)

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


class TernaryInitCalibrationConfig(BaseModel):
    """Calibration corpus for `init: ternary_fit_calibrated`, under `ternary.init_calibration`."""

    num_sequences: int = Field(
        default=128,
        ge=1,
        description=(
            "Sequences whose activations the layer-wise Gram matrices are accumulated "
            "over. 128 is the published operating point for GPTQ-family solvers."
        ),
    )
    sequence_len: int = Field(
        default=2048,
        ge=1,
        description="Tokens per calibration sequence; longer ones are truncated.",
    )
    dataset: str | None = Field(
        default=None,
        description=(
            "Dataset path to calibrate on; null uses the training dataset. Calibrating "
            "on a corpus that does not match the healing corpus costs accuracy."
        ),
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
        default=0.5,
        ge=0.0,
        le=1.0,
        description=(
            "Mixing coefficient between CE and logits-KD: `(1 - w)·CE + w·τ²·KL`. "
            "Pure KD (1.0) matches the teacher's temperature-flattened tails at the "
            "expense of head sharpness and measures worse on perplexity for short "
            "heals; keep the CE blend unless running a long OneBit-style distill."
        ),
    )
    logits_temperature: float = Field(
        default=2.0, gt=0.0, description="Softmax temperature for the logits-KD term."
    )
    hidden_weight: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Weight of the hidden-state feature-KD term (0 disables). Retune it when "
            "changing `hidden_loss`: cosine is bounded in [0, 2] while the raw-residual "
            "losses are not, so the same weight means a different blend."
        ),
    )
    hidden_loss: HiddenLoss = Field(
        default="cosine",
        description=(
            "How the hidden-state feature-KD term compares the two residual streams. "
            "'cosine' matches direction only and ignores magnitude, which makes it "
            "robust to the outlier channels that dominate a residual stream (the "
            "TernaryLLM OFF result; OneBit's normalized MSE is the same objective up "
            "to a constant). 'mse' is raw MSE on the hidden states: magnitude-aware, "
            "which is the argument for it — the residual stream is never renormalized, "
            "so its magnitudes are what set the mixing ratio between blocks — but a "
            "handful of outlier channels then dominate the gradient. 'huber' keeps the "
            "magnitude sensitivity with the outlier influence bounded past "
            "`hidden_huber_delta`. The raw losses are on the scale of the hidden "
            "states squared, not of cosine, so `hidden_weight` needs retuning when "
            "switching rather than being renormalized here."
        ),
    )
    hidden_huber_delta: float = Field(
        default=1.0,
        gt=0.0,
        description=(
            "Elbow of `hidden_loss: huber`: residuals below it are squared, above it "
            "linear. Measured in hidden-state units, so scale it with the residual "
            "stream rather than leaving it at 1.0 for a model whose activations are "
            "much larger."
        ),
    )
    attn_relation_layer: int | None = Field(
        default=None,
        description=(
            "Experimental: layer index for single-layer MiniLM-style attention-relation "
            "KD; null disables it. Negative indexes count from the last layer. Results "
            "are sensitive to the layer choice."
        ),
    )
    schedule: DistillSchedule = Field(
        default="constant",
        description=(
            "When the KD term contributes: 'constant' blends it across the whole run, "
            "'anchored' switches it on only for the last `1 - anchor_start` of the "
            "steps (the LR-decay tail), leaving the bulk of the heal on pure CE."
        ),
    )
    anchor_start: float = Field(
        default=0.9,
        gt=0.0,
        lt=1.0,
        description=(
            "Fraction of max_steps after which `schedule: anchored` turns the KD term "
            "on. Ignored under `schedule: constant`."
        ),
    )

    @model_validator(mode="after")
    def _validate_distill_mode(self) -> "TernaryDistillConfig":
        if self.mode is None:
            if self.teacher_model:
                raise ValueError(
                    "ternary.distill.teacher_model requires ternary.distill.mode"
                )
            if self.schedule != "constant":
                raise ValueError(
                    f"ternary.distill.schedule: {self.schedule} anchors a KD term that "
                    "a run without ternary.distill.mode never computes"
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

    @model_validator(mode="after")
    def _validate_hidden_loss(self) -> "TernaryDistillConfig":
        if self.hidden_loss != "cosine" and not self.hidden_weight:
            raise ValueError(
                f"ternary.distill.hidden_loss: {self.hidden_loss} selects a term that "
                "ternary.distill.hidden_weight: 0 never computes; set a weight or drop "
                "the setting"
            )
        if self.hidden_huber_delta != 1.0 and self.hidden_loss != "huber":
            raise ValueError(
                "ternary.distill.hidden_huber_delta is the elbow of "
                f"`hidden_loss: huber`, but hidden_loss is {self.hidden_loss}"
            )
        return self


class TernaryExportConfig(BaseModel):
    """Export artifacts written after training, under `ternary.export`."""

    formats: list[ExportFormat] = Field(
        default_factory=lambda: list(DEFAULT_EXPORT_FORMATS),
        description=(
            "Artifacts to emit after training. 'master_bf16' is the baked "
            "{-s, 0, +s} bf16 checkpoint; the rest are packed deployment formats. "
            "'mask_sign' is this plugin's own sparsity-adaptive packing — a zero "
            "bitmap plus one sign bit per nonzero, so it costs `2 - zero_fraction` "
            "bits per weight and beats the 2.0 bpw formats on a sparse model."
        ),
    )
    run_parity_gate: bool = Field(
        default=True,
        description=(
            "Verify each packed export against the baked master (exact code equality "
            "plus a dequant-error bound) and fail the export when it does not match."
        ),
    )
    embedding_dtype: EmbeddingDtype = Field(
        default="bf16",
        description=(
            "Weight dtype for the token embeddings and the output head (one tensor "
            "when they are tied). 'int8' packs them as Q8_0, which roughly halves the "
            "unembedding DRAM traffic that dominates decode. GGUF formats only: the "
            "bf16 master always keeps full-precision embeddings because it is the "
            "re-finetunable artifact, and hf_bitnet has no scheme for quantized ones."
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

    @model_validator(mode="after")
    def _validate_embedding_dtype(self) -> "TernaryExportConfig":
        if self.embedding_dtype == "bf16":
            return self
        honoring = [fmt for fmt in self.formats if fmt in GGUF_FORMATS]
        if not honoring:
            raise ValueError(
                "ternary.export.embedding_dtype: int8 is only representable in the "
                f"GGUF formats {sorted(GGUF_FORMATS)}; master_bf16 always keeps "
                "full-precision embeddings (it is the re-finetunable artifact) and "
                "hf_bitnet has no quantized-embedding scheme. Add a GGUF format or "
                "drop embedding_dtype."
            )
        exempt = [fmt for fmt in self.formats if fmt not in GGUF_FORMATS]
        if exempt:
            LOG.warning(
                f"ternary: embedding_dtype: int8 applies to {sorted(honoring)}; "
                f"{sorted(exempt)} keep full-precision embeddings and the output head"
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
            "Weight scale mode: 'absmean' (per-tensor statistic), 'group' (per "
            "group_size block), 'learnable' (per-tensor scale trained jointly), "
            "'trit_planes' (PTQTP-style free sum of two trit planes, nine values per "
            "row, ~4.25 bpw, research tier: master_bf16 only), "
            "'learnable_row' (one trained scale per output channel), 'dual' (five "
            "values {0, ±s_lo, ±s_hi} from two trained per-row scales). 'absmean' and "
            "'learnable' are per-tensor and export everywhere; the rest export to "
            "master_bf16 only."
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
        description=(
            "Latent-weight initialization for the swapped modules: 'absmean' leaves "
            "the FP weights alone (the quantizer rounds them at step 0), "
            "'ternary_fit' runs a data-free alternating scale/state solve, "
            "'ternary_fit_calibrated' adds an activation-aware refit and error "
            "compensation over `init_calibration` sequences. 'svid' is planned."
        ),
    )
    init_calibration: TernaryInitCalibrationConfig = Field(
        default_factory=TernaryInitCalibrationConfig,
        description="Calibration corpus for `init: ternary_fit_calibrated`; unused by the other modes.",
    )
    smoothing: bool = Field(
        default=False,
        description="Fold a SmoothQuant-style norm→linear rescale into the weights before the swap.",
    )
    subln: bool = Field(
        default=False,
        description=(
            "Insert BitDistill-style sub-norms before o_proj/down_proj. The norms live "
            "inside the ternary linears and are saved into the bf16 master, so that "
            "master only reloads with this plugin enabled; the packed deployment "
            "formats have no tensor slots for them."
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
    def _validate_init_calibration(self) -> "TernaryConfig":
        # compared against the defaults rather than `model_fields_set`: axolotl dumps
        # and re-validates the whole config, which marks every field as set
        if (
            self.init not in CALIBRATED_INIT_MODES
            and self.init_calibration != TernaryInitCalibrationConfig()
        ):
            raise ValueError(
                "ternary.init_calibration configures `init: ternary_fit_calibrated`, "
                f"but init is {self.init!r}"
            )
        return self

    init_jitter: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Gaussian noise added to the latent after a fitting init, as a multiple of "
            "the module's quantization scale (the finer of the two under 'dual' and "
            "'trit_planes'). A fit writes the solver's own reconstruction, so every "
            "latent sits at a cell centre — half a scale from any assignment boundary "
            "— and the codes cannot flip: the heal trains scales and norms while the "
            "assignment stays frozen at the solver's output. 0.25 puts a meaningful "
            "fraction of weights within reach of a boundary; 0 keeps the exact "
            "reconstruction."
        ),
    )

    @model_validator(mode="after")
    def _validate_init_jitter(self) -> "TernaryConfig":
        if self.init_jitter and self.init not in FITTING_INIT_MODES:
            raise ValueError(
                f"ternary.init_jitter perturbs the latent a fit wrote, but "
                f"ternary.init: {self.init} leaves the full-precision weights in place "
                "— the noise would corrupt real information rather than unfreeze a "
                "solved assignment. Set a fitting init "
                f"({', '.join(sorted(FITTING_INIT_MODES))}) or drop init_jitter"
            )
        return self

    @model_validator(mode="after")
    def _validate_init_scale_mode(self) -> "TernaryConfig":
        if self.init not in FITTING_INIT_MODES:
            return self
        if self.weight_scale not in LEARNABLE_SCALE_MODES:
            raise ValueError(
                f"ternary.init: {self.init} solves a scale per tensor, but "
                f"ternary.weight_scale: {self.weight_scale} keeps no scale — it "
                "re-derives one from the latent on every forward, and the latent a fit "
                "writes is its own ternary reconstruction, whose absmean is the fitted "
                "scale times the non-zero code fraction. The solved scale would "
                "therefore survive exactly until the first optimizer step and the run "
                "would heal from a worse point than `init: absmean`. Use "
                "`weight_scale: learnable` (per-tensor, exports everywhere), "
                "`learnable_row` or `dual`"
            )
        if self.lambda_schedule != "none":
            LOG.warning(
                f"ternary.init: {self.init} writes the quantized weight into the latent, "
                f"so lambda_schedule: {self.lambda_schedule} interpolates between two "
                "identical tensors and the warmup is a no-op — the run starts fully "
                "ternary. Set lambda_schedule: none to say so, or init: absmean to get "
                "a warmup that means something"
            )
        return self

    @model_validator(mode="after")
    def _validate_export_compat(self) -> "TernaryConfig":
        if self.weight_scale in NON_PER_TENSOR_SCALE_MODES:
            unsupported = sorted(
                fmt for fmt in self.export.formats if fmt in PER_TENSOR_SCALE_FORMATS
            )
            if unsupported:
                raise ValueError(
                    f"ternary.weight_scale: {self.weight_scale} cannot be represented "
                    f"by {unsupported}; those formats carry one per-tensor scale and a "
                    "single trit plane. Export master_bf16 only."
                )
        if self.subln:
            unsupported = sorted(
                fmt for fmt in self.export.formats if fmt in SUBLN_UNSUPPORTED_FORMATS
            )
            if unsupported:
                raise ValueError(
                    f"ternary.subln inserts sub-norms that {unsupported} have no tensor "
                    "slots for; set ternary.export.formats to master_bf16 (optionally "
                    "with mask_sign)."
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
