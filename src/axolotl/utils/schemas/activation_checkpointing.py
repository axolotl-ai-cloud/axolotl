"""Pydantic models for activation checkpointing and activation offloading"""

import inspect
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from torch.utils.checkpoint import checkpoint as _torch_checkpoint

MIGRATION_DOCS = (
    "docs/activation_checkpointing.qmd#sec-migrate-activation-checkpointing"
)

OffloadPlacement = Literal["none", "cpu", "disk"]
OFFLOAD_TARGETS = ("hidden_states", "saves", "all")

# torch passes any other key to the checkpointed function: each decoder layer's forward
TORCH_CHECKPOINT_KWARGS = tuple(
    name
    for name, param in inspect.signature(_torch_checkpoint).parameters.items()
    if param.kind is inspect.Parameter.KEYWORD_ONLY
)
# keys transformers' enable method takes by name rather than forwarding to torch
HF_CHECKPOINT_KWARGS = ("every_n_layers",)
OWNED_CHECKPOINT_KWARGS = {
    "context_fn": (
        "is set by axolotl: selective checkpointing injects its own. Use "
        "`activation_checkpointing.selective` to choose what is saved."
    ),
    "respect_saved_tensors_hooks": (
        "is set by axolotl so selective checkpointing composes with activation "
        "offloading. Remove it."
    ),
    "offload": (
        "is not supported: it copies each checkpointed layer's input to CPU "
        "synchronously. Use `activation_checkpointing.offload.hidden_states: cpu`, "
        "which offloads the same tensors with stream-overlapped copies."
    ),
}

_ACCEPTED_CHECKPOINT_KWARGS = tuple(
    name
    for name in (*TORCH_CHECKPOINT_KWARGS, *HF_CHECKPOINT_KWARGS)
    if name not in OWNED_CHECKPOINT_KWARGS and name != "use_reentrant"
)


class DiskOffloadConfig(BaseModel):
    """Settings shared by every offload target placed on disk"""

    model_config = ConfigDict(extra="forbid")

    path: str | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Directory for offloaded tensors. Defaults to a temporary directory "
                "under $TMPDIR."
            )
        },
    )


class ActivationOffloadConfig(BaseModel):
    """Where each class of tensor kept for backward lives"""

    model_config = ConfigDict(extra="forbid")

    hidden_states: OffloadPlacement = Field(
        default="none",
        json_schema_extra={
            "description": (
                "Checkpointed decoder layers' inputs. `cpu` moves them to CPU with "
                "stream-overlapped copies. With `reentrant: true` this uses the "
                "ALST-style reentrant implementation (DTensor-aware for "
                "sequence/context parallelism). `true` means `cpu`."
            )
        },
    )
    saves: OffloadPlacement = Field(
        default="none",
        json_schema_extra={
            "description": (
                "Tensors that `selective` saves. `cpu` moves them to pinned CPU "
                "memory with side-stream copies, prefetched back during backward. "
                "`true` means `cpu`."
            )
        },
    )
    all: OffloadPlacement = Field(
        default="none",
        json_schema_extra={
            "description": (
                "Every saved activation except the output head's, through TRL's "
                "offloader. With `enabled: true` the decoder layers still recompute, "
                "and this offloads their inputs plus everything saved outside them. "
                "With `enabled: false` nothing is recomputed and every activation is "
                "offloaded, which can be faster for LoRA/QLoRA. `true` means `cpu`."
            )
        },
    )
    disk: DiskOffloadConfig | None = Field(
        default=None,
        json_schema_extra={
            "description": "Settings for the targets set to `disk`.",
        },
    )

    @field_validator(*OFFLOAD_TARGETS, mode="before")
    @classmethod
    def placement_from_bool(cls, value):
        if value is True:
            return "cpu"
        if value is False or value is None:
            return "none"
        return value

    @model_validator(mode="after")
    def reject_disk(self):
        targets = [name for name in OFFLOAD_TARGETS if getattr(self, name) == "disk"]
        if targets:
            names = ", ".join(targets)
            raise ValueError(
                f"activation_checkpointing.offload.{names}: disk offload is disabled "
                "pending a fix; use `cpu`."
            )
        return self

    @model_validator(mode="after")
    def check_disk_settings(self):
        if self.disk is not None and not any(
            getattr(self, name) == "disk" for name in OFFLOAD_TARGETS
        ):
            raise ValueError(
                "`activation_checkpointing.offload.disk` configures disk offload, but "
                "no offload target is set to `disk`."
            )
        return self


class SelectiveCheckpointingConfig(BaseModel):
    """Selective activation checkpointing (SAC) configuration"""

    model_config = ConfigDict(extra="forbid")

    save: list[str] = Field(
        default_factory=lambda: ["attention"],
        json_schema_extra={
            "description": (
                "Ops to save during forward instead of recomputing in backward. "
                "'attention' matches SDPA and flash-attention forward ops; other "
                "entries are substring-matched against qualified torch op names "
                "(e.g. 'aten::mm')."
            )
        },
    )
    save_sliding_window: bool = Field(
        default=False,
        json_schema_extra={
            "description": (
                "In hybrid full/sliding-window attention models, also save "
                "sliding-window attention calls. Default false: SWA is cheap to "
                "recompute, so only full-attention calls are saved."
            )
        },
    )
    recompute_layer_types: list[str] | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Layer types (config.layer_types values) whose attention is "
                "recomputed instead of saved. Defaults to ['sliding_attention', "
                "'chunked_attention']. Ignored when save_sliding_window is true. "
                "Linear-attention layers never dispatch a matchable attention op, "
                "so they need no entry."
            )
        },
    )
    save_modules: list[str] | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Module names whose base matmul outputs are saved (module scope). "
                "Each entry matches model.named_modules() of the model before the "
                "adapter is applied: exact name, a trailing "
                "dotted suffix (e.g. 'down_proj', 'self_attn.o_proj'), or a glob "
                "when it contains *, ? or [. A PEFT-wrapped match resolves to its "
                "base_layer, so LoRA A/B matmuls are not saved. Incompatible with "
                "the fused LoRA kernels covering those modules."
            )
        },
    )
    save_matmul_min_k: int | None = Field(
        default=None,
        ge=1,
        json_schema_extra={
            "description": (
                "Save every matmul whose contraction dim K (a linear's "
                "in_features) is >= this value. Also matches MoE expert "
                "projections (aten::_grouped_mm) and bnb 4-bit gemms. A layer's "
                "last projection (e.g. down_proj feeding only a residual add) is "
                "never replayed by recompute, so saving it costs memory for no "
                "speedup. Incompatible with the fused LoRA kernels."
            )
        },
    )

    @model_validator(mode="before")
    @classmethod
    def reject_offload(cls, data):
        if isinstance(data, dict) and "offload" in data:
            raise ValueError(
                "`activation_checkpointing.selective.offload` is not a field; use "
                "`activation_checkpointing.offload.saves: cpu` to offload the saved "
                "tensors."
            )
        return data

    @field_validator("save_modules")
    @classmethod
    def normalize_save_modules(cls, value):
        if value is None:
            return None
        entries = [entry.strip() for entry in value]
        if any(not entry for entry in entries):
            raise ValueError(
                "activation_checkpointing.selective.save_modules entries must be "
                "non-empty strings"
            )
        return entries or None


class ActivationCheckpointingConfig(BaseModel):
    """Activation checkpointing and activation offloading configuration"""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=True,
        json_schema_extra={
            "description": (
                "Recompute each decoder layer in backward instead of keeping its "
                "activations. `activation_checkpointing: true` is shorthand for "
                "`enabled: true`."
            )
        },
    )
    implementation: Literal["hf", "fsdp"] = Field(
        default="hf",
        json_schema_extra={
            "description": (
                "`hf`: transformers' layer checkpointing. `fsdp`: FSDP2's checkpoint "
                "wrapper around each decoder layer (requires fsdp_config; "
                "non-reentrant only)."
            )
        },
    )
    reentrant: bool | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "use_reentrant for torch.utils.checkpoint. Defaults to false "
                "(non-reentrant), except true for SFT on DeepSpeed ZeRO-3."
            )
        },
    )
    kwargs: dict[str, Any] = Field(
        default_factory=dict,
        json_schema_extra={
            "description": (
                "Extra torch.utils.checkpoint.checkpoint arguments (e.g. "
                "preserve_rng_state, determinism_check), plus transformers' "
                "every_n_layers to checkpoint only every n-th decoder layer."
            )
        },
    )
    selective: SelectiveCheckpointingConfig | None = Field(
        default=None,
        json_schema_extra={
            "description": (
                "Selective activation checkpointing: within each checkpointed layer, "
                "save the listed ops during forward instead of recomputing them in "
                "backward. `true` saves attention (SDPA/flash-attention). Requires "
                "non-reentrant checkpointing."
            )
        },
    )
    offload: ActivationOffloadConfig = Field(
        default_factory=ActivationOffloadConfig,
        json_schema_extra={
            "description": (
                "Activation offloading: `none`, `cpu` or `disk` (disabled pending a "
                "fix) per class of kept tensor."
            )
        },
    )

    @field_validator("selective", mode="before")
    @classmethod
    def expand_selective(cls, value):
        if value is True:
            return {}
        if value is False:
            return None
        return value

    @model_validator(mode="after")
    def check_kwargs(self):
        for key, problem in OWNED_CHECKPOINT_KWARGS.items():
            if key in self.kwargs:
                raise ValueError(f"`activation_checkpointing.kwargs.{key}` {problem}")
        if "use_reentrant" in self.kwargs:
            use_reentrant = self.kwargs.pop("use_reentrant")
            if self.reentrant is not None and use_reentrant != self.reentrant:
                raise ValueError(
                    "`activation_checkpointing.kwargs.use_reentrant` "
                    f"({_yaml_scalar(use_reentrant)}) contradicts "
                    "`activation_checkpointing.reentrant` "
                    f"({_yaml_scalar(self.reentrant)}). Set only `reentrant`."
                )
            self.reentrant = use_reentrant
        for key in self.kwargs:
            if key not in _ACCEPTED_CHECKPOINT_KWARGS:
                raise ValueError(
                    f"`activation_checkpointing.kwargs.{key}` is not a "
                    "torch.utils.checkpoint.checkpoint argument or `every_n_layers` "
                    f"(accepted: {', '.join(_ACCEPTED_CHECKPOINT_KWARGS)}). torch would pass it "
                    "to each "
                    "checkpointed layer's forward, where it is ignored or overrides "
                    "the layer's own argument."
                )
        every_n_layers = self.kwargs.get("every_n_layers")
        if every_n_layers is not None and (
            isinstance(every_n_layers, bool)
            or not isinstance(every_n_layers, int)
            or every_n_layers < 1
        ):
            raise ValueError(
                "`activation_checkpointing.kwargs.every_n_layers` must be an integer "
                ">= 1"
            )
        return self

    @model_validator(mode="after")
    def check_recompute_dependencies(self):
        if not self.enabled:
            fields = [
                name
                for name, is_set in (
                    ("selective", self.selective is not None),
                    ("offload.hidden_states", self.offload.hidden_states != "none"),
                    ("offload.saves", self.offload.saves != "none"),
                    ("reentrant", self.reentrant is not None),
                    ("kwargs", bool(self.kwargs)),
                    ("implementation", self.implementation != "hf"),
                )
                if is_set
            ]
            if fields:
                names = ", ".join(f"`activation_checkpointing.{n}`" for n in fields)
                verb, pronoun = (
                    ("requires", "it") if len(fields) == 1 else ("require", "them")
                )
                raise ValueError(
                    f"{names} {verb} checkpointing; set "
                    f"`activation_checkpointing.enabled: true` or remove {pronoun}."
                )
            return self
        if self.selective is not None and self.reentrant is True:
            raise ValueError(
                "`activation_checkpointing.selective` requires non-reentrant "
                "checkpointing; set `activation_checkpointing.reentrant: false` or "
                "leave it unset."
            )
        if self.offload.saves != "none" and self.selective is None:
            raise ValueError(
                "`activation_checkpointing.offload.saves` offloads the tensors "
                "`selective` saves; set `activation_checkpointing.selective`."
            )
        if self.offload.hidden_states != "none" and self.offload.all != "none":
            raise ValueError(
                "`activation_checkpointing.offload.all` already offloads each "
                "checkpointed layer's input; set `offload.hidden_states: none`."
            )
        if self.implementation == "fsdp" and self.reentrant is True:
            raise ValueError(
                "`activation_checkpointing.reentrant: true` requires "
                "`implementation: hf`: FSDP's checkpoint wrapper is non-reentrant."
            )
        if self.offload.hidden_states != "none" and self.reentrant is None:
            self.reentrant = False
        return self


LEGACY_CHECKPOINTING_KEYS = (
    "gradient_checkpointing",
    "gradient_checkpointing_kwargs",
    "selective_checkpointing",
    "activation_offloading",
)
LEGACY_FSDP_CHECKPOINTING_KEYS = (
    "activation_checkpointing",
    "fsdp_activation_checkpointing",
)


def _yaml_scalar(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return repr(value)


def _render_block(block) -> str:
    if isinstance(block, bool):
        return "true" if block else "false"
    return yaml.safe_dump(
        block, default_flow_style=True, sort_keys=False, width=2**31
    ).strip()


def _join_names(names: list[str]) -> str:
    quoted = [f"`{name}`" for name in names]
    if len(quoted) == 1:
        return quoted[0]
    return f"{', '.join(quoted[:-1])} and {quoted[-1]}"


def legacy_checkpointing_replacement(
    data: dict,
) -> tuple[list[str], dict | bool, list[str]] | None:
    """The `activation_checkpointing` block that replaces a config's legacy keys.

    Returns the legacy keys found, the replacement (`True`/`False` when the shorthand
    is exact) and notes on values that have no exact replacement, or None when the
    config sets no legacy key.
    """
    found = [key for key in LEGACY_CHECKPOINTING_KEYS if data.get(key) is not None]
    fsdp_config = data.get("fsdp_config") or {}
    fsdp_key = next(
        (
            key
            for key in LEGACY_FSDP_CHECKPOINTING_KEYS
            if fsdp_config.get(key) is not None
        ),
        None,
    )
    if fsdp_key:
        found.append(f"fsdp_config.{fsdp_key}")
    if not found:
        return None

    notes = []
    offload: dict[str, str] = {}
    gc = data.get("gradient_checkpointing")
    ao = data.get("activation_offloading")
    if gc in ("offload", "offload_disk"):
        gc = True
        offload["all"] = "cpu"
    if ao in (True, "legacy", "disk"):
        offload["all"] = "cpu"
    elif ao == "hidden_states":
        offload["hidden_states"] = "cpu"

    hf = bool(gc)
    fsdp = bool(fsdp_config.get(fsdp_key)) if fsdp_key else False
    sac = data.get("selective_checkpointing")
    block: dict[str, Any] = {"enabled": hf or fsdp}
    if fsdp and not hf:
        block["implementation"] = "fsdp"

    kwargs = dict(data.get("gradient_checkpointing_kwargs") or {})
    if kwargs and not block["enabled"]:
        notes.append(
            "`gradient_checkpointing_kwargs` had no effect with checkpointing off."
        )
        kwargs = {}
    elif kwargs and not hf:
        ignored = [
            key
            for key, value in kwargs.items()
            if not (key == "use_reentrant" and value is False)
        ]
        if ignored:
            notes.append(
                "FSDP's checkpoint wrapper ignored `gradient_checkpointing_kwargs`, so "
                f"{_join_names(ignored)} had no effect."
            )
        kwargs = {}
    if "use_reentrant" in kwargs:
        block["reentrant"] = kwargs.pop("use_reentrant")
    if kwargs.pop("offload", False):
        if "all" not in offload:
            offload["hidden_states"] = "cpu"
        notes.append(
            "`offload` in the checkpoint kwargs copied each checkpointed layer's "
            "input to CPU synchronously; `offload.hidden_states: cpu` offloads the "
            "same tensors with stream-overlapped copies."
        )
    for key in ("context_fn", "respect_saved_tensors_hooks"):
        if key in kwargs:
            kwargs.pop(key)
            notes.append(f"`{key}` has no replacement: axolotl sets it.")
    unknown = [key for key in kwargs if key not in _ACCEPTED_CHECKPOINT_KWARGS]
    if unknown:
        for key in unknown:
            kwargs.pop(key)
        notes.append(
            f"{_join_names(unknown)} {'are' if len(unknown) > 1 else 'is'} not a "
            "torch.utils.checkpoint.checkpoint argument and is dropped; torch passed "
            "it to each checkpointed layer's forward."
        )
    if kwargs:
        block["kwargs"] = kwargs

    if sac:
        selective = dict(sac) if isinstance(sac, dict) else True
        if isinstance(selective, dict) and selective.pop("offload", False):
            offload["saves"] = "cpu"
        block["selective"] = selective or True

    if offload:
        block["offload"] = {
            name: offload[name] for name in OFFLOAD_TARGETS if name in offload
        }

    replacement: dict | bool = block
    if set(block) == {"enabled"}:
        replacement = block["enabled"]
    return found, replacement, notes


def _legacy_error(data: dict, found: list[str], replacement) -> str | None:
    """Why a legacy config can't be rewritten: combinations that were already rejected,
    or legacy keys next to the block."""
    plural = len(found) > 1
    pronoun = "them" if plural else "it"
    gc = data.get("gradient_checkpointing")
    ao = data.get("activation_offloading")
    fsdp_config = data.get("fsdp_config") or {}
    fsdp = any(fsdp_config.get(key) for key in LEGACY_FSDP_CHECKPOINTING_KEYS)
    if data.get("activation_checkpointing") is not None:
        return (
            f"{_join_names(found)} {'are' if plural else 'is'} deprecated and can't be "
            "combined with `activation_checkpointing`. Merge "
            f"`activation_checkpointing: {_render_block(replacement)}` into your "
            f"`activation_checkpointing` block and delete {pronoun}."
        )
    if ao == "disk" or gc == "offload_disk":
        name = (
            "activation_offloading: disk"
            if ao == "disk"
            else "gradient_checkpointing: offload_disk"
        )
        return (
            f"`{name}` is disabled pending a fix: it never wrote to disk (it ran the "
            "same CPU offloader as `activation_offloading: true`). For CPU offload use "
            f"`activation_checkpointing: {_render_block(replacement)}`."
        )
    needs_recompute = [
        name
        for name, is_set in (
            ("selective_checkpointing", bool(data.get("selective_checkpointing"))),
            ("activation_offloading: hidden_states", ao == "hidden_states"),
        )
        if is_set
    ]
    if needs_recompute and not gc:
        resolved = legacy_checkpointing_replacement(
            {**data, "gradient_checkpointing": True}
        )
        assert resolved is not None
        enabled = resolved[1]
        return (
            f"`{needs_recompute[0]}` requires gradient checkpointing. Use "
            f"`activation_checkpointing: {_render_block(enabled)}`."
        )
    if gc and fsdp:
        return (
            "`gradient_checkpointing` and `fsdp_config.activation_checkpointing` can't "
            "both be set (transformers' Trainer rejects them together). Use "
            "`activation_checkpointing: true` for transformers' checkpointing or "
            "`activation_checkpointing: {enabled: true, implementation: fsdp}` for "
            "FSDP's checkpoint wrapper."
        )
    return None


def _deprecation_message(found: list[str], replacement, notes: list[str]) -> str:
    plural = len(found) > 1
    pronoun = "them" if plural else "it"
    message = (
        f"{_join_names(found)} {'are' if plural else 'is'} deprecated and will be "
        "removed in a future release."
    )
    if replacement is False and "gradient_checkpointing" not in found:
        if found == ["activation_offloading"]:
            reason = "activation offloading is off by default"
        elif found == ["gradient_checkpointing_kwargs"]:
            reason = "checkpointing is off, so it has no effect"
            notes = []
        else:
            reason = "activation checkpointing and offloading are off by default"
        message += f" Delete {pronoun}: {reason}."
    else:
        message += (
            f" Use `activation_checkpointing: {_render_block(replacement)}` instead."
        )
    for note in notes:
        message += f" {note}"
    return f"{message} See {MIGRATION_DOCS}."


def migrate_legacy_checkpointing_keys(data: dict) -> str | None:
    """Rewrite a config's legacy checkpointing keys into `activation_checkpointing`.

    Edits ``data`` in place and returns the deprecation warning, or None when it sets no
    legacy key. Raises for combinations that were already rejected, and for legacy keys
    next to the block.
    """
    resolved = legacy_checkpointing_replacement(data)
    if resolved is None:
        return None
    found, replacement, notes = resolved
    error = _legacy_error(data, found, replacement)
    if error:
        raise ValueError(f"{error} See {MIGRATION_DOCS} for the migration guide.")
    for key in LEGACY_CHECKPOINTING_KEYS:
        data.pop(key, None)
    # type(...) keeps a DictDefault's attribute access on the values set here
    fsdp_config = data.get("fsdp_config")
    if isinstance(fsdp_config, dict):
        stripped = {
            key: value
            for key, value in fsdp_config.items()
            if key not in LEGACY_FSDP_CHECKPOINTING_KEYS
        }
        if fsdp_config and not stripped:
            # an empty fsdp_config turns FSDP off; validation mirrors the version in
            stripped["fsdp_version"] = data.get("fsdp_version") or 2
        data["fsdp_config"] = type(fsdp_config)(stripped)
    if isinstance(replacement, dict):
        data["activation_checkpointing"] = type(data)(replacement)
    elif not (replacement is False and "gradient_checkpointing" not in found):
        data["activation_checkpointing"] = replacement
    return _deprecation_message(found, replacement, notes)
