"""Strict, regex-enumerated Linear → `TernaryLinear` surgery and its manifest."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

from torch import nn
from transformers import PreTrainedModel

from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process
from axolotl.utils.logging import get_logger

from .args import TernaryConfig, resolve_ternary_config
from .modules import TernaryLinear
from .ptq import initialize_model_latents

LOG = get_logger(__name__)

MANIFEST_FILENAME: str = "ternary_manifest.json"
MANIFEST_ATTR: str = "_ternary_manifest"

# never ternarized, whatever the preset or user lists say
ALWAYS_KEEP_FP: tuple[str, ...] = (
    r"lm_head",
    r".*\.lm_head",
    r".*embed_tokens",
    r".*embed_out",
)

# how many offending module names an enumeration error spells out
_MAX_REPORTED: int = 10


@dataclass(frozen=True)
class ArchPreset:
    """Default target / keep regexes for one `model_type`."""

    target_modules: tuple[str, ...]
    keep_fp_modules: tuple[str, ...] = ()


ARCH_PRESETS: dict[str, ArchPreset] = {
    "llama": ArchPreset(
        target_modules=(
            r".*\.self_attn\.(q|k|v|o)_proj",
            r".*\.mlp\.(gate|up|down)_proj",
        ),
    ),
    "qwen3": ArchPreset(
        target_modules=(
            r".*\.self_attn\.(q|k|v|o)_proj",
            r".*\.mlp\.(gate|up|down)_proj",
        ),
    ),
}


@dataclass
class SwapEntry:
    """One swapped Linear."""

    name: str
    in_features: int
    out_features: int
    family: str
    weight_scale: str
    group_size: int | None = None


@dataclass
class SwapManifest:
    """What the swap did, consumed by monitoring, bake, export and the parity gates."""

    model_type: str
    entries: list[SwapEntry] = field(default_factory=list)
    kept_fp: list[str] = field(default_factory=list)
    weight_scale: str = "absmean"
    group_size: int | None = None
    activation_bits: int | None = 8
    init: str = "absmean"
    subln: bool = False
    subln_modules: list[str] = field(default_factory=list)
    final_lambda: float | None = None
    quantizer: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Return a JSON-serializable view of the manifest."""
        return asdict(self)

    def save(self, output_dir: str | Path) -> Path:
        """Write `MANIFEST_FILENAME` into `output_dir` and return its path."""
        path = Path(output_dir) / MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return path

    @classmethod
    def load(cls, output_dir: str | Path) -> "SwapManifest":
        """Read `MANIFEST_FILENAME` from `output_dir`.

        Raises:
            FileNotFoundError: If the directory holds no manifest.
        """
        path = Path(output_dir) / MANIFEST_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                f"no {MANIFEST_FILENAME} in {output_dir}; that directory was not "
                "written by a ternary conversion run"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        entries = [SwapEntry(**entry) for entry in data.pop("entries", [])]
        return cls(entries=entries, **data)


def module_family(name: str) -> str:
    """Return the monitoring family of a module name (its last dotted segment, e.g. `q_proj`)."""
    return name.rpartition(".")[2]


def resolve_preset(model_type: str) -> ArchPreset:
    """Return the preset for `model_type`.

    Raises:
        ValueError: If the architecture has no preset and no explicit
            `ternary.target_modules` was configured.
    """
    preset = ARCH_PRESETS.get(model_type)
    if preset is None:
        raise ValueError(
            f"no ternary arch preset for model_type {model_type!r} "
            f"(known: {sorted(ARCH_PRESETS)}); set ternary.target_modules explicitly"
        )
    return preset


def convert_model(model: PreTrainedModel, cfg: DictDefault) -> SwapManifest:
    """Replace every enumerated `nn.Linear` in `model` with a `TernaryLinear`, in place.

    Walks `named_modules()` and fullmatches each Linear against the resolved target
    and keep lists. Under `strict_enumeration` (the default) a Linear that matches
    neither list, or both, is a hard error — silent misses are how MoE routers get
    ternarized by accident.

    Sub-norms (`ternary.subln`) are inserted on the enumerated Linears before they are
    replaced, and the configured PTQ initializer rewrites the latents afterwards.

    Args:
        model: The freshly built model, before any parallelism wrapping.
        cfg: The axolotl config; `ternary.*` is read through `resolve_ternary_config`.

    Returns:
        The manifest, also attached to the model as `MANIFEST_ATTR`.

    Raises:
        ValueError: On unmatched or doubly-matched Linears, or a biased target.
        NotImplementedError: When the architecture stores experts as fused 3D
            parameter stacks, or a configured option is not implemented yet.
    """
    # local import so the subln seam stays monkeypatchable as a module attribute
    from .subln import assert_subln_reloadable

    ternary_cfg = resolve_ternary_config(cfg)
    # runs whichever way `ternary.subln` is set: the case it catches is a marked
    # master reloaded by a run that would silently drop its gains
    assert_subln_reloadable(model, cfg)
    assert_scale_mode_reloadable(model, cfg)
    model_type = (
        getattr(getattr(model, "config", None), "model_type", None) or "unknown"
    )

    targets = list(ternary_cfg.target_modules or ())
    keeps = list(ternary_cfg.keep_fp_modules or ())
    if not targets:
        preset = resolve_preset(model_type)
        targets = list(preset.target_modules)
        keeps.extend(preset.keep_fp_modules)

    target_res = [re.compile(pattern) for pattern in targets]
    keep_res = [re.compile(pattern) for pattern in keeps]
    protected_res = [re.compile(pattern) for pattern in ALWAYS_KEEP_FP]

    fused_experts = _fused_expert_stacks(model)
    if fused_experts:
        raise NotImplementedError(
            "ternary conversion of fused MoE expert stacks is not supported yet; "
            f"3D expert parameters live in {_summarize(fused_experts)}"
        )

    to_swap: list[tuple[str, nn.Linear]] = []
    kept_fp: list[str] = []
    unmatched: list[str] = []
    double_matched: list[str] = []
    protected_targets: list[str] = []
    biased: list[str] = []

    for name, module in model.named_modules():
        if isinstance(module, TernaryLinear):
            raise ValueError(
                f"{name} is already a TernaryLinear; convert_model runs once per model"
            )
        if not isinstance(module, nn.Linear):
            continue

        is_target = any(pattern.fullmatch(name) for pattern in target_res)
        if any(pattern.fullmatch(name) for pattern in protected_res):
            if is_target:
                protected_targets.append(name)
            kept_fp.append(name)
            continue
        is_keep = any(pattern.fullmatch(name) for pattern in keep_res)

        if is_target and is_keep:
            double_matched.append(name)
            kept_fp.append(name)
        elif is_target:
            if module.bias is not None:
                biased.append(name)
            to_swap.append((name, module))
        elif is_keep:
            kept_fp.append(name)
        else:
            unmatched.append(name)
            kept_fp.append(name)

    if protected_targets:
        raise ValueError(
            f"ternary.target_modules matches {_summarize(protected_targets)}; "
            "embeddings and the lm_head always stay full precision"
        )
    if biased:
        raise ValueError(
            f"ternary conversion does not support biased Linears: {_summarize(biased)}; "
            "add them to ternary.keep_fp_modules to keep them full precision"
        )
    if double_matched:
        if ternary_cfg.strict_enumeration:
            raise ValueError(
                f"{_summarize(double_matched)} match both ternary.target_modules and "
                "ternary.keep_fp_modules; make the two lists disjoint"
            )
        LOG.warning(
            f"ternary: {_summarize(double_matched)} match both target and keep lists; "
            "keeping them full precision"
        )
    if unmatched:
        if ternary_cfg.strict_enumeration:
            raise ValueError(
                f"{_summarize(unmatched)} match neither ternary.target_modules nor "
                "ternary.keep_fp_modules; list every Linear explicitly or set "
                "ternary.strict_enumeration: false to keep the rest full precision"
            )
        LOG.warning(
            f"ternary: keeping {_summarize(unmatched)} full precision (unenumerated)"
        )

    subln_modules: list[str] = []
    if ternary_cfg.subln:
        from .subln import insert_subln

        subln_modules = insert_subln(model, cfg, [name for name, _ in to_swap])

    entries: list[SwapEntry] = []
    for name, linear in to_swap:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(
            parent,
            attr,
            TernaryLinear.from_linear(
                linear,
                weight_scale=ternary_cfg.weight_scale,
                group_size=ternary_cfg.group_size,
                activation_bits=ternary_cfg.activation_bits,
                fused=ternary_cfg.fused_fake_quant,
                int8_forward=ternary_cfg.int8_forward,
            ),
        )
        entries.append(
            SwapEntry(
                name=name,
                in_features=linear.in_features,
                out_features=linear.out_features,
                family=module_family(name),
                weight_scale=ternary_cfg.weight_scale,
                group_size=ternary_cfg.group_size,
            )
        )

    manifest = SwapManifest(
        model_type=model_type,
        entries=entries,
        kept_fp=kept_fp,
        weight_scale=ternary_cfg.weight_scale,
        group_size=ternary_cfg.group_size,
        activation_bits=ternary_cfg.activation_bits,
        init=ternary_cfg.init,
        subln=ternary_cfg.subln,
        subln_modules=subln_modules,
        quantizer=_quantizer_identity(ternary_cfg),
    )
    setattr(model, MANIFEST_ATTR, manifest)
    initialize_model_latents(model, manifest, cfg)
    LOG.info(
        f"ternary: swapped {len(entries)} Linear modules to ternary, "
        f"{len(kept_fp)} kept full precision"
    )

    output_dir = cfg.get("output_dir") if hasattr(cfg, "get") else None
    if output_dir and is_main_process():
        manifest.save(output_dir)

    return manifest


def get_manifest(model: nn.Module) -> SwapManifest | None:
    """Return the manifest attached to `model` by `convert_model`, if any."""
    return getattr(model, MANIFEST_ATTR, None)


def assert_scale_mode_reloadable(model: PreTrainedModel, cfg: DictDefault) -> None:
    """Refuse a master whose saved grid is not the grid this run would read it on.

    A baked master only re-quantizes to itself under the scale mode it was baked with:
    the per-tensor probe reads a per-row or five-value master as ordinary FP values,
    the `baked` flag never lights, and the run silently re-quantizes the master onto
    the wrong grid instead of continuing from it.

    Raises:
        ValueError: If the checkpoint directory holds a manifest whose scale grid
            disagrees with the resolved `ternary` config.
    """
    from .subln import checkpoint_dir

    source = checkpoint_dir(model, cfg)
    if source is None or not (source / MANIFEST_FILENAME).is_file():
        return
    saved = SwapManifest.load(source)
    ternary_cfg = resolve_ternary_config(cfg)
    if (saved.weight_scale, saved.group_size) == (
        ternary_cfg.weight_scale,
        ternary_cfg.group_size,
    ):
        return
    raise ValueError(
        f"{source / MANIFEST_FILENAME} says this master was baked on the "
        f"{_grid(saved.weight_scale, saved.group_size)} grid, but this run configures "
        f"{_grid(ternary_cfg.weight_scale, ternary_cfg.group_size)}. Reading a master "
        "through the wrong grid does not fail loudly — the baked probe simply misses "
        "and every weight is re-quantized onto the new grid, which is a different "
        f"model. Set ternary.weight_scale: {saved.weight_scale}"
        + (
            f" and ternary.group_size: {saved.group_size}"
            if saved.group_size is not None
            else ""
        )
        + ", or start from the original full-precision checkpoint"
    )


def _grid(weight_scale: str, group_size: int | None) -> str:
    return weight_scale if group_size is None else f"{weight_scale}/{group_size}"


def _fused_expert_stacks(model: nn.Module) -> list[str]:
    """Return the names of modules holding experts as fused 3D parameter stacks."""
    stacks = []
    for name, module in model.named_modules():
        if "expert" not in f"{name}.{type(module).__name__}".lower():
            continue
        if any(param.ndim == 3 for param in module.parameters(recurse=False)):
            stacks.append(name)
    return stacks


def _quantizer_identity(ternary_cfg: TernaryConfig) -> dict:
    """Return the quantizer stamp recorded in the manifest for export and parity."""
    dual = ternary_cfg.weight_scale == "dual"
    return {
        "scheme": "dual_ternary" if dual else "ternary",
        "codes": [-2, -1, 0, 1, 2] if dual else [-1, 0, 1],
        "rounding": "half_even",
        "weight_dequant_scale": "f16_round",
        "activation": "per_token_int8" if ternary_cfg.activation_bits else "none",
        "version": 1,
    }


def _summarize(names: list[str]) -> str:
    head = ", ".join(names[:_MAX_REPORTED])
    extra = len(names) - _MAX_REPORTED
    return f"{head} (+{extra} more)" if extra > 0 else head


__all__ = [
    "ALWAYS_KEEP_FP",
    "ARCH_PRESETS",
    "ArchPreset",
    "MANIFEST_ATTR",
    "MANIFEST_FILENAME",
    "SwapEntry",
    "SwapManifest",
    "assert_scale_mode_reloadable",
    "convert_model",
    "get_manifest",
    "module_family",
    "resolve_preset",
]
