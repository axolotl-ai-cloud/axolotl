"""Strict, regex-enumerated Linear → `TernaryLinear` surgery and its manifest."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import cast

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from transformers import PreTrainedModel

from axolotl.utils.dict import DictDefault
from axolotl.utils.distributed import is_main_process
from axolotl.utils.logging import get_logger

from . import aux_modules
from .args import (
    UNIMPLEMENTED_CODEBOOKS,
    TernaryConfig,
    WeightScaleMode,
    resolve_ternary_config,
)
from .modules import TernaryEmbedding, TernaryLinear
from .ptq import initialize_model_latents

LOG = get_logger(__name__)

MANIFEST_FILENAME: str = "ternary_manifest.json"
SCALES_FILENAME: str = "ternary_scales.safetensors"
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
    # a hybrid: full-attention blocks interleaved with linear-attention ones, plus a
    # vision tower. Only the full-attention projections and the dense MLPs are
    # ternarized. The linear-attention projections stay FP because ternary sensitivity
    # in a linear-attention state update is unstudied — its `in_proj_a` / `in_proj_b`
    # feed a recurrent decay whose error compounds along the sequence rather than
    # being re-normalized every block, which is not the regime the ternary evidence
    # covers. The vision tower stays FP on the Prism 4-bit precedent: towers are a
    # small fraction of the parameters and quantize far worse than the LM. Both are
    # covered in the field guide; see [What not to quantize].
    "qwen3_5": ArchPreset(
        target_modules=(
            r".*\.self_attn\.(q|k|v|o)_proj",
            r".*\.mlp\.(gate|up|down)_proj",
        ),
        keep_fp_modules=(
            r".*\.linear_attn\.(in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)",
        )
        + aux_modules.family_patterns("vision_towers"),
    ),
    # the MoE sibling of `qwen3_5`: same hybrid attention split, so the
    # linear-attention projections stay FP for the same recurrent-error reason.
    # Routed experts live as fused 3D stacks the expert-stack converter discovers on
    # its own, so the preset only names the 2D linears. The router and the tiny
    # shared-expert gate stay FP: misrouting compounds across the depth the same way
    # the recurrent decay does, and both are a negligible parameter fraction.
    "qwen3_5_moe": ArchPreset(
        target_modules=(
            r".*\.self_attn\.(q|k|v|o)_proj",
            r".*\.mlp\.shared_expert\.(gate|up|down)_proj",
            # fused 3D stacks: matched as raw parameter keys by the fit-stream; the
            # live-model swap reaches them through the expert-stack converter instead
            r".*\.mlp\.experts\.(gate_up|down)_proj",
        ),
        keep_fp_modules=(
            r".*\.linear_attn\.(in_proj_qkv|in_proj_a|in_proj_b|in_proj_z|out_proj)",
            r".*\.mlp\.gate",
            r".*\.mlp\.shared_expert_gate",
        )
        + aux_modules.family_patterns("vision_towers"),
    ),
    # Mistral 3 multimodal: the language model is nested under `model.language_model`,
    # with a vision tower and a projector beside it. Only the language model's
    # projections are ternarized — the tower and projector stay FP on the same Prism
    # precedent as `qwen3_5`, and the checkpoint's own fp8 pass skipped them too, so
    # they are the parts the vendor already judged precision-sensitive.
    "mistral3": ArchPreset(
        target_modules=(
            r".*\.language_model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj",
            r".*\.language_model\.layers\.\d+\.mlp\.(gate|up|down)_proj",
        ),
        keep_fp_modules=aux_modules.family_patterns("vision_towers", "projectors"),
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
    codebook: str = "ternary"
    # "linear", "embedding" or "experts"; the packers that have no scheme for a
    # quantized embedding key off this rather than guessing from the module name
    kind: str = "linear"

    def parameter_key(self) -> str:
        """The state-dict key this entry's quantized tensor lives under.

        Linears and embeddings hang their tensor off `<module>.weight`; a fused
        expert stack IS the parameter, so its entry name is already the key. The
        consumers ask here rather than assuming — an assumed `.weight` suffix on
        an experts entry would silently miss the tensor.
        """
        if self.kind == "experts":
            return self.name
        return f"{self.name}.weight"


@dataclass
class SwapManifest:
    """What the swap did, consumed by monitoring, bake, export and the parity gates."""

    model_type: str
    entries: list[SwapEntry] = field(default_factory=list)
    kept_fp: list[str] = field(default_factory=list)
    weight_scale: str = "absmean"
    group_size: int | None = None
    codebook: str = "ternary"
    activation_bits: int | None = 8
    init: str = "absmean"
    subln: bool = False
    subln_modules: list[str] = field(default_factory=list)
    ternary_embeddings: list[str] = field(default_factory=list)
    tied_embeddings: bool = False
    final_lambda: float | None = None
    quantizer: dict = field(default_factory=dict)
    # module name -> the per-row scales its master was baked with, kept out of the
    # JSON and written beside it so the floats survive exactly
    scales: dict = field(default_factory=dict, compare=False, repr=False)

    def linear_entries(self) -> list["SwapEntry"]:
        """Entries a weight packer may quantize: the Linears, never the embeddings.

        No packed format has a scheme for a quantized embedding — BitNet converts only
        Linears, and TQ1_0/TQ2_0 would need `hidden % 256`. A ternary embedding's
        values ride into every artifact as the baked bf16/Q8_0 tensor instead, which
        loses nothing: they are already on the grid.
        """
        return [entry for entry in self.entries if entry.kind != "embedding"]

    def embedding_entries(self) -> list["SwapEntry"]:
        """Entries for ternary embeddings."""
        return [entry for entry in self.entries if entry.kind == "embedding"]

    def to_dict(self) -> dict:
        """Return a JSON-serializable view of the manifest."""
        data = asdict(self)
        data.pop("scales", None)
        data["persisted_scales"] = sorted(self.scales)
        return data

    def record_scales(self, name: str, *scales: "torch.Tensor") -> None:
        """Remember the scales a module baked with, for `save` to persist."""
        self.scales[name] = [
            scale.detach().to(torch.float32).reshape(-1).cpu() for scale in scales
        ]

    def save(self, output_dir: str | Path) -> Path:
        """Write `MANIFEST_FILENAME` into `output_dir` and return its path."""
        path = Path(output_dir) / MANIFEST_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self._save_scales(path.parent)
        return path

    def _save_scales(self, output_dir: Path) -> None:
        """Write the recorded scales beside the manifest, or clear a stale sidecar."""
        path = output_dir / SCALES_FILENAME
        if not self.scales:
            path.unlink(missing_ok=True)
            return
        save_file(
            {
                f"{name}.{index}": scale
                for name, pair in self.scales.items()
                for index, scale in enumerate(pair)
            },
            path,
        )

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
        data.pop("persisted_scales", None)
        manifest = cls(entries=entries, **data)
        manifest.scales = _load_scales(Path(output_dir) / SCALES_FILENAME)
        return manifest

    def scales_for(self, name: str) -> "tuple[torch.Tensor, torch.Tensor] | None":
        """Return the persisted per-row scales of `name`, or `None` if it has none.

        Only the two-plane grids record scales, so the pair is an invariant of every
        producer; the cast states it for the consumers that index both halves.
        """
        pair = self.scales.get(name)
        if not pair:
            return None
        return cast(
            "tuple[torch.Tensor, torch.Tensor]",
            tuple(scale.reshape(-1, 1) for scale in pair),
        )


def _load_scales(path: Path) -> dict:
    """Read the scale sidecar a master ships, if it has one."""
    if not path.is_file():
        return {}
    scales: dict[str, list] = {}
    with safe_open(path, framework="pt") as sidecar:
        for key in sorted(sidecar.keys()):  # noqa: SIM118 — handle, not a mapping
            name, _, index = key.rpartition(".")
            scales.setdefault(name, []).insert(int(index), sidecar.get_tensor(key))
    return scales


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

    if ternary_cfg.codebook in UNIMPLEMENTED_CODEBOOKS:
        raise NotImplementedError(
            f"ternary.codebook: {ternary_cfg.codebook} is accepted by the schema but "
            "its quantizer has not landed yet; use `codebook: ternary`"
        )

    targets = list(ternary_cfg.target_modules or ())
    keeps = list(ternary_cfg.keep_fp_modules or ())
    if not targets:
        preset = resolve_preset(model_type)
        targets = list(preset.target_modules)
        keeps.extend(preset.keep_fp_modules)
    keeps = _dedupe(keeps)

    target_res = [re.compile(pattern) for pattern in targets]
    keep_res = [re.compile(pattern) for pattern in keeps]
    protected_res = [re.compile(pattern) for pattern in ALWAYS_KEEP_FP]

    expert_entries = _convert_expert_stacks(model, ternary_cfg)

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
        recognized = aux_modules.explain(unmatched)
        if ternary_cfg.strict_enumeration:
            raise ValueError(
                f"{_summarize(unmatched)} match neither ternary.target_modules nor "
                "ternary.keep_fp_modules; list every Linear explicitly or set "
                "ternary.strict_enumeration: false to keep the rest full precision"
                + (f"\n\n{recognized}" if recognized else "")
            )
        LOG.warning(
            f"ternary: keeping {_summarize(unmatched)} full precision (unenumerated)"
            + (f"\n{recognized}" if recognized else "")
        )
    _warn_targeted_aux_modules([name for name, _ in to_swap])

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
                codebook=ternary_cfg.codebook,
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
                codebook=ternary_cfg.codebook,
            )
        )

    embedding_names, tied = _convert_embeddings(model, ternary_cfg, entries)
    entries.extend(expert_entries)

    manifest = SwapManifest(
        model_type=model_type,
        entries=entries,
        kept_fp=kept_fp,
        weight_scale=ternary_cfg.weight_scale,
        group_size=ternary_cfg.group_size,
        codebook=ternary_cfg.codebook,
        activation_bits=ternary_cfg.activation_bits,
        init=ternary_cfg.init,
        subln=ternary_cfg.subln,
        subln_modules=subln_modules,
        ternary_embeddings=embedding_names,
        tied_embeddings=tied,
        quantizer=_quantizer_identity(ternary_cfg),
    )
    setattr(model, MANIFEST_ATTR, manifest)
    # before the initializer: a reloaded master whose grid was persisted is already
    # baked, and a fit that cannot see that would refit the quantizer's own output
    _adopt_master_scales(model, manifest, cfg)
    initialize_model_latents(model, manifest, cfg)
    _apply_healing_strategy(model, ternary_cfg)
    LOG.info(
        f"ternary: swapped {len(entries)} Linear modules to ternary, "
        f"{len(kept_fp)} kept full precision"
    )

    output_dir = cfg.get("output_dir") if hasattr(cfg, "get") else None
    if output_dir and is_main_process():
        manifest.save(output_dir)

    return manifest


def _convert_embeddings(
    model: PreTrainedModel, ternary_cfg, entries: list[SwapEntry]
) -> tuple[list[str], bool]:
    """Swap the input embedding for a `TernaryEmbedding`, appending its manifest entry.

    The output head is only ever touched through a tie: when `tie_word_embeddings` is
    set it *is* this tensor, so it becomes ternary with it and the manifest says so.
    An untied `lm_head` stays full precision — it is a Linear, and `ALWAYS_KEEP_FP`
    covers it.

    Returns:
        The swapped module names and whether the head is tied to them.
    """
    if getattr(ternary_cfg, "embedding_dtype", "bf16") != "ternary":
        return [], False

    embedding = model.get_input_embeddings()
    if embedding is None:
        raise ValueError(
            "ternary.embedding_dtype: ternary, but the model exposes no input "
            "embedding via get_input_embeddings()"
        )
    if isinstance(embedding, TernaryEmbedding):
        raise ValueError("the input embedding is already a TernaryEmbedding")
    if not isinstance(embedding, nn.Embedding):
        raise ValueError(
            "ternary.embedding_dtype: ternary needs an nn.Embedding input embedding, "
            f"got {type(embedding).__name__}"
        )

    names = [
        name for name, module in model.named_modules() if module is embedding and name
    ]
    if not names:
        raise ValueError("the input embedding is not reachable by name from the model")

    ternary_embedding = TernaryEmbedding.from_embedding(
        embedding,
        scale_structure=ternary_cfg.embedding_scale,
        group_size=ternary_cfg.embedding_group_size,
        codebook=ternary_cfg.codebook,
    )
    for name in names:
        parent_name, _, attr = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr, ternary_embedding)
    model.set_input_embeddings(ternary_embedding)

    grid_scale, grid_size = _embedding_grid(ternary_cfg, embedding.embedding_dim)

    tied = bool(getattr(model.config, "tie_word_embeddings", False))
    if tied:
        _convert_tied_head(model, ternary_cfg, grid_scale, grid_size)
        LOG.warning(
            f"ternary: {names[0]} is ternary and tie_word_embeddings is set, so the "
            "output head is the same tensor. The head now quantizes it on the same "
            "grid, so training sees exactly the error the bake will introduce, and the "
            "shared latent is baked once. Untie the weights to keep a full-precision "
            "head"
        )

    for name in names:
        entries.append(
            SwapEntry(
                name=name,
                in_features=embedding.embedding_dim,
                out_features=embedding.num_embeddings,
                family="embedding",
                weight_scale=grid_scale,
                group_size=grid_size,
                codebook=ternary_cfg.codebook,
                kind="embedding",
            )
        )
    return names, tied


def _convert_tied_head(
    model: PreTrainedModel,
    ternary_cfg,
    grid_scale: WeightScaleMode,
    grid_size: int | None,
) -> None:
    """Put the tied output head on the embedding's grid, over the very same parameter.

    A tie means the head *is* the embedding tensor, so it bakes ternary whether or not
    anything is done here. Leaving the head as a plain Linear would mean it reads the
    latent all through training and only meets the quantization error at save time —
    the healing run would never train against the error it ships with. Quantizing it on
    the embedding's own grid closes that gap exactly: `per_row` is one group per row,
    which is what `group_size = embedding_dim` means to a Linear.

    Activations stay unquantized. The gap being closed is the shared *weight* grid;
    quantizing the head's input is a separate decision that a tie does not imply.
    """
    head = model.get_output_embeddings()
    if head is None or isinstance(head, TernaryLinear):
        return
    if not isinstance(head, nn.Linear) or head.bias is not None:
        LOG.warning(
            "ternary: tie_word_embeddings is set but the output head is a "
            f"{type(head).__name__}"
            f"{' with a bias' if getattr(head, 'bias', None) is not None else ''}, "
            "which the shared-grid head does not cover; it keeps reading the shared "
            "latent and will only meet the quantization error at bake time"
        )
        return

    ternary_head = TernaryLinear.from_linear(
        head,
        weight_scale=grid_scale,
        codebook=ternary_cfg.codebook,
        group_size=grid_size,
        activation_bits=None,
        fused=ternary_cfg.fused_fake_quant,
    )
    for name, module in list(model.named_modules()):
        if module is head and name:
            parent_name, _, attr = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            setattr(parent, attr, ternary_head)
    model.set_output_embeddings(ternary_head)
    # re-assert the tie against the replacements; both views must stay one parameter
    model.tie_weights()


def _embedding_grid(
    ternary_cfg, embedding_dim: int
) -> tuple[WeightScaleMode, int | None]:
    """Map an embedding scale structure onto the manifest's `(weight_scale, group_size)`.

    The bake and the packers read those two fields, and a per-row scale is exactly one
    group per row — so no packer needs to learn a new grid name.
    """
    structure = ternary_cfg.embedding_scale
    if structure == "per_tensor":
        return "absmean", None
    if structure == "per_row":
        return "group", embedding_dim
    return "group", ternary_cfg.embedding_group_size


def _apply_healing_strategy(model, ternary_cfg) -> None:
    """Freeze latents and/or attach low-rank deltas, after the init wrote them."""
    from .modules import TernaryEmbedding, TernaryLinear

    delta_cfg = ternary_cfg.low_rank_delta
    frozen_params = 0
    trainable_params = 0
    delta_modules = 0
    for module in model.modules():
        if isinstance(module, TernaryLinear):
            if delta_cfg is not None:
                module.enable_low_rank_delta(delta_cfg.r, delta_cfg.alpha)
                delta_modules += 1
            if ternary_cfg.heal_codes == "frozen":
                module.weight.requires_grad_(False)
                module.heal_frozen = True
                frozen_params += module.weight.numel()
        elif isinstance(module, TernaryEmbedding):
            if ternary_cfg.heal_codes == "frozen":
                module.weight.requires_grad_(False)
                module.heal_frozen = True
                frozen_params += module.weight.numel()
    if ternary_cfg.heal_codes == "frozen" or delta_cfg is not None:
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        LOG.info(
            f"ternary: healing strategy heal_codes={ternary_cfg.heal_codes}"
            + (
                f" low_rank_delta(r={delta_cfg.r}, alpha={delta_cfg.alpha}) on "
                f"{delta_modules} modules"
                if delta_cfg is not None
                else ""
            )
            + f"; {frozen_params:,} latent params frozen, "
            f"{trainable_params:,} params trainable"
        )


def _adopt_master_scales(
    model: PreTrainedModel, manifest: SwapManifest, cfg: DictDefault
) -> None:
    """Adopt the grid persisted beside `base_model`, when it was a ternary master."""
    base_model = cfg.get("base_model") if hasattr(cfg, "get") else None
    if not base_model or not Path(base_model).is_dir():
        return
    try:
        saved = SwapManifest.load(base_model)
    except (FileNotFoundError, TypeError, ValueError):
        return
    manifest.scales = saved.scales
    adopt_persisted_scales(model, manifest)


def adopt_persisted_scales(model: PreTrainedModel, manifest: SwapManifest) -> int:
    """Give every swapped module the grid its master was baked on, where one is stored.

    `dual` and the single-plane grids can be read back out of the values, but a free
    sum cannot: a row that uses only combination states never stores `s1` itself. A
    master that persisted its scales hands them back here, so the reloaded module
    computes the function it was saved with instead of an inferred one.

    Args:
        model: The freshly converted model.
        manifest: The manifest loaded beside the master.

    Returns:
        The number of modules that adopted a persisted grid.
    """
    if not manifest.scales:
        return 0
    adopted = 0
    for entry in manifest.entries:
        scales = manifest.scales_for(entry.name)
        module = model.get_submodule(entry.name)
        if scales is None or not isinstance(module, TernaryLinear):
            continue
        if module.adopt_scales(*scales):
            adopted += 1
    if adopted:
        LOG.info(f"ternary: adopted persisted scales for {adopted} modules")
    return adopted


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
    if saved.codebook != ternary_cfg.codebook:
        raise ValueError(
            f"{source / MANIFEST_FILENAME} says this master was fitted on the "
            f"{saved.codebook} codebook, but this run configures "
            f"{ternary_cfg.codebook}. The two grids share the {{-s, +s}} values, so "
            "the baked probe still lights and nothing fails loudly — the run simply "
            "continues on a different code set than the one that was trained. Set "
            f"ternary.codebook: {saved.codebook}, or start from the original "
            "full-precision checkpoint"
        )
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


def _dedupe(patterns: list[str]) -> list[str]:
    """Drop repeated regexes, keeping the first occurrence's position."""
    return list(dict.fromkeys(patterns))


def _warn_targeted_aux_modules(names: list[str]) -> None:
    """Warn about enumerated targets the registry knows should stay full precision."""
    for key, matched in aux_modules.group_by_family(names).items():
        family = aux_modules.AUX_MODULE_FAMILIES[key]
        LOG.warning(
            f"ternary: ternary.target_modules ternarizes {_summarize(matched)}, which "
            f"the {family.label} family keeps full precision by default: "
            f"{family.rationale}"
        )


def _grid(weight_scale: str, group_size: int | None) -> str:
    return weight_scale if group_size is None else f"{weight_scale}/{group_size}"


def _convert_expert_stacks(model, ternary_cfg) -> list[SwapEntry]:
    """Parametrize every fused 3-D expert stack with a TernaryExperts quantizer.

    The stack stays where the architecture put it; every read goes through the
    quantizer, and the manifest names the RAW parameter (kind="experts",
    `parameter_key()` returns it verbatim) so bake and the packers can find the
    tensor a fused stack IS rather than hangs a `.weight` off.
    """
    from torch.nn.utils import parametrize

    from .modules import TernaryExperts

    if ternary_cfg.codebook != "ternary":
        for name in _fused_expert_stacks(model):
            raise NotImplementedError(
                f"fused expert stacks support codebook: ternary only ({name})"
            )
    entries: list[SwapEntry] = []
    for name, module in list(model.named_modules()):
        if "expert" not in f"{name}.{type(module).__name__}".lower():
            continue
        for attr, param in list(module.named_parameters(recurse=False)):
            if param.ndim != 3:
                continue
            quantizer = TernaryExperts(param)
            parametrize.register_parametrization(module, attr, quantizer)
            entries.append(
                SwapEntry(
                    name=f"{name}.{attr}",
                    in_features=int(param.shape[-1]),
                    out_features=int(param.shape[-2]),
                    family="experts",
                    weight_scale="learnable",
                    codebook=ternary_cfg.codebook,
                    kind="experts",
                )
            )
            LOG.info(
                f"ternary: parametrized expert stack {name}.{attr} "
                f"{tuple(param.shape)} with per-expert scales"
            )
    return entries


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
    binary = ternary_cfg.codebook == "binary"
    return {
        "scheme": "binary" if binary else "dual_ternary" if dual else "ternary",
        "codes": [-1, 1] if binary else [-2, -1, 0, 1, 2] if dual else [-1, 0, 1],
        "codebook": ternary_cfg.codebook,
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
    "SCALES_FILENAME",
    "SwapEntry",
    "SwapManifest",
    "assert_scale_mode_reloadable",
    "convert_model",
    "get_manifest",
    "module_family",
    "resolve_preset",
]
