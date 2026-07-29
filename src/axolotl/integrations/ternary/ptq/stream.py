"""Shard-at-a-time PTQ fitting — the engine behind `axolotl ternary fit-stream`.

`initialize_model_latents` fits the latents of a model that is already built, which
caps the fittable model at what one host can hold. This pass never builds a model: it
walks the checkpoint's shards, fits each weight tensor as it is read, and writes the
reconstruction straight back out. Peak memory is one shard, so the ceiling moves from
"fits in RAM" to "one shard fits in RAM".

Three things follow from having no model, and they are what this module is:

- **Classification is by tensor name.** There are no `nn.Linear` instances to
  enumerate, so `ternary.target_modules` / `keep_fp_modules` are matched against the
  state-dict key with its `.weight` suffix stripped, and `aux_modules` supplies the
  same auxiliary-module verdicts the swap uses. A name that matches nothing is copied
  through, never fitted — the strict enumeration the swap can afford needs a module
  tree to enumerate.
- **Fused expert stacks are fittable here, on a per-tensor grid only.** `convert_model`
  refuses a 3-D expert parameter because it cannot swap one, but a fit only needs a
  matrix, so each `[i]` slice is fitted on its own. Which of the slice's two axes is
  the output one is not knowable from a state-dict key — transformers stores GptOss and
  Llama4 experts as `(experts, in, out)` and Qwen3-MoE as `(experts, out, in)` — so a
  grid that ties a scale to an axis is refused rather than guessed. A fused stack also
  has no manifest entry (the manifest addresses a module through its `.weight`) and so
  no packed export format: a streamed MoE master is a `master_bf16` artifact.
- **Resumption is per shard.** A fit over a 100-shard checkpoint is long enough that
  it will be interrupted. Every finished shard appends a completion record naming both
  the input it was fitted from and the output bytes it produced, and each output shard
  is written to a temporary name, fsynced and renamed only once it is complete, so an
  interrupted run leaves either a finished shard or no shard — never a half-written one
  that a resumed pass would trust.

The numerics are the ones `ptq/ternary_fit.py` pins; this module only decides which
tensors reach the solver and which codebook it solves for.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import json
import os
import re
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

import torch
from safetensors import SafetensorError, safe_open

from axolotl.utils.dict import DictDefault
from axolotl.utils.logging import get_logger

from .. import aux_modules, quant, swap
from ..args import (
    NON_PER_TENSOR_SCALE_MODES,
    UNIMPLEMENTED_CODEBOOKS,
    TernaryConfig,
    assert_stream_supported,
    resolve_ternary_config,
)
from ..export import bake, bitplanes
from ..swap import ALWAYS_KEEP_FP, SwapEntry, SwapManifest
from .ternary_fit import dequantize_fit, fit_ternary

LOG = get_logger(__name__)

RECORD_FILENAME: str = "ternary_stream_fit.json"

# suffix an output shard carries until it is complete
PARTIAL_SUFFIX: str = ".partial"

# what a streamed pass does with one tensor
TensorRole = Literal["fit", "keep_fp", "copy"]

# init modes with a streamed solver; the rest are refused by `assert_stream_supported`
# or land here as a NotImplementedError, never as a silent copy
_STREAMED_INITS: frozenset[str] = frozenset({"ternary_fit"})

_WEIGHT_SUFFIX: str = ".weight"

_HASH_CHUNK: int = 1 << 20

# what this pass writes into the output itself, and must not copy over from the input
_OWN_FILES: frozenset[str] = frozenset(
    {RECORD_FILENAME, swap.MANIFEST_FILENAME, swap.SCALES_FILENAME}
)

# families whose counts are worth spelling out in the summary before it truncates
_MAX_REPORTED_FAMILIES: int = 12

# tensor names a warning spells out before it truncates
_MAX_REPORTED_TENSORS: int = 5

# packed container each two-plane grid ships in, inverted so the cost this module
# reports and the cost the packer measures cannot drift apart
_TWO_PLANE_FORMATS: dict[str, str] = {
    mode: fmt for fmt, mode in bitplanes.FORMAT_SCALE_MODES.items()
}

# modules a packer must be told to leave alone even when no shard holds a tensor for
# them: a tied head shares the embedding's weight and has none of its own
_IMPLIED_KEEP_FP: tuple[str, ...] = ("lm_head",)


@dataclass(frozen=True)
class ShardRecord:
    """One output shard that finished, and the digest proving it is the finished one."""

    shard: str
    tensors: int
    fitted: int
    copied: int
    sha256: str
    # grid this shard was fitted on; a resume that configures another one refits it
    fingerprint: str = ""
    # input shard these bytes were fitted from; a resume against another checkpoint,
    # or against a re-downloaded one, refits rather than keeping the other model
    input_sha256: str = ""
    weights: int = 0
    zeros: int = 0
    bits: int = 0
    families: dict[str, int] = field(default_factory=dict)
    # tensors fitted with no manifest entry to address them by (fused expert stacks)
    unmanifested: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Return a JSON-serializable view of the record."""
        return asdict(self)


@dataclass
class StreamFitReport:
    """What one `stream_fit` pass did, including the work a resume skipped."""

    input_dir: str
    output_dir: str
    codebook: str
    init: str
    weight_scale: str
    shards: int = 0
    shards_fitted: int = 0
    shards_skipped: int = 0
    tensors_fitted: int = 0
    tensors_copied: int = 0
    weights_fitted: int = 0
    zeros_fitted: int = 0
    bits_fitted: int = 0
    seconds: float = 0.0
    families: dict[str, int] = field(default_factory=dict)
    unmanifested: list[str] = field(default_factory=list)
    records: dict[str, ShardRecord] = field(default_factory=dict)

    @property
    def zero_fraction(self) -> float:
        """Fraction of the fitted weights the solver put on the zero state."""
        if not self.weights_fitted:
            return 0.0
        return self.zeros_fitted / self.weights_fitted

    @property
    def bits_per_weight(self) -> float:
        """Cost of the fitted weights in the container each one's grid packs into.

        Not a function of the codebook alone: single-plane ternary goes to mask+sign,
        whose `2 - zero_fraction` is a measurement of the sparsity the solver found,
        but `dual` and `trit_planes` have flat-cost containers (`fv5`, `tp9`) whose
        per-row scale pair a narrow tensor pays real bits for, and a fused expert
        stack has no container at all and is counted at the master's own width. The
        per-tensor scale of a single-plane container is one float and is ignored.
        """
        if not self.weights_fitted:
            return 0.0
        return self.bits_fitted / self.weights_fitted

    @property
    def weights_per_second(self) -> float:
        """Fitted weights solved per wall-clock second of this pass."""
        if self.seconds <= 0.0:
            return 0.0
        return self.weights_fitted / self.seconds

    def to_dict(self) -> dict:
        """Return a JSON-serializable view of the report."""
        data = asdict(self)
        data["records"] = {
            name: record.to_dict() for name, record in sorted(self.records.items())
        }
        data["zero_fraction"] = self.zero_fraction
        data["bits_per_weight"] = self.bits_per_weight
        data["weights_per_second"] = self.weights_per_second
        return data


@dataclass
class _ShardCounts:
    """Per-shard tallies, merged into both the record and the report."""

    tensors: int = 0
    fitted: int = 0
    copied: int = 0
    weights: int = 0
    zeros: int = 0
    bits: int = 0
    families: dict[str, int] = field(default_factory=dict)
    unmanifested: list[str] = field(default_factory=list)


@dataclass
class _Pass:
    """The mutable state one streamed pass carries from shard to shard."""

    ternary_cfg: TernaryConfig
    device: str | torch.device
    manifest: SwapManifest
    entries: dict[str, SwapEntry]
    kept_fp: dict[str, None]


def stream_fit(
    input_dir: str | Path,
    output_dir: str | Path,
    cfg: DictDefault,
    device: str | torch.device = "cpu",
    resume: bool = True,
) -> StreamFitReport:
    """Fit every targeted tensor of a checkpoint shard by shard, without building a model.

    Args:
        input_dir: A full-precision checkpoint directory (safetensors shards plus the
            usual config/tokenizer files).
        output_dir: Where the fitted checkpoint is written; must not be `input_dir`.
        cfg: The axolotl config; `ternary.*` is read through `resolve_ternary_config`
            and supplies the target/keep lists, the codebook, the scale mode and the
            init mode being applied.
        device: Where each shard's tensors are fitted. The solver is the bottleneck,
            so a GPU is worth the transfer even though the shards are read and written
            on the host.
        resume: Skip shards whose completion record is already in `output_dir` and
            whose input and output bytes both still match its digests. `False` refits
            everything.

    Returns:
        A `StreamFitReport` covering this pass, including the shards a resume skipped.

    Raises:
        FileNotFoundError: If `input_dir` holds no shards.
        ValueError: If `output_dir` is `input_dir`, or `ternary.init` cannot run
            streamed (see `args.assert_stream_supported`).
        NotImplementedError: For a codebook or scale mode whose streamed solver has
            not landed.
    """
    ternary_cfg = resolve_ternary_config(cfg)
    assert_stream_supported(ternary_cfg)
    _assert_knobs_supported(ternary_cfg)
    if ternary_cfg.codebook in UNIMPLEMENTED_CODEBOOKS:
        raise NotImplementedError(
            f"ternary.codebook: {ternary_cfg.codebook} is accepted by the schema but "
            "its quantizer has not landed yet; use `codebook: ternary`"
        )
    if ternary_cfg.init not in _STREAMED_INITS:
        raise NotImplementedError(
            f"ternary.init: {ternary_cfg.init} has no streamed solver yet; use "
            f"`init: {sorted(_STREAMED_INITS)[0]}`"
        )

    source, output = Path(input_dir), Path(output_dir)
    shards = bake.shard_paths(source)
    if output.resolve() == source.resolve():
        raise ValueError(
            f"fit-stream writes a new checkpoint and cannot overwrite its input "
            f"({source}); an interrupted in-place pass would leave a directory that is "
            "part fitted and part original, with nothing to tell the halves apart"
        )
    ternary_cfg = _with_arch_preset(ternary_cfg, source)
    output.mkdir(parents=True, exist_ok=True)

    fingerprint = _fingerprint(ternary_cfg)
    records = _resumable_records(output, fingerprint) if resume else {}
    previous = _load_manifest(output) if records else None
    if records and previous is None:
        records = {}

    work = _Pass(
        ternary_cfg=ternary_cfg,
        device=device,
        manifest=_new_manifest(ternary_cfg, source, previous),
        entries=({entry.name: entry for entry in previous.entries} if previous else {}),
        kept_fp=_initial_kept_fp(previous),
    )
    report = StreamFitReport(
        input_dir=str(source),
        output_dir=str(output),
        codebook=ternary_cfg.codebook,
        init=ternary_cfg.init,
        weight_scale=ternary_cfg.weight_scale,
        shards=len(shards),
        records=dict(records),
    )

    bake.copy_aux_files(
        source, output, skip={path.name for path in shards} | _OWN_FILES
    )
    started = time.monotonic()
    for path in shards:
        destination = output / path.name
        source_digest = _file_sha256(path)
        record = records.get(path.name)
        if record is not None and _is_finished(record, destination, source_digest):
            report.shards_skipped += 1
            _merge(report, record)
            continue
        tensors, metadata, counts = _fit_shard(path, work)
        write_shard_atomic(tensors, destination, metadata)
        record = ShardRecord(
            shard=path.name,
            tensors=counts.tensors,
            fitted=counts.fitted,
            copied=counts.copied,
            sha256=_file_sha256(destination),
            fingerprint=fingerprint,
            input_sha256=source_digest,
            weights=counts.weights,
            zeros=counts.zeros,
            bits=counts.bits,
            families=dict(counts.families),
            unmanifested=list(counts.unmanifested),
        )
        _persist(output, work, records, record)
        report.shards_fitted += 1
        report.records[record.shard] = record
        _merge(report, record)
    report.seconds = time.monotonic() - started

    bake.write_quantizer_metadata(output, work.manifest)
    # pylint: disable=protected-access
    swap._warn_targeted_aux_modules(list(work.entries))
    _warn_unmanifested(report.unmanifested)
    LOG.info(
        f"ternary: fit-stream wrote {report.tensors_fitted} fitted tensors into "
        f"{output} over {report.shards_fitted}/{report.shards} shards "
        f"({report.shards_skipped} skipped), {report.bits_per_weight:.3f} bits per "
        f"weight at {report.zero_fraction:.1%} zero"
    )
    return report


def classify_tensor(name: str, ternary_cfg: TernaryConfig) -> TensorRole:
    """Return what a streamed pass does with the state-dict entry `name`.

    Matching is on the module name — `name` with its trailing `.weight` removed — so
    the same regexes that drive `convert_model` drive this. `keep_fp` and `copy` both
    write the tensor through untouched and are distinguished only for the report: a
    `keep_fp` tensor was named by the config or the auxiliary-module registry, a
    `copy` tensor matched nothing (norms, embeddings, biases, buffers).

    The precedence is the swap's own, minus the errors it can only raise with a module
    tree in front of it: `ALWAYS_KEEP_FP` first, then the keep list, then the target
    list, then the registry's default-keep verdict.

    Args:
        name: A state-dict key.
        ternary_cfg: The resolved ternary config supplying the target/keep lists.

    Returns:
        `"fit"`, `"keep_fp"` or `"copy"`.
    """
    module = _module_name(name)
    if _matches(module, ALWAYS_KEEP_FP):
        return "keep_fp"
    if _matches(module, tuple(ternary_cfg.keep_fp_modules or ())):
        return "keep_fp"
    if _matches(module, tuple(ternary_cfg.target_modules or ())):
        return "fit"
    family = aux_modules.classify(module)
    return "keep_fp" if family is not None and family.keep_fp else "copy"


def fit_tensor(
    tensor: torch.Tensor,
    ternary_cfg: TernaryConfig,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Fit one weight tensor and return its reconstruction in the input dtype.

    A 2-D tensor goes straight to the solver. A 3-D fused expert stack is fitted one
    `[i]` slice at a time and restacked, so each expert carries its own scale, and only
    under a per-tensor grid: a state-dict key does not say which of the slice's two
    axes is the output one, and a scale tied to the wrong axis varies along the
    contraction axis, which no dequantization kernel can express. The solver is chosen
    by `ternary_cfg.codebook`: the alternating fit for `ternary`, the closed-form
    `mean(|W|)` scale plus `sign(W)` for `binary`.

    What is returned is the reconstruction `codes · f16(scale)`, not the codes: a
    streamed artifact is an ordinary checkpoint whose values happen to sit on the
    grid, so it reloads through `convert_model` and is detected as baked.

    Args:
        tensor: A 2-D weight or a 3-D fused expert stack.
        ternary_cfg: The resolved ternary config supplying the codebook, scale mode
            and group size.
        device: Where the solve runs; the result comes back on `tensor.device`.

    Returns:
        The reconstruction, same shape and dtype as `tensor`.

    Raises:
        ValueError: If `tensor` is neither 2-D nor 3-D, or a 3-D stack is configured
            on a grid that ties its scale to an axis.
        NotImplementedError: For a codebook whose streamed solver has not landed.
    """
    if tensor.ndim == 3:
        _reject_unfittable_stack(ternary_cfg)
        return torch.stack(
            [_fit_matrix(part, ternary_cfg, device)[0] for part in tensor]
        )
    return _fit_matrix(tensor, ternary_cfg, device)[0]


def write_shard_atomic(
    tensors: dict[str, torch.Tensor],
    path: str | Path,
    metadata: dict[str, str] | None = None,
) -> Path:
    """Write a shard to `path` through a `PARTIAL_SUFFIX` temporary, then rename it.

    The rename is what makes a completion record trustworthy: a shard either does not
    exist or is whole, so a resumed pass never reads a torn file the record claims is
    finished. The bytes are fsynced before the rename and the directory after it, so
    the guarantee survives host and power loss rather than only a killed process.

    Args:
        tensors: The shard's contents.
        path: Final path of the shard.
        metadata: safetensors metadata to preserve from the source shard.

    Returns:
        `path`.
    """
    path = Path(path)
    # the marker goes before the suffix: the writer picks its format from the suffix,
    # and a `.partial` extension would silently write a torch pickle
    partial = path.with_name(f"{path.stem}{PARTIAL_SUFFIX}{path.suffix}")
    bake.save_shard(tensors, partial, metadata)
    _fsync(partial)
    partial.replace(path)
    _fsync(path.parent)
    return path


def load_records(output_dir: str | Path) -> dict[str, ShardRecord]:
    """Return the completion records in `output_dir`, keyed by shard name.

    A missing or unreadable record file is not an error — it means nothing finished
    and the pass starts over.
    """
    path = Path(output_dir) / RECORD_FILENAME
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    if not isinstance(data, dict):
        return {}
    known = {entry.name for entry in fields(ShardRecord)}
    records: dict[str, ShardRecord] = {}
    for name, raw in data.items():
        if not isinstance(raw, dict):
            continue
        try:
            records[name] = ShardRecord(
                **{key: value for key, value in raw.items() if key in known}
            )
        except TypeError:
            continue
    return records


def write_records(output_dir: str | Path, records: dict[str, ShardRecord]) -> Path:
    """Write the completion records into `output_dir` and return the record path.

    Called after every finished shard, not once at the end: the record is the resume
    state, so it is only useful if it survives the interruption it exists for.
    """
    path = Path(output_dir) / RECORD_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {name: record.to_dict() for name, record in sorted(records.items())}
    partial = path.with_name(f"{path.stem}{PARTIAL_SUFFIX}{path.suffix}")
    partial.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _fsync(partial)
    partial.replace(path)
    _fsync(path.parent)
    return path


def _fit_shard(
    path: Path, work: _Pass
) -> tuple[dict[str, torch.Tensor], dict[str, str], _ShardCounts]:
    """Fit one input shard, one resident tensor at a time, into its output contents."""
    counts = _ShardCounts()
    tensors: dict[str, torch.Tensor] = {}
    for key, tensor in _iter_tensors(path):
        counts.tensors += 1
        role = classify_tensor(key, work.ternary_cfg)
        if role != "fit":
            tensors[key] = tensor
            counts.copied += 1
            _record_kept(work, key, role)
            continue
        if not tensor.is_floating_point():
            raise ValueError(
                f"{key} matches ternary.target_modules but holds {tensor.dtype}; "
                "fit-stream solves float weights and would read a quantized payload "
                "as one. Narrow the target list or start from a float checkpoint"
            )
        fitted, manifested = _fit_entry(key, tensor, work)
        tensors[key] = fitted
        counts.fitted += 1
        counts.weights += tensor.numel()
        zeros = int((fitted == 0).sum())
        counts.zeros += zeros
        counts.bits += _container_bits(fitted, zeros, work.ternary_cfg)
        if not manifested:
            counts.unmanifested.append(key)
        family = swap.module_family(_module_name(key))
        counts.families[family] = counts.families.get(family, 0) + 1
    return tensors, _shard_metadata(path), counts


def _fit_entry(
    key: str, tensor: torch.Tensor, work: _Pass
) -> tuple[torch.Tensor, bool]:
    """Fit one targeted tensor, and register what the manifest has to say about it.

    Returns the reconstruction and whether a manifest entry now addresses it: a fused
    expert stack is one parameter holding many modules, so there is no `<name>.weight`
    for an entry to name and no packed format that can reach it.
    """
    if tensor.ndim == 3:
        fitted = fit_tensor(tensor, work.ternary_cfg, work.device)
        for index, part in enumerate(fitted):
            _assert_requantizes(f"{key}[{index}]", part, work.ternary_cfg)
        return fitted, False

    fitted, scales = _fit_matrix(tensor, work.ternary_cfg, work.device)
    _assert_requantizes(key, fitted, work.ternary_cfg, scales)
    name = _module_name(key)
    manifested = key.endswith(_WEIGHT_SUFFIX)
    if manifested:
        work.entries[name] = SwapEntry(
            name=name,
            in_features=int(tensor.shape[1]),
            out_features=int(tensor.shape[0]),
            family=swap.module_family(name),
            weight_scale=work.ternary_cfg.weight_scale,
            group_size=work.ternary_cfg.group_size,
            codebook=work.ternary_cfg.codebook,
        )
    if scales is not None:
        work.manifest.record_scales(name, *scales)
    return fitted, manifested


def _fit_matrix(
    tensor: torch.Tensor, ternary_cfg: TernaryConfig, device: str | torch.device
) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
    """Return one matrix's reconstruction and the scale pair a two-plane grid persists."""
    if tensor.ndim != 2:
        raise ValueError(
            f"fit_tensor expects a 2-D weight or a 3-D fused expert stack, got shape "
            f"{tuple(tensor.shape)}"
        )
    dense = tensor.detach().to(device)
    if ternary_cfg.codebook == "binary":
        return _fit_binary(tensor, dense, ternary_cfg), None

    codes, scale = fit_ternary(
        dense,
        scale_mode=ternary_cfg.weight_scale,
        group_size=ternary_cfg.group_size,
    )
    fitted = dequantize_fit(
        codes, scale, tensor.dtype, scale_mode=ternary_cfg.weight_scale
    ).to(tensor.device)
    return fitted, _persisted_scales(scale, ternary_cfg)


def _fit_binary(
    tensor: torch.Tensor, dense: torch.Tensor, ternary_cfg: TernaryConfig
) -> torch.Tensor:
    """Return the closed-form sign-codebook reconstruction of one matrix."""
    group_size = _scale_group_size(dense, ternary_cfg)
    scale = quant.binary_scale(dense, group_size)
    codes = quant.binary_codes(dense, scale)
    return quant.dequantize_codes(codes, quant.f16_round_scale(scale), tensor.dtype).to(
        tensor.device
    )


def _persisted_scales(
    scale: torch.Tensor, ternary_cfg: TernaryConfig
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return the `(first, second)` pair a two-plane master ships beside its values.

    The ordering is the one the modules record: `(s_lo, s_hi)` for `dual`, `(s1, s2)`
    for the free sum, whose larger scale is the fit's second column. Both columns are
    cloned — two views of one `(rows, 2)` tensor share storage, which the sidecar
    refuses to write.
    """
    if ternary_cfg.weight_scale == "dual":
        return scale[:, :1].clone(), scale[:, 1:].clone()
    if ternary_cfg.weight_scale == "trit_planes":
        return scale[:, 1:].clone(), scale[:, :1].clone()
    return None


def _assert_requantizes(
    key: str,
    fitted: torch.Tensor,
    ternary_cfg: TernaryConfig,
    scales: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> None:
    """Refuse a reconstruction the quantizer would not read back as its own output.

    Every fitted tensor is a fixed point by construction — the solver assigns through
    the same `f16`-rounded scale the packers derive — so a tensor that fails here is a
    real disagreement between the fit and the export path, on the very first shard
    rather than after the whole checkpoint has been rewritten.
    """
    if ternary_cfg.codebook == "binary":
        recovered = quant.baked_binary_codes_and_scale(
            fitted, _scale_group_size(fitted, ternary_cfg)
        )
        if recovered is None:
            raise RuntimeError(
                f"{key}: the binary fit does not re-quantize to itself; its values are "
                "not exactly {-s, +s}"
            )
        return
    try:
        bake.derive_codes_and_scale(
            fitted, ternary_cfg.group_size, ternary_cfg.weight_scale, scales
        )
    except ValueError as error:
        raise RuntimeError(
            f"{key}: the fit does not re-quantize to itself on the "
            f"{ternary_cfg.weight_scale} grid ({error})"
        ) from error


def _reject_unfittable_stack(ternary_cfg: TernaryConfig) -> None:
    """Refuse a fused stack on a grid that ties its scale to an axis of the slice.

    The solver reduces over the last axis, which is the input axis of a `(out, in)`
    Linear weight — but a fused expert stack does not say which axis is which, and
    transformers ships both conventions: GptOss and Llama4 store
    `(experts, in, out)`, Qwen3-MoE and Qwen3-Next `(experts, out, in)`. Guessing is
    worse than refusing, because the wrong guess is silent: it produces a scale
    varying along the contraction axis, which no dequantization kernel expresses and
    the manifest sidecar — keyed by module, one grid per module — cannot carry either.
    """
    if ternary_cfg.weight_scale not in NON_PER_TENSOR_SCALE_MODES:
        return
    raise ValueError(
        f"weight_scale: {ternary_cfg.weight_scale} ties each scale to one axis of the "
        "matrix, and a fused expert stack does not say which of its two trailing axes "
        "is the output one: transformers stores GptOss and Llama4 experts as "
        "(experts, in, out) and Qwen3-MoE as (experts, out, in), so the same config "
        "would fit a per-output-row grid on one architecture and a per-input-feature "
        "grid — which nothing can dequantize — on the other. Fit fused experts under "
        "a per-tensor grid (`weight_scale: learnable`)"
    )


def _scale_group_size(tensor: torch.Tensor, ternary_cfg: TernaryConfig) -> int | None:
    """Group size the quantizer sees: a per-row scale is one group of `in_features`."""
    if ternary_cfg.weight_scale == "learnable_row":
        return int(tensor.shape[-1])
    return ternary_cfg.group_size


def _container_bits(
    fitted: torch.Tensor, zeros: int, ternary_cfg: TernaryConfig
) -> int:
    """Return what one fitted tensor costs in the container its grid packs into.

    The two-plane grids are measured through `bitplanes` itself, per-row scales
    included, so this accounting and the packer's cannot disagree. A single-plane
    container's one scale per tensor is left out — 32 bits against a whole tensor.
    """
    numel = fitted.numel()
    if fitted.ndim != 2:
        # no packed format reaches a fused stack; it ships at the master's own width
        return numel * fitted.element_size() * 8
    fmt = _TWO_PLANE_FORMATS.get(ternary_cfg.weight_scale)
    if fmt is not None:
        shape = (int(fitted.shape[0]), int(fitted.shape[1]))
        return round(bitplanes.bits_per_weight(fmt, shape) * numel)
    if ternary_cfg.codebook == "binary":
        return numel
    return 2 * numel - zeros


def _record_kept(work: _Pass, key: str, role: TensorRole) -> None:
    """Remember a module the config or the registry kept full precision.

    Only a `keep_fp` verdict is recorded: `kept_fp` is read downstream as the modules
    a packer must be told to leave alone, and a `copy` tensor is a norm, a bias or a
    buffer that no packer would have touched.
    """
    if role == "keep_fp" and key.endswith(_WEIGHT_SUFFIX):
        work.kept_fp[_module_name(key)] = None


def _initial_kept_fp(previous: SwapManifest | None) -> dict[str, None]:
    """Seed `kept_fp` with the modules no shard can record, then a resume's own.

    A tied head has no tensor of its own, so nothing the pass reads names it — and a
    packer that is not told about it wraps it in a quantized Linear whose codes the
    artifact does not contain, which fails at load rather than at export.
    """
    return dict.fromkeys([*_IMPLIED_KEEP_FP, *(previous.kept_fp if previous else ())])


def _persist(
    output: Path,
    work: _Pass,
    records: dict[str, ShardRecord],
    record: ShardRecord,
) -> None:
    """Commit one finished shard: manifest first, then the record that trusts it.

    A resumed pass reads its skipped shards' entries out of the manifest rather than
    out of the shard, so the manifest has to be durable before the record that lets
    the shard be skipped.
    """
    work.manifest.entries = list(work.entries.values())
    work.manifest.kept_fp = list(work.kept_fp)
    work.manifest.save(output)
    records[record.shard] = record
    write_records(output, records)


def _merge(report: StreamFitReport, record: ShardRecord) -> None:
    """Fold one shard's record into the totals, whether it was fitted or skipped."""
    report.tensors_fitted += record.fitted
    report.tensors_copied += record.copied
    report.weights_fitted += record.weights
    report.zeros_fitted += record.zeros
    report.bits_fitted += record.bits
    report.unmanifested.extend(record.unmanifested)
    for family, count in record.families.items():
        report.families[family] = report.families.get(family, 0) + count


def _is_finished(record: ShardRecord, destination: Path, source_digest: str) -> bool:
    """Whether `record` describes this exact input fitted into this exact output.

    Both halves are load-bearing. The output digest proves the write finished; the
    input digest proves it was *this* checkpoint's shard that was fitted, so a resume
    into a directory another checkpoint was fitted into — or one whose source has been
    re-downloaded since — refits rather than shipping the other model's weights under
    this run's report. Every shard is hashed on the way past for that, which is one
    extra sequential read against a solve that is the bottleneck.
    """
    if not _matches_digest(destination, record.sha256):
        return False
    if record.input_sha256 == source_digest:
        return True
    LOG.warning(
        f"ternary: {record.shard} in the output was fitted from different input bytes "
        "than the ones in this pass's input directory; refitting it"
    )
    return False


def _assert_knobs_supported(ternary_cfg: TernaryConfig) -> None:
    """Refuse or flag the `ternary.*` knobs a streamed pass cannot honour.

    The same shape as `args.assert_stream_supported`: the constraint belongs to this
    entry point, not to the schema, because `axolotl train` builds a model and these
    knobs are about the module tree it builds.

    Raises:
        NotImplementedError: If `ternary.smoothing` is set, which `axolotl train`
            refuses too — a silent no-op here would make the two paths disagree on
            the same config.
    """
    if ternary_cfg.smoothing:
        raise NotImplementedError(
            "ternary.smoothing is accepted by the schema but not implemented yet; "
            "`axolotl train` refuses it, so a streamed fit that quietly skipped the "
            "fold would be the one entry point where the config means nothing"
        )
    if ternary_cfg.subln:
        LOG.warning(
            "ternary: subln inserts sub-norms into the module tree and fit-stream "
            "never builds one, so the streamed master carries none and its manifest "
            "records subln: false. The heal that reloads it inserts them at swap time"
        )


def _warn_unmanifested(names: list[str]) -> None:
    """Say out loud which fitted tensors no manifest entry — and so no packer — covers."""
    if not names:
        return
    LOG.warning(
        f"ternary: {_summarize(names)} were fitted but have no manifest entry: the "
        "manifest addresses a module through its `.weight`, and a fused expert stack "
        "is one parameter holding many modules. No packed export format reaches them "
        "— they ship at the master's own width, which the reported bits per weight "
        "counts — and `axolotl train` refuses to reload a fused stack, so a streamed "
        "MoE master is a master_bf16 artifact"
    )


def _resumable_records(output: Path, fingerprint: str) -> dict[str, ShardRecord]:
    """Return the records written by a pass configuring the same grid as this one."""
    records = load_records(output)
    kept = {
        name: record
        for name, record in records.items()
        if record.fingerprint == fingerprint
    }
    if records and not kept:
        LOG.warning(
            f"ternary: {output / RECORD_FILENAME} was written by a pass on a different "
            "grid; refitting every shard"
        )
    return kept


def _fingerprint(ternary_cfg: TernaryConfig) -> str:
    """Return the digest of everything that decides what a fitted tensor holds."""
    identity = json.dumps(
        {
            "codebook": ternary_cfg.codebook,
            "weight_scale": ternary_cfg.weight_scale,
            "group_size": ternary_cfg.group_size,
            "init": ternary_cfg.init,
            "target_modules": list(ternary_cfg.target_modules or ()),
            "keep_fp_modules": list(ternary_cfg.keep_fp_modules or ()),
        },
        sort_keys=True,
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _new_manifest(
    ternary_cfg: TernaryConfig, source: Path, previous: SwapManifest | None
) -> SwapManifest:
    """Return the manifest this pass writes, carrying a resumed pass's scales forward."""
    manifest = SwapManifest(
        model_type=_model_type(source),
        weight_scale=ternary_cfg.weight_scale,
        group_size=ternary_cfg.group_size,
        codebook=ternary_cfg.codebook,
        activation_bits=ternary_cfg.activation_bits,
        init=ternary_cfg.init,
        # pylint: disable=protected-access
        quantizer=swap._quantizer_identity(ternary_cfg),
    )
    if previous is not None:
        manifest.scales = previous.scales
    return manifest


def _load_manifest(output: Path) -> SwapManifest | None:
    """Return the manifest a previous pass left in `output`, or `None` if unusable.

    A torn scale sidecar counts as unusable rather than as a crash: `SwapManifest.save`
    writes it after the JSON and without a rename, so an interruption in that window is
    exactly the state a resume exists to recover from.
    """
    try:
        return SwapManifest.load(output)
    except (FileNotFoundError, SafetensorError, TypeError, ValueError):
        return None


def _with_arch_preset(ternary_cfg: TernaryConfig, source: Path) -> TernaryConfig:
    """Fill an empty target list from the checkpoint's own `model_type`, as the swap does."""
    if ternary_cfg.target_modules:
        return ternary_cfg
    preset = swap.resolve_preset(_model_type(source))
    keeps = list(ternary_cfg.keep_fp_modules or ()) + list(preset.keep_fp_modules)
    return ternary_cfg.model_copy(
        update={
            "target_modules": list(preset.target_modules),
            "keep_fp_modules": list(dict.fromkeys(keeps)),
        }
    )


def _model_type(source: Path) -> str:
    """Return the `model_type` in the checkpoint's config, or `'unknown'`."""
    path = source / bake.CONFIG_FILENAME
    if not path.is_file():
        return "unknown"
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "unknown"
    return str(config.get("model_type") or "unknown")


def _module_name(key: str) -> str:
    """Return the module a state-dict key belongs to, keeping fused parameters as they are."""
    if key.endswith(_WEIGHT_SUFFIX):
        return key[: -len(_WEIGHT_SUFFIX)]
    return key


@functools.lru_cache(maxsize=None)
def _compiled(patterns: tuple[str, ...]) -> tuple[re.Pattern, ...]:
    return tuple(re.compile(pattern) for pattern in patterns)


def _matches(name: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern.fullmatch(name) for pattern in _compiled(patterns))


def _iter_tensors(path: Path) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield one shard's tensors, materializing them one at a time where the format allows."""
    if path.suffix != ".safetensors":
        yield from bake.load_shard(path)[0].items()
        return
    with safe_open(path, framework="pt") as shard:
        for key in shard.keys():  # noqa: SIM118 — safetensors handle, not a mapping
            yield key, shard.get_tensor(key)


def _shard_metadata(path: Path) -> dict[str, str]:
    """Return the safetensors file-level metadata of `path`, if it has any."""
    if path.suffix != ".safetensors":
        return {}
    with safe_open(path, framework="pt") as shard:
        return shard.metadata() or {}


def _matches_digest(path: Path, sha256: str) -> bool:
    """Whether `path` exists and still hashes to what a record claims it does."""
    return path.is_file() and _file_sha256(path) == sha256


def _file_sha256(path: Path) -> str:
    """Return the sha256 of a file, read in chunks so a shard never lands in memory."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync(path: Path) -> None:
    """Flush a file's contents, or a directory's entries, out to the device."""
    with contextlib.suppress(OSError):
        handle = os.open(path, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def _summarize(names: list[str]) -> str:
    """Return a truncated, comma-separated view of a list of tensor names."""
    head = ", ".join(names[:_MAX_REPORTED_TENSORS])
    extra = len(names) - _MAX_REPORTED_TENSORS
    return f"{head} (+{extra} more)" if extra > 0 else head


def summarize_families(families: dict[str, int]) -> str:
    """Return the per-family fitted counts as one line, most-fitted first."""
    ordered = sorted(families.items(), key=lambda item: (-item[1], item[0]))
    head = ", ".join(
        f"{name} {count}" for name, count in ordered[:_MAX_REPORTED_FAMILIES]
    )
    extra = len(ordered) - _MAX_REPORTED_FAMILIES
    return f"{head} (+{extra} more)" if extra > 0 else head


__all__ = [
    "PARTIAL_SUFFIX",
    "RECORD_FILENAME",
    "ShardRecord",
    "StreamFitReport",
    "TensorRole",
    "classify_tensor",
    "fit_tensor",
    "load_records",
    "stream_fit",
    "summarize_families",
    "write_records",
    "write_shard_atomic",
]
