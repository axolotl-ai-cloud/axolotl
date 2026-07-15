"""Subspace-Aligned Rewiring: spectral projection math and streaming checkpoint driver."""

from __future__ import annotations

import fnmatch
import json
import math
import os
import shutil
from dataclasses import dataclass, field
from typing import Any

import torch
from huggingface_hub import save_torch_state_dict, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)

DEFAULT_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

PROJECTIONS = ("spectral", "none")
REWIRINGS = ("full", "diagonal", "off_diagonal")

SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"
SAFETENSORS_SINGLE_FILE = "model.safetensors"

SNAPSHOT_ALLOW_PATTERNS = [
    "*.safetensors",
    "*.safetensors.index.json",
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "vocab*",
    "merges.txt",
    "added_tokens.json",
    "chat_template*",
]

COPY_FILE_PATTERNS = [
    "config.json",
    "generation_config.json",
    "tokenizer*",
    "special_tokens_map.json",
    "vocab*",
    "merges.txt",
    "added_tokens.json",
    "chat_template*",
]

SAVE_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

FP16_MAX = 65504.0

MAX_SHARD_BYTES = 5 * 1024**3

REWIRING_FORMULA_VERSION = 1


@dataclass
class SARProjection:
    """Result of projecting a single weight matrix.

    Attributes:
        delta_star: fp32 ``[dout, din]`` reconstructed update to add to the target weight.
        m: fp32 ``[rank, rank]`` rewiring matrix, or None when ``projection="none"``.
        rank: base-subspace rank ``k`` used.
        delta_rank: delta truncation rank ``k'`` used.
    """

    delta_star: torch.Tensor
    m: torch.Tensor | None
    rank: int
    delta_rank: int


@dataclass
class SARResult:
    """Summary of a :func:`run_sar` invocation.

    Attributes:
        outputs: mapping of rank ratio to the directory holding that projected model.
        num_projected: number of weight tensors projected per output.
        m_params: total rewiring-matrix parameter count at the largest rank ratio.
        total_params: total parameter count of one output model.
    """

    outputs: dict[float, str] = field(default_factory=dict)
    num_projected: int = 0
    m_params: int = 0
    total_params: int = 0


def _validate_options(projection: str, rewiring: str) -> None:
    if projection not in PROJECTIONS:
        raise ValueError(f"projection must be one of {PROJECTIONS}, got {projection!r}")
    if rewiring not in REWIRINGS:
        raise ValueError(f"rewiring must be one of {REWIRINGS}, got {rewiring!r}")
    if projection == "none" and rewiring != "full":
        raise ValueError(
            "rewiring masks are undefined with projection='none'; use rewiring='full'"
        )


def _rank_for(ratio: float, min_dim: int) -> int:
    # the epsilon keeps binary-float noise (e.g. 0.07 * 300 = 21.000000000000004)
    # from bumping the ceil past the exact-arithmetic rank
    return min(min_dim, max(1, math.ceil(ratio * min_dim - 1e-9)))


def _truncated_delta(
    delta_u: torch.Tensor,
    delta_s: torch.Tensor,
    delta_vh: torch.Tensor,
    delta_rank: int,
) -> torch.Tensor:
    return (delta_u[:, :delta_rank] * delta_s[:delta_rank]) @ delta_vh[:delta_rank, :]


def _rewire(
    base_u: torch.Tensor,
    base_vh: torch.Tensor,
    delta_k: torch.Tensor,
    rank: int,
    rewiring: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    u_k = base_u[:, :rank]
    vh_k = base_vh[:rank, :]
    m = u_k.mT @ delta_k @ vh_k.mT
    if rewiring == "diagonal":
        m = torch.diag(torch.diagonal(m))
    elif rewiring == "off_diagonal":
        m = m.clone()
        m.fill_diagonal_(0.0)
    delta_star = u_k @ m @ vh_k
    return delta_star, m


def sar_project_matrix(
    w_base: torch.Tensor,
    w_trained: torch.Tensor,
    rank: int,
    *,
    delta_rank: int | None = None,
    projection: str = "spectral",
    rewiring: str = "full",
) -> SARProjection:
    """Project a trained weight's update onto the base weight's top-k spectral subspace.

    Args:
        w_base: base weight ``[dout, din]``, any float dtype; upcast to fp32 internally.
        w_trained: trained weight, same shape as ``w_base``.
        rank: subspace rank ``k``; ``1 <= rank <= min(dout, din)``.
        delta_rank: truncation rank ``k'`` for the update SVD; defaults to ``rank``.
        projection: ``"spectral"`` for the full projection, ``"none"`` to return the
            rank-``k'`` truncated update without projecting.
        rewiring: mask applied to the rewiring matrix — ``"full"``, ``"diagonal"``,
            or ``"off_diagonal"``.

    Returns:
        SARProjection with fp32 ``delta_star`` (and ``m`` when projecting).
    """
    _validate_options(projection, rewiring)
    if w_base.dim() != 2:
        raise ValueError(
            f"expected a 2D weight matrix, got shape {tuple(w_base.shape)}"
        )
    if w_base.shape != w_trained.shape:
        raise ValueError(
            f"shape mismatch: base {tuple(w_base.shape)} vs trained {tuple(w_trained.shape)}"
        )
    min_dim = min(w_base.shape)
    if not 1 <= rank <= min_dim:
        raise ValueError(f"rank must be in [1, {min_dim}], got {rank}")
    resolved_delta_rank = rank if delta_rank is None else delta_rank
    if not 1 <= resolved_delta_rank <= min_dim:
        raise ValueError(
            f"delta_rank must be in [1, {min_dim}], got {resolved_delta_rank}"
        )

    base32 = w_base.detach().to(torch.float32)
    delta = w_trained.detach().to(torch.float32) - base32

    delta_u, delta_s, delta_vh = torch.linalg.svd(delta, full_matrices=False)
    delta_k = _truncated_delta(delta_u, delta_s, delta_vh, resolved_delta_rank)
    if projection == "none":
        return SARProjection(
            delta_star=delta_k, m=None, rank=rank, delta_rank=resolved_delta_rank
        )

    base_u, _, base_vh = torch.linalg.svd(base32, full_matrices=False)
    delta_star, m = _rewire(base_u, base_vh, delta_k, rank, rewiring)
    return SARProjection(
        delta_star=delta_star, m=m, rank=rank, delta_rank=resolved_delta_rank
    )


@dataclass
class _TensorInfo:
    shard: str
    shape: tuple[int, ...]


@dataclass
class _ModelSource:
    ref: str
    directory: str
    revision: str | None
    shards: list[str]
    tensors: dict[str, _TensorInfo]


def _resolve_model_dir(model_ref: str, revision: str | None) -> tuple[str, str | None]:
    if os.path.isdir(model_ref):
        return model_ref, None
    directory = snapshot_download(
        model_ref, revision=revision, allow_patterns=SNAPSHOT_ALLOW_PATTERNS
    )
    # snapshot dirs are named by the resolved commit hash
    return directory, os.path.basename(directory)


def _load_source(model_ref: str, revision: str | None = None) -> _ModelSource:
    directory, resolved_revision = _resolve_model_dir(model_ref, revision)
    index_path = os.path.join(directory, SAFETENSORS_INDEX_FILE)
    single_path = os.path.join(directory, SAFETENSORS_SINGLE_FILE)
    if os.path.exists(index_path):
        with open(index_path, encoding="utf-8") as fp:
            weight_map: dict[str, str] = json.load(fp)["weight_map"]
        shards = [
            os.path.join(directory, shard) for shard in sorted(set(weight_map.values()))
        ]
        missing = [shard for shard in shards if not os.path.exists(shard)]
        if missing:
            raise FileNotFoundError(
                f"{model_ref}: shards listed in {SAFETENSORS_INDEX_FILE} are missing: "
                f"{[os.path.basename(shard) for shard in missing]}"
            )
    elif os.path.exists(single_path):
        shards = [single_path]
    else:
        hint = ""
        if any(
            os.path.exists(os.path.join(directory, name))
            for name in ("pytorch_model.bin", "pytorch_model.bin.index.json")
        ):
            hint = " Found a pytorch .bin checkpoint; convert it to safetensors first."
        raise FileNotFoundError(
            f"{model_ref}: no {SAFETENSORS_SINGLE_FILE} or {SAFETENSORS_INDEX_FILE} in "
            f"{directory}. SAR only supports safetensors checkpoints.{hint}"
        )

    tensors: dict[str, _TensorInfo] = {}
    for shard in shards:
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                tensors[name] = _TensorInfo(
                    shard=shard, shape=tuple(handle.get_slice(name).get_shape())
                )
    return _ModelSource(
        ref=model_ref,
        directory=directory,
        revision=resolved_revision,
        shards=shards,
        tensors=tensors,
    )


class _ShardCache:
    """Lazily opened, mmap-backed safetensors handles keyed by shard path."""

    def __init__(self) -> None:
        self._handles: dict[str, Any] = {}

    def read(self, source: _ModelSource, name: str) -> torch.Tensor:
        path = source.tensors[name].shard
        handle = self._handles.get(path)
        if handle is None:
            handle = safe_open(path, framework="pt", device="cpu")
            self._handles[path] = handle
        return handle.get_tensor(name)

    def close(self) -> None:
        self._handles.clear()


def _is_target(
    name: str,
    shape: tuple[int, ...],
    target_modules: list[str],
    exclude_modules: list[str],
) -> bool:
    return (
        name.endswith(".weight")
        and len(shape) == 2
        and any(module in name for module in target_modules)
        and not any(module in name for module in exclude_modules)
    )


def _targeted_names(
    source: _ModelSource, target_modules: list[str], exclude_modules: list[str]
) -> set[str]:
    return {
        name
        for name, info in source.tensors.items()
        if _is_target(name, info.shape, target_modules, exclude_modules)
    }


def _validate_sources(
    base: _ModelSource,
    others: list[tuple[str, _ModelSource]],
    target_modules: list[str],
    exclude_modules: list[str],
) -> set[str]:
    base_targets = _targeted_names(base, target_modules, exclude_modules)
    errors: list[str] = []
    for label, source in others:
        source_targets = _targeted_names(source, target_modules, exclude_modules)
        missing = sorted(base_targets - source_targets)
        extra = sorted(source_targets - base_targets)
        if missing:
            errors.append(
                f"{label} ({source.ref}) is missing targeted tensors: {missing}"
            )
        if extra:
            errors.append(
                f"{label} ({source.ref}) has targeted tensors absent from base: {extra}"
            )
        mismatched = [
            f"{name}: base {base.tensors[name].shape} vs {label} {source.tensors[name].shape}"
            for name in sorted(base_targets & source_targets)
            if base.tensors[name].shape != source.tensors[name].shape
        ]
        if mismatched:
            errors.append(f"{label} ({source.ref}) shape mismatches: {mismatched}")
    if errors:
        raise ValueError(
            "SAR model sources disagree on targeted tensors:\n" + "\n".join(errors)
        )
    return base_targets


def _resolve_device(svd_device: str) -> str:
    if svd_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if svd_device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("svd_device='cuda' requested but CUDA is not available")
    return svd_device


def _cast_for_save(
    tensor: torch.Tensor,
    name: str,
    dtype: torch.dtype,
    fallback_names: set[str],
) -> torch.Tensor:
    if not tensor.is_floating_point():
        return tensor
    if (
        dtype == torch.float16
        and tensor.dtype != torch.float16
        and tensor.abs().amax().item() > FP16_MAX
    ):
        LOG.warning("%s exceeds the float16 range; saving it in bfloat16 instead", name)
        fallback_names.add(name)
        return tensor.to(torch.bfloat16)
    return tensor.to(dtype)


def _copy_model_files(source_dir: str, output_dir: str) -> None:
    for filename in sorted(os.listdir(source_dir)):
        path = os.path.join(source_dir, filename)
        if not os.path.isfile(path):
            continue
        if any(fnmatch.fnmatch(filename, pattern) for pattern in COPY_FILE_PATTERNS):
            shutil.copy2(path, os.path.join(output_dir, filename))


def _patch_config_dtype(output_dir: str, save_dtype: str) -> None:
    config_path = os.path.join(output_dir, "config.json")
    if not os.path.isfile(config_path):
        return
    with open(config_path, encoding="utf-8") as fp:
        config = json.load(fp)
    keys = [key for key in ("dtype", "torch_dtype") if key in config] or ["dtype"]
    for key in keys:
        config[key] = save_dtype
    with open(config_path, "w", encoding="utf-8") as fp:
        json.dump(config, fp, indent=2)


class _ShardWriter:
    """Streams tensors to size-bounded safetensors shards with a final index."""

    def __init__(self, out_dir: str, max_shard_bytes: int) -> None:
        self._out_dir = out_dir
        self._max_shard_bytes = max_shard_bytes
        self._buffer: dict[str, torch.Tensor] = {}
        self._buffer_bytes = 0
        self._weight_map: dict[str, int] = {}
        self._num_flushed = 0
        self.total_bytes = 0

    def _shard_path(self, index: int) -> str:
        return os.path.join(self._out_dir, f"model-{index:05d}.safetensors")

    def _flush(self) -> None:
        if not self._buffer:
            return
        index = self._num_flushed + 1
        save_file(self._buffer, self._shard_path(index), metadata={"format": "pt"})
        for name in self._buffer:
            self._weight_map[name] = index
        self._num_flushed = index
        self._buffer = {}
        self._buffer_bytes = 0

    def add(self, name: str, tensor: torch.Tensor) -> None:
        size = tensor.numel() * tensor.element_size()
        if self._buffer and self._buffer_bytes + size > self._max_shard_bytes:
            self._flush()
        self._buffer[name] = tensor
        self._buffer_bytes += size
        self.total_bytes += size

    def finalize(self) -> None:
        self._flush()
        single_path = os.path.join(self._out_dir, SAFETENSORS_SINGLE_FILE)
        if self._num_flushed == 0:
            save_file({}, single_path, metadata={"format": "pt"})
            return
        if self._num_flushed == 1:
            os.replace(self._shard_path(1), single_path)
            return
        total = self._num_flushed
        shard_names = {
            index: f"model-{index:05d}-of-{total:05d}.safetensors"
            for index in range(1, total + 1)
        }
        for index, final_name in shard_names.items():
            os.replace(self._shard_path(index), os.path.join(self._out_dir, final_name))
        index_payload = {
            "metadata": {"total_size": self.total_bytes},
            "weight_map": {
                name: shard_names[index]
                for name, index in sorted(self._weight_map.items())
            },
        }
        with open(
            os.path.join(self._out_dir, SAFETENSORS_INDEX_FILE), "w", encoding="utf-8"
        ) as fp:
            json.dump(index_payload, fp, indent=2)


def run_sar(
    base_model: str,
    trained_model: str,
    output_dir: str,
    *,
    merge_target: str | None = None,
    rank_ratios: list[float],
    delta_rank_ratio: float | None = None,
    projection: str = "spectral",
    rewiring: str = "full",
    scale: float = 1.0,
    target_modules: list[str] | None = None,
    exclude_modules: list[str] | None = None,
    svd_device: str = "auto",
    save_dtype: str = "float16",
    save_rewiring_matrix: bool = False,
    base_model_revision: str | None = None,
    trained_model_revision: str | None = None,
    merge_target_revision: str | None = None,
) -> SARResult:
    """Stream a safetensors checkpoint pair and write SAR-projected model(s).

    Args:
        base_model: local dir or HF hub id of the spectral reference ``W0``.
        trained_model: local dir or HF hub id of the update source ``W_rl``.
        output_dir: destination; multi-ratio runs write ``rank_{ratio}`` subdirs.
        merge_target: optional expert checkpoint the projected update is added to;
            defaults to ``base_model``.
        rank_ratios: rank ratios in ``(0, 1]``; one output per ratio.
        delta_rank_ratio: optional ratio for the update truncation rank ``k'``;
            defaults to each output's rank ratio.
        projection: ``"spectral"`` or ``"none"``.
        rewiring: ``"full"``, ``"diagonal"``, or ``"off_diagonal"``.
        scale: coefficient applied to the reconstructed update.
        target_modules: parameter-name substrings to project; defaults to
            ``DEFAULT_TARGET_MODULES``.
        exclude_modules: substrings excluding otherwise-matched parameters.
        svd_device: ``"auto"``, ``"cuda"``, or ``"cpu"``.
        save_dtype: ``"float16"``, ``"bfloat16"``, or ``"float32"``.
        save_rewiring_matrix: persist per-layer rewiring matrices under ``rewiring/``.
        base_model_revision: hub revision for ``base_model``; ignored for local dirs.
        trained_model_revision: hub revision for ``trained_model``.
        merge_target_revision: hub revision for ``merge_target``.

    Returns:
        SARResult mapping each ratio to its output directory.
    """
    _validate_options(projection, rewiring)
    if not rank_ratios:
        raise ValueError("rank_ratios must contain at least one ratio")
    ratios = list(dict.fromkeys(rank_ratios))
    for ratio in ratios:
        if not 0 < ratio <= 1:
            raise ValueError(f"rank_ratio must be in (0, 1], got {ratio}")
    if delta_rank_ratio is not None and not 0 < delta_rank_ratio <= 1:
        raise ValueError(f"delta_rank_ratio must be in (0, 1], got {delta_rank_ratio}")
    if scale <= 0:
        raise ValueError(f"scale must be > 0, got {scale}")
    if isinstance(target_modules, str):
        raise ValueError("target_modules must be a list of substrings, not a string")
    if isinstance(exclude_modules, str):
        raise ValueError("exclude_modules must be a list of substrings, not a string")
    if save_dtype not in SAVE_DTYPES:
        raise ValueError(
            f"save_dtype must be one of {sorted(SAVE_DTYPES)}, got {save_dtype!r}"
        )
    dtype = SAVE_DTYPES[save_dtype]
    device = _resolve_device(svd_device)
    target_modules = (
        list(target_modules) if target_modules else list(DEFAULT_TARGET_MODULES)
    )
    exclude_modules = list(exclude_modules) if exclude_modules else []

    base_source = _load_source(base_model, base_model_revision)
    trained_source = _load_source(trained_model, trained_model_revision)
    merge_source = (
        _load_source(merge_target, merge_target_revision) if merge_target else None
    )
    target_source = merge_source if merge_source else base_source

    others: list[tuple[str, _ModelSource]] = [("trained_model", trained_source)]
    if merge_source:
        others.append(("merge_target", merge_source))
    targeted = _validate_sources(base_source, others, target_modules, exclude_modules)

    out_dirs: dict[float, str] = {
        ratio: output_dir
        if len(ratios) == 1
        else os.path.join(output_dir, f"rank_{ratio}")
        for ratio in ratios
    }
    save_rewiring = save_rewiring_matrix and projection != "none"
    if save_rewiring_matrix and projection == "none":
        LOG.warning(
            "save_rewiring_matrix requested but projection='none' produces no rewiring "
            "matrices; skipping the artifact"
        )
    # create output dirs before the (potentially hours-long) SVD pass so
    # unwritable paths fail immediately
    writers: dict[float, _ShardWriter] = {}
    for ratio, out_dir in out_dirs.items():
        os.makedirs(out_dir, exist_ok=True)
        if save_rewiring:
            os.makedirs(os.path.join(out_dir, "rewiring"), exist_ok=True)
        writers[ratio] = _ShardWriter(out_dir, MAX_SHARD_BYTES)
    m_states: dict[float, dict[str, torch.Tensor]] = {ratio: {} for ratio in ratios}
    layer_ranks: dict[float, dict[str, dict[str, int]]] = {
        ratio: {} for ratio in ratios
    }
    total_params = 0
    fp16_fallbacks: set[str] = set()

    cache = _ShardCache()
    try:
        for shard in target_source.shards:
            with safe_open(shard, framework="pt", device="cpu") as handle:
                for name in sorted(handle.keys()):
                    if name not in targeted:
                        tensor = _cast_for_save(
                            handle.get_tensor(name), name, dtype, fp16_fallbacks
                        )
                        total_params += tensor.numel()
                        for ratio in ratios:
                            writers[ratio].add(name, tensor)
                        continue

                    w_target = handle.get_tensor(name)
                    total_params += w_target.numel()
                    base32 = cache.read(base_source, name).to(
                        device=device, dtype=torch.float32
                    )
                    delta = (
                        cache.read(trained_source, name).to(
                            device=device, dtype=torch.float32
                        )
                        - base32
                    )
                    target32 = (
                        base32
                        if merge_source is None
                        else w_target.to(device=device, dtype=torch.float32)
                    )
                    min_dim = min(w_target.shape)

                    delta_u, delta_s, delta_vh = torch.linalg.svd(
                        delta, full_matrices=False
                    )
                    base_u: torch.Tensor | None = None
                    base_vh: torch.Tensor | None = None
                    if projection == "spectral":
                        base_u, _, base_vh = torch.linalg.svd(
                            base32, full_matrices=False
                        )

                    delta_k_cache: dict[int, torch.Tensor] = {}
                    for ratio in ratios:
                        rank = _rank_for(ratio, min_dim)
                        delta_rank = (
                            rank
                            if delta_rank_ratio is None
                            else _rank_for(delta_rank_ratio, min_dim)
                        )
                        delta_k = delta_k_cache.get(delta_rank)
                        if delta_k is None:
                            delta_k = _truncated_delta(
                                delta_u, delta_s, delta_vh, delta_rank
                            )
                            delta_k_cache[delta_rank] = delta_k
                        if projection == "none":
                            delta_star: torch.Tensor = delta_k
                            m: torch.Tensor | None = None
                        else:
                            assert base_u is not None and base_vh is not None
                            delta_star, m = _rewire(
                                base_u, base_vh, delta_k, rank, rewiring
                            )
                        out = target32 + scale * delta_star
                        writers[ratio].add(
                            name,
                            _cast_for_save(out.to("cpu"), name, dtype, fp16_fallbacks),
                        )
                        layer_ranks[ratio][name] = {
                            "rank": rank,
                            "delta_rank": delta_rank,
                        }
                        if save_rewiring_matrix and m is not None:
                            m_states[ratio][name] = m.to(
                                device="cpu", dtype=torch.float32
                            )
                        del delta_star, m, out
                    del (
                        w_target,
                        base32,
                        delta,
                        target32,
                        delta_u,
                        delta_s,
                        delta_vh,
                        base_u,
                        base_vh,
                        delta_k_cache,
                    )
    finally:
        cache.close()

    num_projected = len(targeted)
    m_params_by_ratio = {
        ratio: 0
        if projection == "none"
        else sum(ranks["rank"] ** 2 for ranks in layer_ranks[ratio].values())
        for ratio in ratios
    }

    for ratio in ratios:
        out_dir = out_dirs[ratio]
        writers[ratio].finalize()
        _copy_model_files(target_source.directory, out_dir)
        _patch_config_dtype(out_dir, save_dtype)

        sar_config: dict[str, Any] = {
            "base_model": base_model,
            "base_model_revision": base_source.revision,
            "trained_model": trained_model,
            "trained_model_revision": trained_source.revision,
            "merge_target": merge_target,
            "merge_target_revision": merge_source.revision if merge_source else None,
            "rank_ratio": ratio,
            "rank_ratios": ratios,
            "delta_rank_ratio": delta_rank_ratio,
            "projection": projection,
            "rewiring": rewiring,
            "scale": scale,
            "target_modules": target_modules,
            "exclude_modules": exclude_modules,
            "svd_device": device,
            "save_dtype": save_dtype,
            "bfloat16_fallback_tensors": sorted(fp16_fallbacks),
            "save_rewiring_matrix": save_rewiring_matrix,
            "num_projected": num_projected,
            "m_params": m_params_by_ratio[ratio],
            "total_params": total_params,
            "layer_ranks": layer_ranks[ratio],
        }
        with open(
            os.path.join(out_dir, "sar_config.json"), "w", encoding="utf-8"
        ) as fp:
            json.dump(sar_config, fp, indent=2)

        m_state = m_states.pop(ratio)
        if m_state:
            rewiring_dir = os.path.join(out_dir, "rewiring")
            save_torch_state_dict(
                m_state, rewiring_dir, filename_pattern="rewiring{suffix}.safetensors"
            )
            metadata = {
                "formula_version": REWIRING_FORMULA_VERSION,
                "base_model": base_model,
                "base_model_revision": base_source.revision,
                "trained_model": trained_model,
                "trained_model_revision": trained_source.revision,
                "merge_target": merge_target,
                "merge_target_revision": (
                    merge_source.revision if merge_source else None
                ),
                "rank_ratio": ratio,
                "delta_rank_ratio": delta_rank_ratio,
                "projection": projection,
                "rewiring": rewiring,
                "scale": scale,
                "layer_ranks": layer_ranks[ratio],
            }
            with open(
                os.path.join(rewiring_dir, "metadata.json"), "w", encoding="utf-8"
            ) as fp:
                json.dump(metadata, fp, indent=2)

        if projection == "none":
            LOG.info(
                "SAR rank_ratio=%s: applied low-rank delta to %d tensors "
                "(projection=none, no rewiring matrices) -> %s",
                ratio,
                num_projected,
                out_dir,
            )
        else:
            LOG.info(
                "SAR rank_ratio=%s: projected %d tensors, M params %d (%.4f%% of %d total) -> %s",
                ratio,
                num_projected,
                m_params_by_ratio[ratio],
                100.0 * m_params_by_ratio[ratio] / total_params
                if total_params
                else 0.0,
                total_params,
                out_dir,
            )

    return SARResult(
        outputs=out_dirs,
        num_projected=num_projected,
        m_params=m_params_by_ratio[max(ratios)],
        total_params=total_params,
    )
