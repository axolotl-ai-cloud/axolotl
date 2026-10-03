"""Dataset preparation for the model-agnostic diffusion-decision plugin."""

from __future__ import annotations

import hashlib
import json
import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from glob import glob
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from datasets import load_dataset
from torch.utils.data import Dataset

from axolotl.common.datasets import TrainDatasetMeta
from axolotl.integrations.diffusion.lm.sampling import resolve_native_packing_budget
from axolotl.loaders import load_tokenizer
from axolotl.model_support import (
    DiffusionNoise,
    get_model_support_for_cfg,
    resolve_model_support,
)
from axolotl.utils.dict import DictDefault

from .adapters import normalize_record
from .budgets import decision_budget, exceeds_budget
from .data_audit import build_preparation_audit, write_preparation_audit
from .grouping import group_records
from .hygiene import assert_split_isolation, decontaminate, family_key
from .loss import decision_example_from_canvas
from .mixture import (
    DeterministicMixtureSampler,
    MixtureSample,
    prepare_source_pools,
)
from .permutation import permute_record
from .prepared_cache import (
    identity as prepared_cache_identity,
    load as load_prepared_cache,
    store as store_prepared_cache,
)
from .preprocessing import build_decision_canvas
from .template import SchemaError

LOG = logging.getLogger(__name__)


def _value(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _rows(entry: Any) -> list[dict[str, Any]]:
    path = _value(entry, "path")
    if not isinstance(path, str) or not path:
        raise ValueError("decision dataset requires a path")
    split = str(_value(entry, "split", "train"))
    kwargs = {
        key: _value(entry, key)
        for key in ("name", "revision")
        if _value(entry, key) is not None
    }
    data_files = _value(entry, "data_files")
    local_jsonl = _local_jsonl_paths(path, data_files, split)
    if local_jsonl is not None:
        return _read_jsonl(local_jsonl)
    if (
        data_files is not None
        and path in {"json", "parquet", "csv", "text"}
        and not isinstance(data_files, Mapping)
    ):
        kwargs["data_files"] = {split: data_files}
    elif data_files is not None:
        kwargs["data_files"] = data_files
    dataset = load_dataset(path, split=split, **kwargs)
    return [dict(row) for row in dataset]


def _local_jsonl_paths(path: str, data_files: Any, split: str) -> list[Path] | None:
    if path != "json" or data_files is None:
        return None
    selected = data_files.get(split) if isinstance(data_files, Mapping) else data_files
    if isinstance(selected, str):
        names = [selected]
    elif isinstance(selected, Sequence) and not isinstance(selected, (str, bytes)):
        names = list(selected)
    else:
        return None
    if not names or not all(isinstance(name, str) for name in names):
        raise ValueError("local JSONL data_files must be a nonempty path or path list")
    expanded: list[Path] = []
    for name in names:
        if urlparse(name).scheme:
            return None
        matches = sorted(Path(match) for match in glob(name))
        if matches:
            if any(match.suffix != ".jsonl" for match in matches):
                return None
            expanded.extend(matches)
        elif Path(name).suffix == ".jsonl":
            raise ValueError(f"local JSONL path matched no files: {name}")
        else:
            return None
    if not expanded or not all(item.is_file() for item in expanded):
        raise ValueError("local JSONL data_files must resolve to regular files")
    return expanded


def _prepared_cache_source_paths(entries: Sequence[Any]) -> list[Path] | None:
    paths: list[Path] = []
    for entry in entries:
        resolved = _local_jsonl_paths(
            _value(entry, "path"),
            _value(entry, "data_files"),
            str(_value(entry, "split", "train")),
        )
        if resolved is None:
            return None
        paths.extend(resolved)
    return paths


def _prepared_cache_eligible(cfg: Any) -> bool:
    latent = _value(_value(cfg, "decision"), "latent")
    return latent is None or _value(latent, "mode", "none") in {"none", "pad", "mask"}


def _premix_identity(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record.get("source", "")),
        str(record.get("id", "")),
        str(record.get("group", "")),
    )


def _validate_premixed_records(records: Sequence[Mapping[str, Any]]) -> None:
    draw_origins: dict[str, str] = {}
    content_by_origin: dict[tuple[str, str, str], str] = {}
    for record in records:
        metadata = record.get("source_metadata")
        premix = metadata.get("premix") if isinstance(metadata, Mapping) else None
        if not isinstance(premix, Mapping):
            raise ValueError("premixed decision rows require source_metadata.premix")
        draw_id = premix.get("draw_id")
        origin = premix.get("origin")
        if not isinstance(draw_id, str) or not draw_id:
            raise ValueError("premixed decision rows require a nonempty premix draw_id")
        if not isinstance(origin, Mapping) or not origin:
            raise ValueError("premixed decision rows require a premix origin mapping")
        identity = _premix_identity(record)
        if (
            tuple(str(origin.get(field, "")) for field in ("source", "id", "group"))
            != identity
        ):
            raise ValueError(
                "premixed decision origin must match source, id, and group"
            )
        origin_value = json.dumps(origin, sort_keys=True, separators=(",", ":"))
        if draw_id in draw_origins:
            raise ValueError(f"premixed decision draw_id is duplicated: {draw_id}")
        draw_origins[draw_id] = origin_value
        content = dict(record)
        content_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        content_metadata.pop("premix", None)
        content["source_metadata"] = content_metadata
        canonical_content = json.dumps(
            content, sort_keys=True, separators=(",", ":"), default=str
        )
        previous = content_by_origin.setdefault(identity, canonical_content)
        if previous != canonical_content:
            raise ValueError(
                "premixed copies of one decision record must have identical content"
            )


def _read_jsonl(paths: Sequence[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid JSONL at {path}:{line_number}: {error.msg}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"JSONL record at {path}:{line_number} must be an object"
                    )
                records.append(record)
    return records


def _is_eval(entry: Any) -> bool:
    return str(_value(entry, "split", "train")).lower() in {
        "eval",
        "test",
        "validation",
        "dev",
        "calibration",
        "ood",
    }


def _family_dev_split(records: Sequence[dict[str, Any]], ratio: float = 0.2):
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[family_key(record)].append(record)
    selected: set[tuple[str, str]] = set()
    target = round(len(records) * ratio)
    total = 0
    for key in sorted(
        groups,
        key=lambda item: hashlib.sha256(f"decision-dev-v1:{item}".encode()).hexdigest(),
    ):
        if total >= target:
            break
        selected.add(key)
        total += len(groups[key])
    return (
        [record for record in records if family_key(record) not in selected],
        [record for record in records if family_key(record) in selected],
    )


def _token_ids(value: Any, name: str) -> tuple[int, ...]:
    if value is None:
        return ()
    values = (
        (value,) if isinstance(value, int) and not isinstance(value, bool) else value
    )
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be an integer sequence")
    result: list[int] = []
    for token_id in values:
        if isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0:
            raise ValueError(f"{name} must contain nonnegative integers")
        result.append(token_id)
    return tuple(result)


def _required_token_id(value: Any, name: str) -> int:
    token_ids = _token_ids(value, name)
    if len(token_ids) != 1:
        raise ValueError(f"{name} must resolve to exactly one token ID")
    return token_ids[0]


def _mask_token_id(diffusion: Any, model_config: Any) -> int | None:
    value = _value(diffusion, "mask_token_id")
    if value is None:
        value = _value(model_config, "mask_token_id")
    token_ids = _token_ids(value, "mask_token_id")
    if len(token_ids) > 1:
        raise ValueError("mask_token_id must resolve to at most one token ID")
    return token_ids[0] if token_ids else None


def _thought_delimiters(model_config: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        _token_ids(_value(model_config, "thought_open_ids"), "thought_open_ids"),
        _token_ids(_value(model_config, "thought_close_ids"), "thought_close_ids"),
    )


def _active_control_ids(
    tokenizer: Any,
    model_config: Any,
    mask_token_id: int | None,
    thought_open_ids: Sequence[int],
    thought_close_ids: Sequence[int],
) -> set[int]:
    control_ids = set(thought_open_ids) | set(thought_close_ids)
    for source, prefix in ((tokenizer, "tokenizer"), (model_config, "model_config")):
        for name in ("pad_token_id", "bos_token_id", "eos_token_id", "unk_token_id"):
            control_ids.update(_token_ids(_value(source, name), f"{prefix}.{name}"))
    if mask_token_id is not None:
        control_ids.add(mask_token_id)
    return control_ids


def _canvas_row(
    tokenizer,
    record: dict[str, Any],
    cfg: Any,
    source_weight: float,
    *,
    spec,
    vocab_size: int | None = None,
):
    diffusion = _value(cfg, "diffusion")
    decision = _value(cfg, "decision")
    model_config = _value(cfg, "model_config")
    configured_width = _value(diffusion, "canvas_width")
    width = 128 if configured_width is None else int(configured_width)
    if spec.max_canvas is not None:
        width = min(width, spec.max_canvas)
    mask_token_id = _mask_token_id(diffusion, model_config)
    record = permute_record(
        record,
        seed=int(_value(cfg, "seed", 0)),
        codebook=_value(_value(decision, "labels"), "codebook", "vendored26"),
    )
    if vocab_size is None:
        vocab_size = _value(model_config, "vocab_size")
    if vocab_size is None:
        vocab_size = len(tokenizer)
    turn_close_id = _required_token_id(
        _value(model_config, "turn_close_token_id", _value(tokenizer, "eos_token_id")),
        "turn_close_token_id",
    )
    canvas = build_decision_canvas(
        tokenizer,
        record,
        scaffold_ids=(),
        turn_close_id=turn_close_id,
        pad_id=_required_token_id(
            _value(tokenizer, "pad_token_id"), "tokenizer.pad_token_id"
        ),
        vocab_size=int(vocab_size),
        width=width,
        seed=int(_value(cfg, "seed", 0)),
        steps=int(_value(_value(diffusion, "unroll"), "k_max", 1)),
        noise_kind="absorbing" if spec.noise is DiffusionNoise.ABSORBING else "uniform",
        mask_token_id=mask_token_id,
        codebook=_value(_value(decision, "labels"), "codebook", "vendored26"),
        prevalidated_record=True,
    )
    row = {
        "canvas": canvas,
        "decision_example": decision_example_from_canvas(
            canvas, source_weight=source_weight
        ),
        "source": record["source"],
        "length": len(canvas.prompt_ids) + len(canvas.canvas_ids),
        "record": record,
        "max_questions": int(_value(decision, "max_questions_per_canvas", 20)),
    }
    return row


def _build_canvas_worker(
    item: tuple[int, dict[str, Any], float],
    *,
    tokenizer: Any,
    cfg: Any,
    spec: Any,
    vocab_size: int,
) -> tuple[int, dict[str, Any] | None, bool]:
    index, record, source_weight = item
    try:
        row = _canvas_row(
            tokenizer,
            record,
            cfg,
            source_weight,
            spec=spec,
            vocab_size=vocab_size,
        )
    except SchemaError as error:
        if not _is_canvas_overflow(error):
            raise
        return index, None, True
    return index, row, False


class DecisionDataset(Dataset):
    """In-memory rows retaining typed canvases and decision-label examples."""

    def __init__(
        self,
        rows: Sequence[dict[str, Any]],
        manifest: Mapping[str, Any],
        *,
        expose_lengths: bool = True,
    ):
        self._rows = tuple(rows)
        self.manifest = dict(manifest)
        self.column_names = ("length",) if expose_lengths else ()

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(
        self, index: int | str | Sequence[int]
    ) -> dict[str, Any] | list[int] | list[dict[str, Any]]:
        if isinstance(index, str):
            if index != "length":
                raise KeyError(index)
            return [int(row["length"]) for row in self._rows]
        if isinstance(index, int):
            return self._rows[index]
        if isinstance(index, Sequence):
            rows = []
            for item in index:
                if not isinstance(item, int):
                    raise TypeError("packed decision indices must be integers")
                rows.append(self._rows[item])
            return rows
        raise TypeError("decision indices must be integers, sequences, or column names")

    def remove_columns(self, columns: Sequence[str]) -> "DecisionDataset":
        if tuple(columns) != ("length",) or "length" not in self.column_names:
            raise ValueError("DecisionDataset only exposes the derived length column")
        return DecisionDataset(self._rows, self.manifest, expose_lengths=False)


def _is_canvas_overflow(error: SchemaError) -> bool:
    message = str(error)
    return (
        message.startswith("answer template is ") and " canvas holds " in message
    ) or (
        message.startswith("question ")
        and " alone needs " in message
        and message.endswith(" canvas rows")
    )


def _canvas_rows(
    tokenizer: Any,
    records: Sequence[dict[str, Any]],
    cfg: Any,
    weights: Mapping[str, float],
    spec,
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    canvas_too_long = 0
    vocab_size = _value(_value(cfg, "model_config"), "vocab_size")
    if vocab_size is None:
        vocab_size = len(tokenizer)
    workers = _value(cfg, "dataset_num_proc", 1) or 1
    workers = int(workers)
    if workers > 1:
        LOG.info(
            "Preparing %s decision canvases with %s worker threads",
            len(records),
            workers,
        )
        work = (
            (index, record, float(weights.get(record["source"], 1.0)))
            for index, record in enumerate(records)
        )
        worker = partial(
            _build_canvas_worker,
            tokenizer=tokenizer,
            cfg=cfg,
            spec=spec,
            vocab_size=int(vocab_size),
        )
        with ThreadPoolExecutor(max_workers=workers) as executor:
            prepared = list(executor.map(worker, work))
        prepared.sort(key=lambda item: item[0])
        rows = [row for _, row, _ in prepared if row is not None]
        return rows, sum(overflow for _, _, overflow in prepared)
    for index, record in enumerate(records):
        if index % 1000 == 0:
            LOG.info("Preparing decision canvases: %s/%s", index, len(records))
        try:
            rows.append(
                _canvas_row(
                    tokenizer,
                    record,
                    cfg,
                    float(weights.get(record["source"], 1.0)),
                    spec=spec,
                    vocab_size=int(vocab_size),
                )
            )
        except SchemaError as error:
            if not _is_canvas_overflow(error):
                raise
            canvas_too_long += 1
    return rows, canvas_too_long


def _filter_budget_rows(
    rows: Sequence[dict[str, Any]], cfg: Any, spec, *, is_eval: bool = False
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    logical_limit = _value(cfg, "sequence_len")
    sample_packed = bool(_value(cfg, "sample_packing", False)) and (
        not is_eval or _value(cfg, "eval_sample_packing") is not False
    )
    packed = bool(_value(cfg, "batch_flattening", False)) or sample_packed
    resolved = (
        resolve_native_packing_budget(
            DictDefault(cfg) if isinstance(cfg, Mapping) else cfg,
            packed=packed,
            batch_size=(
                _value(cfg, "micro_batch_size")
                if sample_packed
                else _value(cfg, "eval_batch_size")
                if is_eval
                else _value(cfg, "micro_batch_size")
            ),
        )
        if packed
        else None
    )
    physical_limit = None if resolved is None else resolved.payload_capacity
    kept: list[dict[str, Any]] = []
    drops = {"logical": 0, "physical": 0}
    for row in rows:
        canvas = row["canvas"]
        reason = exceeds_budget(
            decision_budget(
                prompt_tokens=len(canvas.prompt_ids),
                canvas_tokens=len(canvas.canvas_ids),
                layout=spec.layout,
            ),
            logical_limit=None if logical_limit is None else int(logical_limit),
            physical_limit=None if physical_limit is None else int(physical_limit),
        )
        if reason is None:
            kept.append(row)
        else:
            drops[reason] += 1
    return kept, drops


_DECISION_EVALUATION_SPLITS = frozenset(
    {"dev", "validation", "eval", "test", "calibration", "ood"}
)


def _evaluation_source_input(
    entry: Any, *, collection: str, index: int
) -> dict[str, Any]:
    item = {
        "collection": collection,
        "index": index,
        "adapter": _value(entry, "type"),
        "path": _value(entry, "path"),
        "revision": _value(entry, "revision"),
        "name": _value(entry, "name"),
        "split": _value(entry, "split", "train"),
    }
    data_files = _value(entry, "data_files")
    if data_files is not None:
        item["data_files"] = data_files
    return item


def _evaluation_entries(cfg: Any, split: str) -> list[tuple[str, int, Any]]:
    if split not in _DECISION_EVALUATION_SPLITS:
        choices = ", ".join(sorted(_DECISION_EVALUATION_SPLITS))
        raise ValueError(f"decision evaluation split must be one of: {choices}")
    selected: list[tuple[str, int, Any]] = []
    for collection in ("test_datasets", "datasets"):
        entries = _value(cfg, collection, ()) or ()
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            raise ValueError(f"{collection} must be a sequence")
        for index, entry in enumerate(entries):
            if not str(_value(entry, "type", "")).startswith("decision."):
                continue
            if str(_value(entry, "split", "train")).lower() != split:
                continue
            if collection == "datasets" and not _is_eval(entry):
                continue
            selected.append((collection, index, entry))
    if not selected:
        raise ValueError(
            f"decision has no explicitly declared evaluation dataset for split={split!r}"
        )
    return selected


def load_decision_evaluation_dataset(
    cfg: Any, tokenizer: Any, split: str
) -> DecisionDataset:
    """Prepare only explicitly declared evaluation sources for one original split."""
    selected_split = str(split).lower()
    selected = _evaluation_entries(cfg, selected_split)
    decision = _value(cfg, "decision")
    profile = resolve_model_support(get_model_support_for_cfg(cfg))
    spec = None if profile is None else profile.diffusion
    if spec is None:
        raise ValueError("decision requires a resolved DiffusionSpec")
    weights = _value(_value(decision, "mixture"), "loss_weight", {}) or {}
    max_questions = int(_value(decision, "max_questions_per_canvas", 20))
    prepared_path = _value(cfg, "dataset_prepared_path")
    cache_identity = None
    selected_entries = [
        (collection, index, entry) for collection, index, entry in selected
    ]
    source_inputs = [
        _evaluation_source_input(entry, collection=collection, index=index)
        for collection, index, entry in selected
    ]
    if (
        isinstance(prepared_path, (str, Path))
        and str(prepared_path)
        and _prepared_cache_eligible(cfg)
    ):
        source_paths = _prepared_cache_source_paths(
            [entry for _, _, entry in selected_entries]
        )
        if source_paths is not None:
            cache_identity = prepared_cache_identity(
                cfg,
                tokenizer,
                spec,
                source_paths,
                selected_entries=selected_entries,
                scope={"kind": "explicit_evaluation", "split": selected_split},
            )
    if cache_identity is not None:
        cache_key, cache_payload = cache_identity
        cached = load_prepared_cache(
            Path(prepared_path), cache_key, cache_payload, include_audit=True
        )
        if cached is not None:
            _cached_train, cached_rows, cached_output_manifest, cached_audit = cached
            cached_manifest = dict(cached_output_manifest)
            filename = f"decision_{selected_split}_preparation_audit.json"
            cached_manifest["preparation_audit"] = str(
                write_preparation_audit(
                    cfg,
                    (),
                    cached_rows,
                    cached_manifest,
                    source_inputs=source_inputs,
                    filename=filename,
                    audit=(
                        cached_audit
                        if cached_audit.get("source_inputs") == source_inputs
                        else None
                    ),
                )
            )
            LOG.info(
                "Decision prepared-row cache hit for evaluation split %s",
                selected_split,
            )
            return DecisionDataset(cached_rows, cached_manifest)
        LOG.info(
            "Decision prepared-row cache miss for evaluation split %s", selected_split
        )
    normalized: list[dict[str, Any]] = []
    for _collection, _index, entry in selected:
        adapter = str(_value(entry, "type"))
        normalized.extend(
            normalize_record(
                adapter,
                row,
                training=False,
                source_split=str(_value(entry, "split", "train")).lower(),
                target_basis=_value(
                    _value(decision, "labels"), "open_jev_target_basis"
                ),
                codebook=_value(_value(decision, "labels"), "codebook", "vendored26"),
            )
            for row in _rows(entry)
        )
    grouped = group_records(normalized, max_questions=max_questions)
    rows, canvas_too_long = _canvas_rows(tokenizer, grouped, cfg, weights, spec)
    rows, budget_drops = _filter_budget_rows(rows, cfg, spec, is_eval=True)
    manifest: dict[str, Any] = {
        "selected_split": selected_split,
        "selected_source_inputs": source_inputs,
        "input_rows": len(normalized),
        "grouped_rows": len(grouped),
        "eval_rows": len(rows),
        "eval_sources": sorted({row["source"] for row in rows}),
        "canvas_too_long": {"eval": canvas_too_long},
        "budget_drops": {"eval": budget_drops},
    }
    if _value(cfg, "dataset_prepared_path") or _value(cfg, "output_dir"):
        filename = f"decision_{selected_split}_preparation_audit.json"
        audit = build_preparation_audit(
            cfg, (), rows, manifest, source_inputs=source_inputs
        )
        manifest["preparation_audit"] = str(
            write_preparation_audit(
                cfg,
                (),
                rows,
                manifest,
                source_inputs=source_inputs,
                filename=filename,
                audit=audit,
            )
        )
    else:
        audit = None
    if cache_identity is not None:
        store_prepared_cache(
            Path(prepared_path),
            cache_key,
            cache_payload,
            cfg,
            (),
            rows,
            manifest,
            audit=audit,
        )
    return DecisionDataset(rows, manifest)


def load_decision_datasets(
    cfg: Any, preprocess: bool = False, *, tokenizer: Any = None
) -> TrainDatasetMeta:
    """Load, normalize, decontaminate, canvasize, and return plugin-owned splits."""
    del preprocess
    decision = _value(cfg, "decision")
    profile = resolve_model_support(get_model_support_for_cfg(cfg))
    spec = None if profile is None else profile.diffusion
    if spec is None:
        raise ValueError("decision requires a resolved DiffusionSpec")
    train_entries = [
        entry
        for entry in (_value(cfg, "datasets", []) or [])
        if str(_value(entry, "type", "")).startswith("decision.")
    ]
    test_entries = [
        entry
        for entry in (_value(cfg, "test_datasets", []) or [])
        if str(_value(entry, "type", "")).startswith("decision.")
    ]
    if not train_entries:
        raise ValueError("decision requires at least one decision dataset")
    if tokenizer is None:
        tokenizer = load_tokenizer(cfg)
    prepared_path = _value(cfg, "dataset_prepared_path")
    cache_identity = None
    if (
        isinstance(prepared_path, (str, Path))
        and str(prepared_path)
        and _prepared_cache_eligible(cfg)
    ):
        source_paths = _prepared_cache_source_paths([*train_entries, *test_entries])
        if source_paths is not None:
            cache_identity = prepared_cache_identity(cfg, tokenizer, spec, source_paths)
        else:
            LOG.info("Decision prepared-row cache bypassed: nonlocal or opaque source")
    elif isinstance(prepared_path, (str, Path)) and str(prepared_path):
        LOG.info("Decision prepared-row cache bypassed: unsupported latent mode")
    if cache_identity is not None:
        cache_key, cache_payload = cache_identity
        cached = load_prepared_cache(
            Path(prepared_path), cache_key, cache_payload, include_audit=True
        )
        if cached is not None:
            cached_train, cached_eval, cached_manifest, cached_audit = cached
            write_preparation_audit(
                cfg, cached_train, cached_eval, cached_manifest, audit=cached_audit
            )
            LOG.info("Decision prepared-row cache hit")
            return TrainDatasetMeta(
                train_dataset=DecisionDataset(cached_train, cached_manifest),
                eval_dataset=DecisionDataset(cached_eval, cached_manifest)
                if cached_eval
                else None,
                total_num_steps=None,
            )
        LOG.info("Decision prepared-row cache miss")
    train: list[dict[str, Any]] = []
    explicit_eval: list[dict[str, Any]] = []
    protected_eval: list[dict[str, Any]] = []
    for entry, force_dev in [
        *((entry, False) for entry in train_entries),
        *((entry, True) for entry in test_entries),
    ]:
        adapter = str(_value(entry, "type"))
        LOG.info(
            "Loading decision source %s (%s)",
            _value(entry, "path"),
            _value(entry, "split", "train"),
        )
        normalized = [
            normalize_record(
                adapter,
                row,
                training=not (_is_eval(entry) or force_dev),
                source_split=str(_value(entry, "split", "train")).lower(),
                target_basis=_value(
                    _value(decision, "labels"), "open_jev_target_basis"
                ),
                codebook=_value(_value(decision, "labels"), "codebook", "vendored26"),
            )
            for row in _rows(entry)
        ]
        split = str(_value(entry, "split", "train")).lower()
        if split in {"test", "calibration", "ood"}:
            protected_eval.extend(normalized)
        elif force_dev or split in {"dev", "validation", "eval"}:
            explicit_eval.extend(normalized)
        elif _is_eval(entry):
            protected_eval.extend(normalized)
        else:
            train.extend(normalized)
    if explicit_eval:
        dev = explicit_eval
    else:
        train, dev = _family_dev_split(train)
    LOG.info(
        "Decontaminating %s training records against %s development and %s protected records",
        len(train),
        len(dev),
        len(protected_eval),
    )
    mixture = _value(decision, "mixture")
    premixed = bool(_value(mixture, "premixed", False))
    retained, drops = decontaminate(train, [*dev, *protected_eval])
    if premixed and len(retained) != len(train):
        raise ValueError("premixed decision rows overlap an evaluation split")
    train = [dict(record) for record in retained]
    max_questions = int(_value(decision, "max_questions_per_canvas", 20))
    if premixed:
        _validate_premixed_records(train)
    else:
        LOG.info("Grouping compatible decision records")
        train = group_records(train, max_questions=max_questions)
    dev = group_records(dev, max_questions=max_questions)
    assert_split_isolation({"train": train, "eval": dev})
    weights = _value(_value(decision, "mixture"), "loss_weight", {}) or {}
    train_canvas_too_long = 0
    train_budget_drops = {"logical": 0, "physical": 0}
    pools = None
    sampled_batches: tuple[tuple[MixtureSample[dict[str, Any]], ...], ...] = ()
    sampler = None
    if premixed:
        train_rows, train_canvas_too_long = _canvas_rows(
            tokenizer, [dict(record) for record in train], cfg, weights, spec
        )
        train_rows, train_budget_drops = _filter_budget_rows(train_rows, cfg, spec)
        if len(train_rows) != len(train):
            raise ValueError(
                "premixed decision rows must all pass canvas and budget validation"
            )
    else:
        pools = prepare_source_pools(
            train,
            (),
            max_examples_per_source=_value(mixture, "max_examples_per_source"),
            seed=int(_value(cfg, "seed", 0)),
        )
        usable_pools: dict[str, tuple[dict[str, Any], ...]] = {}
        for source, records in pools.records.items():
            rows, dropped = _canvas_rows(
                tokenizer, [dict(record) for record in records], cfg, weights, spec
            )
            train_canvas_too_long += dropped
            rows, budget_drops = _filter_budget_rows(rows, cfg, spec)
            for reason, count in budget_drops.items():
                train_budget_drops[reason] += count
            if rows:
                usable_pools[source] = tuple(rows)
        configured_weights = _value(mixture, "weights")
        if isinstance(configured_weights, Mapping):
            configured_weights = {
                source: configured_weights[source]
                for source in usable_pools
                if source in configured_weights
            }
        sampler = DeterministicMixtureSampler(
            usable_pools,
            weights=configured_weights,
            temperature=_value(mixture, "temperature"),
            seed=int(_value(cfg, "seed", 0)),
        )
        batch_size = int(_value(cfg, "micro_batch_size", 1))
        sampled_batches = tuple(
            sampler.epoch_batches(
                batch_size,
                stratified=bool(_value(mixture, "per_batch_stratified", True)),
            )
        )
        train_rows = [
            dict(sample.record) for batch in sampled_batches for sample in batch
        ]
    eval_rows, eval_canvas_too_long = _canvas_rows(tokenizer, dev, cfg, weights, spec)
    eval_rows, eval_budget_drops = _filter_budget_rows(
        eval_rows, cfg, spec, is_eval=True
    )
    drop_counts = dict(drops)
    if pools is not None:
        drop_counts.update(
            {
                key: value
                for key, value in pools.counts.items()
                if key not in drop_counts
            }
        )
    manifest = {
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "protected_eval_rows": len(protected_eval),
        "train_sources": sorted({row["source"] for row in train}),
        "eval_sources": sorted({row["source"] for row in dev}),
        "dropped": drop_counts,
        "canvas_too_long": {
            "train": train_canvas_too_long,
            "eval": eval_canvas_too_long,
        },
        "budget_drops": {"train": train_budget_drops, "eval": eval_budget_drops},
        "mixture_probabilities": dict(sampler.probabilities) if sampler else {},
        "premixed": premixed,
        "per_batch_stratified": bool(_value(mixture, "per_batch_stratified", True)),
        "stratified_micro_batch_size": int(_value(cfg, "micro_batch_size", 1)),
        "mixture_seed": int(_value(cfg, "seed", 0)),
        "stratified_epoch_batches": tuple(
            tuple(sample.source for sample in batch) for batch in sampled_batches
        ),
    }
    if any(
        _value(record.get("source_metadata", {}), "official_split") is not None
        for record in [*train, *dev, *protected_eval]
        if record["source"] == "procedural"
    ):
        manifest["procedural_grouping_limit"] = (
            "official partitions and state decontamination only; no parent-family key"
        )
    if _value(cfg, "dataset_prepared_path") or _value(cfg, "output_dir"):
        write_preparation_audit(cfg, train_rows, eval_rows, manifest)
    if cache_identity is not None:
        store_prepared_cache(
            Path(prepared_path),
            cache_key,
            cache_payload,
            cfg,
            train_rows,
            eval_rows,
            manifest,
        )
    LOG.info(
        "Prepared %s training draws and %s evaluation rows; budget drops: %s",
        len(train_rows),
        len(eval_rows),
        manifest["budget_drops"],
    )
    return TrainDatasetMeta(
        train_dataset=DecisionDataset(train_rows, manifest),
        eval_dataset=DecisionDataset(eval_rows, manifest) if eval_rows else None,
        total_num_steps=None,
    )
