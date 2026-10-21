"""Dataset preparation for the model-agnostic diffusion-decision plugin."""

from __future__ import annotations

import hashlib
import logging
import multiprocessing
from collections import defaultdict
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from glob import glob
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from datasets import load_dataset
from torch.utils.data import Dataset

from axolotl.common.datasets import TrainDatasetMeta
from axolotl.loaders import load_tokenizer
from axolotl.model_support import DiffusionNoise  # noqa: F401  (re-exported)

from ._util import (
    _value,
    canonical_json,
    model_config_overrides,
    read_jsonl,
    require_diffusion_spec,
)
from .adapters import normalize_record
from .batch_plan import (
    DecisionBatchPlan,
    SourcePool,
    encode_reserve,
)
from .canvas import (  # noqa: F401  (canvas building lives in .canvas; re-exported)
    _budget_limits,
    _canvas_row,
    _canvas_settings,
    _canvas_width,
    _filter_budget_rows,
    _grouping_fits,
    _is_canvas_overflow,
    _prepared_segments,
    _read_overflow_width,
    _reserve_segments,
    _resolve_slot_plan,
    _scaffold_ids,
    _thought_delimiters,
    canvas_prefix_ids,
)
from .grouping import group_records
from .hygiene import assert_split_isolation, decontaminate, family_key
from .loss import decision_example_from_canvas
from .mixture import cap_window, prepare_source_pools
from .prepared_cache import (
    identity as prepared_cache_identity,
    load as load_prepared_cache,
    prepared_cache_verify,
    store as store_prepared_cache,
)
from .prompts import resolve_image_refs
from .segments import label_tables, segment_safety
from .slot_sampling import (
    DecisionDraw,
    project_max_canvas,
    project_slot_plan,
    sample_slot_count,
)
from .slots import SlotPlan
from .template import SchemaError

LOG = logging.getLogger(__name__)


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
        return [
            _with_resolved_images(record, path.parent)
            for path in local_jsonl
            for record in read_jsonl(path)
        ]
    if (
        data_files is not None
        and path in {"json", "parquet", "csv", "text"}
        and not isinstance(data_files, Mapping)
    ):
        kwargs["data_files"] = {split: data_files}
    elif data_files is not None:
        kwargs["data_files"] = data_files
    dataset = load_dataset(path, split=split, **kwargs)
    base_dir = _local_data_dir(path, data_files, split)
    return [_with_resolved_images(dict(row), base_dir) for row in dataset]


def _local_data_dir(path: str, data_files: Any, split: str) -> Path | None:
    """The single directory holding a local source's data files, else None."""
    if data_files is None:
        names = [path]
    else:
        selected = (
            data_files.get(split) if isinstance(data_files, Mapping) else data_files
        )
        names = [selected] if isinstance(selected, str) else list(selected or ())
        if Path(path).is_dir():
            names = [str(Path(path) / str(name)) for name in names]
    parents: set[Path] = set()
    for name in names:
        if not isinstance(name, str) or urlparse(name).scheme:
            return None
        matches = [Path(match) for match in glob(name)]
        if not matches:
            return None
        parents.update(
            (match if match.is_dir() else match.parent).resolve() for match in matches
        )
    return parents.pop() if len(parents) == 1 else None


def _with_resolved_images(
    record: dict[str, Any], base_dir: Path | None
) -> dict[str, Any]:
    """Relative image paths are relative to the data file's directory, never to cwd."""
    images = record.get("images")
    if isinstance(images, list) and images:
        if base_dir is None and any(
            isinstance(ref, str)
            and ref
            and urlparse(ref).scheme not in {"http", "https"}
            and not Path(ref).is_absolute()
            for ref in images
        ):
            raise ValueError(
                "relative decision image paths need a local data file source in a "
                "single directory; use absolute paths or http(s) URLs"
            )
        record = {**record, "images": list(resolve_image_refs(images, base_dir))}
    return record


def _prepared_cache_image_paths(source_paths: Sequence[Path]) -> list[Path] | None:
    paths: dict[Path, None] = {}
    for source in source_paths:
        for record in read_jsonl(source):
            images = record.get("images")
            if not isinstance(images, list):
                continue
            for ref in resolve_image_refs(images, source.parent):
                if urlparse(ref).scheme:
                    return None
                image = Path(ref)
                if not image.is_file():
                    raise ValueError(f"decision image not found: {ref}")
                paths[image] = None
    return list(paths)


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
    latent = _value(_value(cfg, "diffusion_decision"), "latent")
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
        origin_value = canonical_json(origin, ensure_ascii=True)
        if draw_id in draw_origins:
            raise ValueError(f"premixed decision draw_id is duplicated: {draw_id}")
        draw_origins[draw_id] = origin_value
        content = dict(record)
        content_metadata = dict(metadata) if isinstance(metadata, Mapping) else {}
        content_metadata.pop("premix", None)
        content["source_metadata"] = content_metadata
        canonical_content = canonical_json(content, ensure_ascii=True, default=str)
        previous = content_by_origin.setdefault(identity, canonical_content)
        if previous != canonical_content:
            raise ValueError(
                "premixed copies of one decision record must have identical content"
            )


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
        size = len(groups[key])
        if total + size >= len(records):
            continue
        selected.add(key)
        total += size
    if records:
        LOG.info(
            "Automatic decision dev split holds out %s of %s records",
            total,
            len(records),
        )
    return (
        [record for record in records if family_key(record) not in selected],
        [record for record in records if family_key(record) in selected],
    )


_CANVAS_CHUNK_SIZE = 64
_CANVAS_WORKER_STATE: tuple[Any, ...] | None = None


def _canvas_mp_context() -> Any:
    # Fork (as ``datasets.map(num_proc=...)`` uses on Linux) inherits the loaded
    # modules and tokenizer; spawned workers would each re-import torch and
    # axolotl, which takes close to a minute per worker.
    method = "fork" if "fork" in multiprocessing.get_all_start_methods() else "spawn"
    return multiprocessing.get_context(method)


def _init_canvas_worker(state: tuple[Any, ...]) -> None:
    global _CANVAS_WORKER_STATE
    _CANVAS_WORKER_STATE = state


# Item modes: "row" builds a canvas row; "segments" also pre-tokenizes the row's
# segments for re-permuted training draws; "reserve" pre-tokenizes only the
# segments of a record whose canvas is built when a later epoch draws it.
_ROW, _ROW_SEGMENTS, _RESERVE = "row", "segments", "reserve"


def _build_canvas_chunk(
    items: Sequence[tuple[int, Mapping[str, Any], float, str, bool]],
) -> list[tuple[dict[str, Any] | None, bool, Any, str | None]]:
    if _CANVAS_WORKER_STATE is None:
        raise RuntimeError("decision canvas worker was not initialized")
    (
        tokenizer,
        cfg,
        spec,
        vocab_size,
        slot_plan,
        thought_open_ids,
        thought_close_ids,
        train,
    ) = _CANVAS_WORKER_STATE
    return [
        _build_canvas_worker(
            item,
            tokenizer=tokenizer,
            cfg=cfg,
            spec=spec,
            vocab_size=vocab_size,
            slot_plan=slot_plan,
            thought_open_ids=thought_open_ids,
            thought_close_ids=thought_close_ids,
            train=train,
        )[1:]
        for item in items
    ]


def _build_canvas_worker(
    item: tuple[int, Mapping[str, Any], float]
    | tuple[int, Mapping[str, Any], float, str]
    | tuple[int, Mapping[str, Any], float, str, bool],
    *,
    tokenizer: Any,
    cfg: Any,
    spec: Any,
    vocab_size: int,
    slot_plan: SlotPlan | None,
    thought_open_ids: tuple[int, ...],
    thought_close_ids: tuple[int, ...],
    train: bool = False,
) -> tuple[int, dict[str, Any] | None, bool, Any, str | None]:
    """(index, row, overflowed, segments, why no segments) of one item.

    An item's fifth field, when present, overrides ``train`` for that item, so
    training and evaluation canvases can share one worker pool.
    """
    index, record, source_weight = item[:3]
    mode = item[3] if len(item) > 3 else _ROW
    if len(item) > 4:
        train = bool(item[4])
    if mode == _RESERVE:
        segments, reason = _reserve_segments(
            tokenizer,
            record,
            cfg,
            spec,
            vocab_size=vocab_size,
            slot_plan=slot_plan,
            thought_open_ids=thought_open_ids,
            thought_close_ids=thought_close_ids,
        )
        return index, None, False, segments, reason
    try:
        row = _canvas_row(
            tokenizer,
            record,
            cfg,
            source_weight,
            spec=spec,
            vocab_size=vocab_size,
            slot_plan=slot_plan,
            thought_open_ids=thought_open_ids,
            thought_close_ids=thought_close_ids,
            train=train,
        )
    except SchemaError as error:
        if not _is_canvas_overflow(error):
            raise
        return index, None, True, None, None
    if mode != _ROW_SEGMENTS:
        return index, row, False, None, None
    segments, reason = _prepared_segments(
        tokenizer,
        record,
        row,
        cfg,
        spec,
        vocab_size=vocab_size,
        slot_plan=slot_plan,
        thought_open_ids=thought_open_ids,
        thought_close_ids=thought_close_ids,
    )
    return index, row, False, segments, reason


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
        self, index: int | str | DecisionDraw | Sequence[int]
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
        return _project_draw(self._rows[index.index], index)

    def remove_columns(self, columns: Sequence[str]) -> "DecisionDataset":
        if tuple(columns) != ("length",) or "length" not in self.column_names:
            raise ValueError("DecisionDataset only exposes the derived length column")
        return DecisionDataset(self._rows, self.manifest, expose_lengths=False)


def _project_draw(maximum: Mapping[str, Any], draw: DecisionDraw) -> dict[str, Any]:
    """Project a maximum-slot row to the slot count sampled for ``draw``."""
    recipe = maximum.get("slot_sampling")
    plan = maximum.get("slot_plan")
    if not isinstance(recipe, Mapping) or not isinstance(plan, SlotPlan):
        raise ValueError("decision draw requires a sampled maximum slot row")
    count = sample_slot_count(
        seed=int(recipe["seed"]),
        epoch=draw.epoch,
        global_draw_ordinal=draw.global_draw_ordinal,
        max_slots=int(recipe["max_slots"]),
    )
    canvas = project_max_canvas(
        maximum["canvas"],
        plan,
        count,
        pad_token_id=int(recipe["pad_token_id"]),
        padding_pinned=bool(recipe["padding_pinned"]),
    )
    row = dict(maximum)
    row["canvas"] = canvas
    row["slot_plan"] = project_slot_plan(plan, count)
    row["decision_example"] = decision_example_from_canvas(
        canvas, source_weight=maximum["decision_example"].source_weight
    )
    row["decision_slot_count"] = count
    row["decision_draw"] = draw
    return row


class DecisionPlanDataset(Dataset):
    """Training draws of a :class:`DecisionBatchPlan`, one epoch at a time.

    A :class:`DecisionDraw` addresses position ``index`` of epoch ``epoch``; the
    trainer's plan sampler yields those. Integer indices address epoch 0, so
    index-based consumers (core multipack, plain iteration) see a fixed epoch.
    ``set_epoch`` (called by accelerate on the dataset, also after a resume skip)
    moves the shared plan, which the plan sampler reads.
    """

    def __init__(
        self,
        plan: DecisionBatchPlan,
        manifest: Mapping[str, Any],
        *,
        expose_lengths: bool = True,
    ):
        self.batch_plan = plan
        self.manifest = dict(manifest)
        self.column_names = ("length",) if expose_lengths else ()

    @property
    def carries_images(self) -> bool:
        """Whether any training record (prepared or reserve) has images."""
        return bool(self.manifest.get("train_has_images", False))

    def __len__(self) -> int:
        return self.batch_plan.draws_per_epoch

    def set_epoch(self, epoch: int) -> None:
        self.batch_plan.set_epoch(epoch)

    def _draw_row(self, epoch: int, position: int) -> dict[str, Any]:
        if not 0 <= position < len(self):
            raise IndexError(position)
        return self.batch_plan.row(epoch, position)

    def __getitem__(
        self, index: int | str | DecisionDraw | Sequence[int]
    ) -> dict[str, Any] | list[int] | list[dict[str, Any]]:
        if isinstance(index, str):
            if index != "length":
                raise KeyError(index)
            return self.batch_plan.lengths(0)
        if isinstance(index, int):
            return self._draw_row(0, index)
        if isinstance(index, DecisionDraw):
            row = self._draw_row(index.epoch, index.index)
            if isinstance(row.get("slot_sampling"), Mapping):
                return _project_draw(row, index)
            return row
        if isinstance(index, Sequence):
            rows = []
            for item in index:
                if not isinstance(item, int):
                    raise TypeError("packed decision indices must be integers")
                rows.append(self._draw_row(0, item))
            return rows
        raise TypeError(f"unsupported decision dataset index {type(index)!r}")

    def remove_columns(self, columns: Sequence[str]) -> "DecisionPlanDataset":
        if tuple(columns) != ("length",) or "length" not in self.column_names:
            raise ValueError(
                "DecisionPlanDataset only exposes the derived length column"
            )
        return DecisionPlanDataset(self.batch_plan, self.manifest, expose_lengths=False)


def _canvas_rows(
    tokenizer: Any,
    records: Sequence[dict[str, Any]],
    cfg: Any,
    weights: Mapping[str, float],
    spec,
    *,
    train: bool = False,
    aligned: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Build canvases; ``aligned`` keeps a None in place of each overflowing record."""
    prepared = _prepare_canvas_items(
        tokenizer,
        [(record, _ROW) for record in records],
        cfg,
        weights,
        spec,
        train=train,
    )
    return _collect_canvas_rows(prepared, aligned=aligned)


def _collect_canvas_rows(
    prepared: Sequence[tuple[dict[str, Any] | None, bool, Any, str | None]],
    *,
    aligned: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """(rows, overflow count) of prepared ``_ROW`` items; see :func:`_canvas_rows`."""
    rows: list[Any] = []
    canvas_too_long = 0
    for row, overflow, _, _ in prepared:
        if row is None:
            canvas_too_long += int(overflow)
            if aligned:
                rows.append(None)
            continue
        rows.append(row)
    return rows, canvas_too_long


def _warm_segment_caches(tokenizer: Any, cfg: Any) -> None:
    """Fill the per-process tokenizer caches once, before workers fork.

    The segment safety check (parses the tokenizer JSON), the label tables and
    the compiled chat template would otherwise be rebuilt in every worker of
    every source's pool.
    """
    codebook = _value(
        _value(_value(cfg, "diffusion_decision"), "labels"), "codebook", "vendored26"
    )
    try:
        if segment_safety(tokenizer) is None:
            label_tables(tokenizer, codebook)
        tokenizer.apply_chat_template(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:  # noqa: BLE001 - a cold cache is only slower
        LOG.debug("Decision segment caches not warmed", exc_info=True)


def _item_train(item: Sequence[Any], default: bool) -> bool:
    """An item's own ``train`` flag (its third field), else ``default``."""
    return bool(item[2]) if len(item) > 2 else default


def _prepare_canvas_items(
    tokenizer: Any,
    items: Sequence[
        tuple[Mapping[str, Any], str] | tuple[Mapping[str, Any], str, bool]
    ],
    cfg: Any,
    weights: Mapping[str, float],
    spec,
    *,
    train: bool = False,
) -> list[tuple[dict[str, Any] | None, bool, Any, str | None]]:
    """(row, overflowed, segments, why no segments) per (record, mode) item, in order.

    An item may carry a third field overriding ``train`` for that item. Callers
    pass every record of a build in one call: the worker pool (and its fork,
    tokenizer and template state) is then started once, and small sources share
    it instead of running serially in the parent.
    """
    results: list[tuple[dict[str, Any] | None, bool, Any, str | None]] = []
    vocab_size = _value(model_config_overrides(cfg), "vocab_size")
    if vocab_size is None:
        vocab_size = len(tokenizer)
    slot_plan, thought_open_ids, thought_close_ids = _resolve_slot_plan(
        tokenizer, cfg, spec, int(vocab_size)
    )
    workers = _value(cfg, "dataset_num_proc", 1) or 1
    workers = min(int(workers), max(1, len(items) // _CANVAS_CHUNK_SIZE))
    if workers > 1 and any(item[1] != _ROW for item in items):
        _warm_segment_caches(tokenizer, cfg)
    if workers > 1:
        # Canvas building is GIL-bound Python (template resolution, validation,
        # per-canvas tokenizer calls), so threads never scaled; worker
        # processes each receive the tokenizer and config once, through the
        # pool initializer, and build ordered chunks of records.
        LOG.info(
            "Preparing %s decision canvases with %s worker processes",
            len(items),
            workers,
        )
        state = (
            tokenizer,
            cfg,
            spec,
            int(vocab_size),
            slot_plan,
            tuple(thought_open_ids),
            tuple(thought_close_ids),
            train,
        )
        chunks = (
            [
                (
                    index,
                    items[index][0],
                    float(weights.get(items[index][0]["source"], 1.0)),
                    items[index][1],
                    _item_train(items[index], train),
                )
                for index in range(start, min(start + _CANVAS_CHUNK_SIZE, len(items)))
            ]
            for start in range(0, len(items), _CANVAS_CHUNK_SIZE)
        )
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_canvas_mp_context(),
            initializer=_init_canvas_worker,
            initargs=(state,),
        ) as executor:
            index = 0
            for prepared in executor.map(_build_canvas_chunk, chunks):
                for row, overflow, segments, reason in prepared:
                    record = items[index][0]
                    index += 1
                    if row is not None:
                        # Rows come back unpickled; restore the sharing the serial
                        # path has with the parent's slot plan and (large) state
                        # strings.
                        if slot_plan is not None:
                            row["slot_plan"] = slot_plan
                        text = record.get("state")
                        if isinstance(text, str) and row["record"].get("state") == text:
                            row["record"]["state"] = text
                    results.append((row, overflow, segments, reason))
                if index % (_CANVAS_CHUNK_SIZE * 40) < len(prepared):
                    LOG.info("Preparing decision canvases: %s/%s", index, len(items))
        return results
    for index, item in enumerate(items):
        record, mode = item[0], item[1]
        if index % 1000 == 0:
            LOG.info("Preparing decision canvases: %s/%s", index, len(items))
        results.append(
            _build_canvas_worker(
                (
                    index,
                    record,
                    float(weights.get(record["source"], 1.0)),
                    mode,
                    _item_train(item, train),
                ),
                tokenizer=tokenizer,
                cfg=cfg,
                spec=spec,
                vocab_size=int(vocab_size),
                slot_plan=slot_plan,
                thought_open_ids=thought_open_ids,
                thought_close_ids=thought_close_ids,
                train=train,
            )[1:]
        )
    return results


def load_decision_datasets(
    cfg: Any, preprocess: bool = False, *, tokenizer: Any = None
) -> TrainDatasetMeta:
    """Load, normalize, decontaminate, canvasize, and return plugin-owned splits."""
    del preprocess
    spec = require_diffusion_spec(cfg)
    train_entries = [
        entry
        for entry in (_value(cfg, "datasets", []) or [])
        if str(_value(entry, "type", "")).startswith("diffusion_decision.")
    ]
    test_entries = [
        entry
        for entry in (_value(cfg, "test_datasets", []) or [])
        if str(_value(entry, "type", "")).startswith("diffusion_decision.")
    ]
    if not train_entries:
        raise ValueError("diffusion_decision requires at least one decision dataset")
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
        image_paths = (
            None if source_paths is None else _prepared_cache_image_paths(source_paths)
        )
        if source_paths is not None and image_paths is not None:
            cache_identity = prepared_cache_identity(
                cfg,
                tokenizer,
                spec,
                source_paths,
                image_paths=image_paths,
                verify=prepared_cache_verify(cfg),
            )
            if cache_identity is None:
                LOG.warning(
                    "Decision prepared-row cache bypassed: could not resolve the tokenizer "
                    "files or revision (set revision_of_model to a commit hash to pin them)"
                )
        else:
            LOG.info("Decision prepared-row cache bypassed: nonlocal or opaque source")
    elif isinstance(prepared_path, (str, Path)) and str(prepared_path):
        LOG.info("Decision prepared-row cache bypassed: unsupported latent mode")
    layer = None
    if cache_identity is not None:
        cache_key, cache_payload = cache_identity
        cached = load_prepared_cache(
            Path(prepared_path),
            cache_key,
            cache_payload,
            verify=prepared_cache_verify(cfg),
        )
        if cached is not None:
            cached_train, cached_eval, cached_manifest = cached
            LOG.info("Decision prepared-row cache hit")
            layer = (cached_train, cached_eval, cached_manifest)
        else:
            LOG.info("Decision prepared-row cache miss")
    if layer is None:
        layer = _build_canvas_layer(cfg, tokenizer, spec, train_entries, test_entries)
        if cache_identity is not None:
            store_prepared_cache(
                Path(prepared_path), cache_key, cache_payload, cfg, *layer
            )
    return _assemble_datasets(cfg, tokenizer, spec, *layer)


def _build_canvas_layer(
    cfg: Any,
    tokenizer: Any,
    spec: Any,
    train_entries: Sequence[Any],
    test_entries: Sequence[Any],
) -> tuple[list[Any], list[dict[str, Any]], dict[str, Any]]:
    """Normalize, decontaminate, group and canvasize; no batch setting is read here.

    Returns (train, eval rows, base manifest). ``train`` is a list of
    :class:`SourcePool` (the epoch-0 capped subset prepared, the rest of each capped
    source kept as compact reserve records), or, for premixed data, the prepared
    draws in input order. Budget filtering and mixture draws happen in
    :func:`_assemble_datasets`, so batch settings never invalidate this layer.
    """
    decision = _value(cfg, "diffusion_decision")
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
    mixture = _value(decision, "mixture")
    premixed = bool(_value(mixture, "premixed", False))
    retained: Sequence[Mapping[str, Any]]
    drops: dict[str, int]
    if bool(_value(decision, "decontaminate", False)):
        LOG.info(
            "Decontaminating %s training records against %s development and %s protected records",
            len(train),
            len(dev),
            len(protected_eval),
        )
        retained, drops = decontaminate(train, [*dev, *protected_eval])
    else:
        LOG.info(
            "Skipping decontamination of %s training records (diffusion_decision.decontaminate "
            "is off); train/dev isolation is still asserted, protected splits are not checked",
            len(train),
        )
        retained, drops = train, {}
    if premixed and len(retained) != len(train):
        raise ValueError("premixed decision rows overlap an evaluation split")
    train = [dict(record) for record in retained]
    max_questions = int(_value(decision, "max_questions_per_canvas", 20))
    fits = _grouping_fits(tokenizer, cfg, spec)
    if premixed:
        _validate_premixed_records(train)
    else:
        LOG.info("Grouping compatible decision records")
        train = group_records(train, max_questions=max_questions, fits=fits)
    dev = group_records(dev, max_questions=max_questions, fits=fits)
    assert_split_isolation({"train": train, "eval": dev})
    weights = _value(_value(decision, "mixture"), "loss_weight", {}) or {}
    train_canvas_too_long = 0
    drop_counts = dict(drops)
    layer_train: list[Any]
    # Every canvas of the build (train sources, then eval) goes through one
    # _prepare_canvas_items call, so one worker pool serves the whole build.
    eval_items = [(record, _ROW, False) for record in dev]
    if premixed:
        built = _prepare_canvas_items(
            tokenizer,
            [
                *((dict(record), _ROW, True) for record in train),
                *eval_items,
            ],
            cfg,
            weights,
            spec,
            train=True,
        )
        layer_train, train_canvas_too_long = _collect_canvas_rows(built[: len(train)])
        eval_built = built[len(train) :]
        del built
    else:
        seed = int(_value(cfg, "seed", 0))
        cap = _value(mixture, "max_examples_per_source")
        pools = prepare_source_pools(train, (), max_examples_per_source=cap, seed=seed)
        assert pools.candidates is not None
        layer_train = []
        reserve = 0
        segment_counts: dict[str, Any] = {
            "prepared": 0,
            "reserve": 0,
            "fallback": {},
        }
        plans: list[tuple[str, Any, Any, list[int], list[int]]] = []
        items: list[tuple[Mapping[str, Any], str, bool]] = []
        for source, candidates in pools.candidates.items():
            source_cap = cap.get(source) if isinstance(cap, Mapping) else cap
            window = cap_window(len(candidates), source_cap, seed, source)
            prepared = set(window)
            rest = [index for index in range(len(candidates)) if index not in prepared]
            plans.append((source, source_cap, candidates, list(window), rest))
            items.extend(
                (dict(candidates[index]), _ROW_SEGMENTS, True) for index in window
            )
            items.extend((candidates[index], _RESERVE, True) for index in rest)
        items.extend(eval_items)
        built = _prepare_canvas_items(tokenizer, items, cfg, weights, spec, train=True)
        del items
        offset = 0
        for source, source_cap, candidates, window, rest in plans:
            prepared = set(window)
            source_built = built[offset : offset + len(candidates)]
            offset += len(candidates)
            entries: list[Any] = [None] * len(candidates)
            records: list[Any] = [None] * len(candidates)
            segments: list[Any] = [None] * len(candidates)
            for index, (row, overflow, segment, reason) in zip(
                [*window, *rest], source_built, strict=True
            ):
                if index in prepared:
                    train_canvas_too_long += int(overflow)
                    entries[index] = row
                    records[index] = candidates[index]
                    if row is None:
                        continue
                else:
                    entries[index] = encode_reserve(candidates[index])
                    reserve += 1
                segments[index] = segment
                kind = "prepared" if index in prepared else "reserve"
                if segment is not None:
                    segment_counts[kind] += 1
                else:
                    key = f"{kind}: {reason}"
                    segment_counts["fallback"][key] = (
                        segment_counts["fallback"].get(key, 0) + 1
                    )
            layer_train.append(
                SourcePool(
                    source=source,
                    cap=None if source_cap is None else int(source_cap),
                    entries=entries,
                    records=records,
                    segments=segments,
                )
            )
        eval_built = built[offset:]
        del built
        drop_counts.update(
            {
                key: value
                for key, value in pools.counts.items()
                if key not in drop_counts
            }
        )
        drop_counts["reserve"] = reserve
        LOG.info("Pre-tokenized decision canvas segments: %s", segment_counts)
    eval_rows, eval_canvas_too_long = _collect_canvas_rows(eval_built)
    del eval_built
    manifest: dict[str, Any] = {
        "protected_eval_rows": len(protected_eval),
        "train_sources": sorted({row["source"] for row in train}),
        "eval_sources": sorted({row["source"] for row in dev}),
        "dropped": drop_counts,
        "canvas_too_long": {
            "train": train_canvas_too_long,
            "eval": eval_canvas_too_long,
        },
        "premixed": premixed,
        "train_has_images": any(record.get("images") for record in train),
    }
    if not premixed:
        manifest["segments"] = segment_counts
    if any(
        _value(record.get("source_metadata", {}), "official_split") is not None
        for record in [*train, *dev, *protected_eval]
        if record["source"] == "procedural"
    ):
        manifest["procedural_grouping_limit"] = (
            "official partitions and state decontamination only; no parent-family key"
        )
    return layer_train, eval_rows, manifest


# Top-level config keys a plan row build reads (``_canvas_row`` reads the decision
# section; ``_canvas_settings`` the rest). A pickled builder keeps only these, so a
# DataLoader worker does not receive the whole training config.
_ROW_BUILDER_CFG_KEYS = (
    "seed",
    "diffusion_decision",
    "diffusion_lm",
    "overrides_of_model_config",
    "model_config",
)


def _row_builder_cfg(cfg: Any) -> dict[str, Any]:
    """The config keys of ``_ROW_BUILDER_CFG_KEYS`` that ``cfg`` sets."""
    if isinstance(cfg, Mapping):
        return {key: cfg[key] for key in _ROW_BUILDER_CFG_KEYS if key in cfg}
    missing = object()
    values = {key: getattr(cfg, key, missing) for key in _ROW_BUILDER_CFG_KEYS}
    return {key: value for key, value in values.items() if value is not missing}


class _PlanRowBuilder:
    """Lazily build the training row of one draw of a record (see DecisionBatchPlan).

    A module-level callable rather than a closure so a DecisionPlanDataset pickles
    into DataLoader workers (``spawn`` / ``forkserver`` start methods, the default
    from Python 3.14). Pickling keeps the config keys a build reads, the tokenizer,
    spec, resolved slot plan and canvas settings (computed at the first build).
    """

    def __init__(self, cfg: Any, tokenizer: Any, spec: Any) -> None:
        decision = _value(cfg, "diffusion_decision")
        self._weights = _value(_value(decision, "mixture"), "loss_weight", {}) or {}
        vocab_size = _value(model_config_overrides(cfg), "vocab_size")
        if vocab_size is None:
            vocab_size = len(tokenizer)
        self._cfg = cfg
        self._tokenizer = tokenizer
        self._spec = spec
        self._vocab_size = vocab_size
        (
            self._slot_plan,
            self._thought_open_ids,
            self._thought_close_ids,
        ) = _resolve_slot_plan(tokenizer, cfg, spec, int(vocab_size))
        self._settings: Mapping[str, Any] | None = None
        self._settings_ready = False  # computed at the first build

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_cfg"] = _row_builder_cfg(self._cfg)
        return state

    def _canvas_settings(self) -> Mapping[str, Any] | None:
        if not self._settings_ready:
            self._settings_ready = True
            try:
                self._settings = _canvas_settings(
                    self._tokenizer,
                    self._cfg,
                    self._spec,
                    int(self._vocab_size),
                    self._slot_plan,
                )
            except (TypeError, ValueError):
                self._settings = None  # _canvas_row raises it where it did before
        return self._settings

    def __call__(
        self,
        source: str,
        record: Mapping[str, Any],
        draw: tuple[int, int],
        segments: Any = None,
        cached_prompt: Sequence[int] | None = None,
    ) -> dict[str, Any] | None:
        settings = self._canvas_settings()
        try:
            return _canvas_row(
                self._tokenizer,
                dict(record),
                self._cfg,
                float(self._weights.get(source, 1.0)),
                spec=self._spec,
                vocab_size=int(self._vocab_size),
                slot_plan=self._slot_plan,
                thought_open_ids=self._thought_open_ids,
                thought_close_ids=self._thought_close_ids,
                train=True,
                draw=draw,
                segments=segments,
                cached_prompt=cached_prompt,
                settings=settings,
            )
        except SchemaError as error:
            # Records outside the prepared subset were never canvasized; one that
            # cannot be (not only overflow) is replaced rather than ending the run.
            LOG.debug("Decision draw of %s could not be built: %s", source, error)
            return None


def _plan_row_builder(cfg: Any, tokenizer: Any, spec: Any) -> _PlanRowBuilder:
    """Lazily build the training row of one draw of a record (see DecisionBatchPlan)."""
    return _PlanRowBuilder(cfg, tokenizer, spec)


def _assemble_datasets(
    cfg: Any,
    tokenizer: Any,
    spec: Any,
    layer_train: Sequence[Any],
    layer_eval: Sequence[dict[str, Any]],
    base_manifest: Mapping[str, Any],
) -> TrainDatasetMeta:
    """Apply the batch settings to a canvas layer: budgets, mixture plan, datasets."""
    decision = _value(cfg, "diffusion_decision")
    mixture = _value(decision, "mixture")
    premixed = bool(base_manifest.get("premixed", False))
    stratified = bool(_value(mixture, "per_batch_stratified", True))
    micro_batch_size = int(_value(cfg, "micro_batch_size", 1))
    seed = int(_value(cfg, "seed", 0))
    manifest = dict(base_manifest)
    plan = None
    if premixed:
        train_rows, train_budget_drops = _filter_budget_rows(
            list(layer_train), cfg, spec
        )
        if manifest["canvas_too_long"]["train"] or len(train_rows) != len(layer_train):
            raise ValueError(
                "premixed decision rows must all pass canvas and budget validation"
            )
        manifest.update(
            {
                "train_rows": len(train_rows),
                "mixture_probabilities": {},
                "stratified_epoch_batches": (),
            }
        )
    else:
        plan = DecisionBatchPlan(
            layer_train,
            seed=seed,
            micro_batch_size=micro_batch_size,
            stratified=stratified,
            weights=_value(mixture, "weights"),
            temperature=_value(mixture, "temperature"),
            limits=_budget_limits(cfg),
            layout=spec.layout,
            build_row=_plan_row_builder(cfg, tokenizer, spec),
        )
        train_budget_drops = dict(plan.budget_drops)
        epoch_sources = plan.epoch_sources(0)
        manifest.update(
            {
                "train_rows": plan.draws_per_epoch,
                "mixture_probabilities": dict(plan.probabilities),
                "stratified_epoch_batches": tuple(
                    tuple(epoch_sources[start : start + micro_batch_size])
                    for start in range(0, len(epoch_sources), micro_batch_size)
                ),
                "batch_plan": {
                    "per_epoch": True,
                    "draws_per_epoch": plan.draws_per_epoch,
                    "batches_per_epoch": plan.batches_per_epoch,
                    "reserve_records": plan.reserve_records,
                },
            }
        )
    eval_rows, eval_budget_drops = _filter_budget_rows(
        list(layer_eval), cfg, spec, is_eval=True
    )
    manifest.update(
        {
            "eval_rows": len(eval_rows),
            "budget_drops": {"train": train_budget_drops, "eval": eval_budget_drops},
            "per_batch_stratified": stratified,
            "stratified_micro_batch_size": micro_batch_size,
            "mixture_seed": seed,
        }
    )
    LOG.info(
        "Prepared %s training draws per epoch and %s evaluation rows; budget drops: %s",
        manifest["train_rows"],
        len(eval_rows),
        manifest["budget_drops"],
    )
    return TrainDatasetMeta(
        train_dataset=DecisionPlanDataset(plan, manifest)
        if plan is not None
        else DecisionDataset(train_rows, manifest),
        eval_dataset=DecisionDataset(eval_rows, manifest) if eval_rows else None,
        total_num_steps=None,
    )
