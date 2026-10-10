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
from axolotl.core.trainers.diffusion_lm.sampling import resolve_native_packing_budget
from axolotl.loaders import load_tokenizer
from axolotl.model_support import (
    DiffusionLayout,
    DiffusionNoise,
)
from axolotl.utils.dict import DictDefault

from ._util import (
    _value,
    active_control_ids,
    canonical_json,
    model_config_overrides,
    parse_token_ids,
    read_jsonl,
    require_diffusion_spec,
    required_token_id,
    resolve_mask_token_id,
)
from .adapters import normalize_record
from .batch_plan import (
    BudgetLimits,
    DecisionBatchPlan,
    SourcePool,
    budget_reason,
    encode_reserve,
)
from .grouping import group_records
from .hygiene import assert_split_isolation, decontaminate, family_key
from .loss import decision_example_from_canvas
from .mixture import cap_window, prepare_source_pools
from .permutation import permute_record, shuffle_labels
from .prepared_cache import (
    identity as prepared_cache_identity,
    load as load_prepared_cache,
    prepared_cache_verify,
    store as store_prepared_cache,
)
from .preprocessing import build_decision_canvas, canvas_head, template_fits
from .prompts import resolve_image_refs
from .slot_sampling import (
    DecisionDraw,
    project_max_canvas,
    project_slot_plan,
    sample_slot_count,
)
from .slots import SlotInit, SlotPlan
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


def _thought_delimiters(model_config: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        parse_token_ids(_value(model_config, "thought_open_ids"), "thought_open_ids"),
        parse_token_ids(_value(model_config, "thought_close_ids"), "thought_close_ids"),
    )


def _resolve_slot_plan(
    tokenizer: Any, cfg: Any, spec: Any, vocab_size: int
) -> tuple[SlotPlan | None, tuple[int, ...], tuple[int, ...]]:
    decision = _value(cfg, "diffusion_decision")
    latent = _value(decision, "latent")
    if latent is None:
        return None, (), ()
    mode = _value(latent, "mode", "none")
    if mode == "none":
        return None, (), ()
    model_config = model_config_overrides(cfg)
    thought_open_ids, thought_close_ids = _thought_delimiters(model_config)
    token_ids = parse_token_ids(
        _value(latent, "token_ids", ()) or (), "latent.token_ids"
    )
    mask_token_id = resolve_mask_token_id(_value(cfg, "diffusion_lm"), model_config)
    if mode in {"pinned", "learned", "prompt"}:
        control_ids = active_control_ids(tokenizer, model_config)
        control_ids.update(thought_open_ids, thought_close_ids)
        if mask_token_id is not None:
            control_ids.add(mask_token_id)
        collisions = sorted(set(token_ids) & control_ids)
        if collisions:
            raise ValueError(
                "fixed decision slot token_ids cannot use active control IDs: "
                f"{collisions}"
            )
    plan = SlotInit(
        mode=mode,
        token_ids=token_ids,
        num_slots=_value(latent, "num_slots", 0),
        vocab_size=vocab_size,
        pad_id=required_token_id(
            _value(tokenizer, "pad_token_id"), "tokenizer.pad_token_id"
        ),
        spec=spec,
        mask_token_id=mask_token_id,
    ).build(seed=int(_value(cfg, "seed", 0)))
    if (
        plan.placement == "thought"
        and _value(spec, "layout") is DiffusionLayout.ENCODER_CANVAS
        and (not thought_open_ids or not thought_close_ids)
    ):
        raise ValueError(
            "encoder-canvas thought slots require `model_config: {thought_open_ids: "
            "[...], thought_close_ids: [...]}` in the training config"
        )
    return plan, thought_open_ids, thought_close_ids


def _read_overflow_width(
    decision: Any, spec, slot_plan: SlotPlan | None, scaffold_ids: Sequence[int]
) -> int | None:
    """The read-time fallback width for a canvas whose template overflows, or None."""
    value = _value(decision, "read_overflow_canvas_width")
    if value is None:
        return None
    if _value(spec, "layout") is not DiffusionLayout.FULL_SEQUENCE:
        raise ValueError(
            "diffusion_decision.read_overflow_canvas_width requires a full_sequence "
            "layout; encoder-canvas batches need one shared canvas width"
        )
    if slot_plan is not None or scaffold_ids:
        raise ValueError(
            "diffusion_decision.read_overflow_canvas_width cannot be combined with "
            "latent slots"
        )
    width = int(value)
    if spec.max_canvas is not None:
        width = min(width, int(spec.max_canvas))
    return width


def _canvas_width(cfg: Any, spec) -> int:
    configured_width = _value(_value(cfg, "diffusion_lm"), "canvas_width")
    width = 128 if configured_width is None else int(configured_width)
    if spec.max_canvas is not None:
        width = min(width, spec.max_canvas)
    return width


def _scaffold_ids(latent: Any, slot_plan: SlotPlan | None) -> tuple[int, ...]:
    if slot_plan is not None:
        return ()
    return tuple(int(token) for token in (_value(latent, "token_ids", ()) or ()))


def _grouping_fits(tokenizer: Any, cfg: Any, spec) -> Any:
    """The width check ``group_records`` packs against, or None for count-only grouping.

    Only an explicit answer format (lines or indexed) packs by width, so configs on
    the default auto format group exactly as before.
    """
    decision = _value(cfg, "diffusion_decision")
    answer_format = str(_value(decision, "answer_format", "auto") or "auto")
    if answer_format == "auto":
        return None
    vocab_size = _value(model_config_overrides(cfg), "vocab_size")
    if vocab_size is None:
        vocab_size = len(tokenizer)
    slot_plan, thought_open_ids, thought_close_ids = _resolve_slot_plan(
        tokenizer, cfg, spec, int(vocab_size)
    )
    head, pad_slots = canvas_head(
        _scaffold_ids(_value(decision, "latent"), slot_plan),
        slot_plan,
        thought_open_ids,
        thought_close_ids,
    )
    width = _canvas_width(cfg, spec)
    codebook = _value(_value(decision, "labels"), "codebook", "vendored26")

    def fits(questions: Mapping[str, Mapping[str, Any]]) -> bool:
        return template_fits(
            tokenizer,
            questions,
            head=head,
            pad_slots=len(pad_slots),
            width=width,
            answer_format=answer_format,
            codebook=codebook,
        )

    return fits


def _canvas_row(
    tokenizer,
    record: dict[str, Any],
    cfg: Any,
    source_weight: float,
    *,
    spec,
    vocab_size: int | None = None,
    slot_plan: SlotPlan | None = None,
    thought_open_ids: Sequence[int] = (),
    thought_close_ids: Sequence[int] = (),
    train: bool = False,
    draw: tuple[int, int] | None = None,
):
    diffusion = _value(cfg, "diffusion_lm")
    decision = _value(cfg, "diffusion_decision")
    model_config = model_config_overrides(cfg)
    latent = _value(decision, "latent")
    width = _canvas_width(cfg, spec)
    mask_token_id = resolve_mask_token_id(diffusion, model_config)
    scaffold_ids = _scaffold_ids(latent, slot_plan)
    record = permute_record(
        record,
        seed=int(_value(cfg, "seed", 0)),
        codebook=_value(_value(decision, "labels"), "codebook", "vendored26"),
        draw=draw,
    )
    shuffle_prob = float(
        _value(_value(decision, "labels"), "label_shuffle_prob", 0.0) or 0.0
    )
    if train and shuffle_prob > 0.0:
        record = shuffle_labels(
            record,
            seed=int(_value(cfg, "seed", 0)),
            probability=shuffle_prob,
            codebook=_value(_value(decision, "labels"), "codebook", "vendored26"),
            draw=draw,
        )
    if vocab_size is None:
        vocab_size = _value(model_config, "vocab_size")
    if vocab_size is None:
        vocab_size = len(tokenizer)
    turn_close_id = required_token_id(
        _value(model_config, "turn_close_token_id", _value(tokenizer, "eos_token_id")),
        "turn_close_token_id",
    )

    def build(canvas_width: int):
        return build_decision_canvas(
            tokenizer,
            record,
            scaffold_ids=scaffold_ids,
            turn_close_id=turn_close_id,
            pad_id=required_token_id(
                _value(tokenizer, "pad_token_id"), "tokenizer.pad_token_id"
            ),
            vocab_size=int(vocab_size),
            width=canvas_width,
            seed=int(_value(cfg, "seed", 0)),
            steps=int(_value(_value(diffusion, "unroll"), "k_max", 1)),
            noise_kind="absorbing"
            if spec.noise is DiffusionNoise.ABSORBING
            else "uniform",
            mask_token_id=mask_token_id,
            slot_plan=slot_plan,
            thought_open_ids=thought_open_ids,
            thought_close_ids=thought_close_ids,
            codebook=_value(_value(decision, "labels"), "codebook", "vendored26"),
            prevalidated_record=True,
            max_image_size=int(_value(decision, "max_image_size", 1400)),
            answer_format=str(_value(decision, "answer_format", "auto") or "auto"),
        )

    try:
        canvas = build(width)
    except SchemaError as error:
        overflow_width = _read_overflow_width(decision, spec, slot_plan, scaffold_ids)
        if (
            train
            or overflow_width is None
            or overflow_width <= width
            or not _is_canvas_overflow(error)
        ):
            raise
        canvas = build(overflow_width)
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
    if slot_plan is not None:
        row["slot_plan"] = slot_plan
        if _value(latent, "sample_num_slots", False):
            row["slot_sampling"] = {
                "seed": int(_value(cfg, "seed", 0)),
                "max_slots": len(slot_plan.ids),
                "pad_token_id": required_token_id(
                    _value(tokenizer, "pad_token_id"), "tokenizer.pad_token_id"
                ),
                "padding_pinned": True,
            }
    return row


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


def _build_canvas_chunk(
    items: Sequence[tuple[int, dict[str, Any], float]],
) -> list[tuple[dict[str, Any] | None, bool]]:
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
    item: tuple[int, dict[str, Any], float],
    *,
    tokenizer: Any,
    cfg: Any,
    spec: Any,
    vocab_size: int,
    slot_plan: SlotPlan | None,
    thought_open_ids: tuple[int, ...],
    thought_close_ids: tuple[int, ...],
    train: bool = False,
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
            slot_plan=slot_plan,
            thought_open_ids=thought_open_ids,
            thought_close_ids=thought_close_ids,
            train=train,
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
            raise ValueError("DecisionPlanDataset only exposes the derived length column")
        return DecisionPlanDataset(self.batch_plan, self.manifest, expose_lengths=False)


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
    *,
    train: bool = False,
    aligned: bool = False,
) -> tuple[list[dict[str, Any]], int]:
    """Build canvases; ``aligned`` keeps a None in place of each overflowing record."""
    rows: list[dict[str, Any]] = []
    canvas_too_long = 0
    vocab_size = _value(model_config_overrides(cfg), "vocab_size")
    if vocab_size is None:
        vocab_size = len(tokenizer)
    slot_plan, thought_open_ids, thought_close_ids = _resolve_slot_plan(
        tokenizer, cfg, spec, int(vocab_size)
    )
    workers = _value(cfg, "dataset_num_proc", 1) or 1
    workers = min(int(workers), max(1, len(records) // _CANVAS_CHUNK_SIZE))
    if workers > 1:
        # Canvas building is GIL-bound Python (template resolution, validation,
        # per-canvas tokenizer calls), so threads never scaled; worker
        # processes each receive the tokenizer and config once, through the
        # pool initializer, and build ordered chunks of records.
        LOG.info(
            "Preparing %s decision canvases with %s worker processes",
            len(records),
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
                    records[index],
                    float(weights.get(records[index]["source"], 1.0)),
                )
                for index in range(start, min(start + _CANVAS_CHUNK_SIZE, len(records)))
            ]
            for start in range(0, len(records), _CANVAS_CHUNK_SIZE)
        )
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=_canvas_mp_context(),
            initializer=_init_canvas_worker,
            initargs=(state,),
        ) as executor:
            index = 0
            for prepared in executor.map(_build_canvas_chunk, chunks):
                for row, overflow in prepared:
                    record = records[index]
                    index += 1
                    if row is None:
                        canvas_too_long += int(overflow)
                        if aligned:
                            rows.append(None)
                        continue
                    # Rows come back unpickled; restore the sharing the serial path
                    # has with the parent's slot plan and (large) state strings.
                    if slot_plan is not None:
                        row["slot_plan"] = slot_plan
                    text = record.get("state")
                    if isinstance(text, str) and row["record"].get("state") == text:
                        row["record"]["state"] = text
                    rows.append(row)
                if index % (_CANVAS_CHUNK_SIZE * 40) < len(prepared):
                    LOG.info("Preparing decision canvases: %s/%s", index, len(records))
        return rows, canvas_too_long
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
                    slot_plan=slot_plan,
                    thought_open_ids=thought_open_ids,
                    thought_close_ids=thought_close_ids,
                    train=train,
                )
            )
        except SchemaError as error:
            if not _is_canvas_overflow(error):
                raise
            canvas_too_long += 1
            if aligned:
                rows.append(None)
    return rows, canvas_too_long


def _budget_limits(cfg: Any, *, is_eval: bool = False) -> BudgetLimits:
    """Logical (sequence_len) and packed physical token limits for one split."""
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
    return BudgetLimits(
        logical=None if logical_limit is None else int(logical_limit),
        physical=None if resolved is None else int(resolved.payload_capacity),
    )


def _filter_budget_rows(
    rows: Sequence[dict[str, Any]], cfg: Any, spec, *, is_eval: bool = False
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    limits = _budget_limits(cfg, is_eval=is_eval)
    kept: list[dict[str, Any]] = []
    drops = {"logical": 0, "physical": 0}
    for row in rows:
        reason = budget_reason(row, limits, spec.layout)
        if reason is None:
            kept.append(row)
        else:
            drops[reason] += 1
    return kept, drops


def load_decision_datasets(
    cfg: Any, preprocess: bool = False, *, tokenizer: Any = None
) -> TrainDatasetMeta:
    """Load, normalize, decontaminate, canvasize, and return plugin-owned splits."""
    del preprocess
    decision = _value(cfg, "diffusion_decision")
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
            store_prepared_cache(Path(prepared_path), cache_key, cache_payload, cfg, *layer)
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
    if premixed:
        layer_train, train_canvas_too_long = _canvas_rows(
            tokenizer,
            [dict(record) for record in train],
            cfg,
            weights,
            spec,
            train=True,
        )
    else:
        seed = int(_value(cfg, "seed", 0))
        cap = _value(mixture, "max_examples_per_source")
        pools = prepare_source_pools(
            train, (), max_examples_per_source=cap, seed=seed
        )
        assert pools.candidates is not None
        layer_train = []
        reserve = 0
        for source, candidates in pools.candidates.items():
            source_cap = cap.get(source) if isinstance(cap, Mapping) else cap
            window = cap_window(len(candidates), source_cap, seed, source)
            built, dropped = _canvas_rows(
                tokenizer,
                [dict(candidates[index]) for index in window],
                cfg,
                weights,
                spec,
                train=True,
                aligned=True,
            )
            train_canvas_too_long += dropped
            entries: list[Any] = [None] * len(candidates)
            records: list[Any] = [None] * len(candidates)
            for index, row in zip(window, built, strict=True):
                entries[index] = row
                records[index] = candidates[index]
            prepared = set(window)
            for index, candidate in enumerate(candidates):
                if index not in prepared:
                    entries[index] = encode_reserve(candidate)
                    reserve += 1
            layer_train.append(
                SourcePool(
                    source=source,
                    cap=None if source_cap is None else int(source_cap),
                    entries=entries,
                    records=records,
                )
            )
        drop_counts.update(
            {
                key: value
                for key, value in pools.counts.items()
                if key not in drop_counts
            }
        )
        drop_counts["reserve"] = reserve
    eval_rows, eval_canvas_too_long = _canvas_rows(tokenizer, dev, cfg, weights, spec)
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
    }
    if any(
        _value(record.get("source_metadata", {}), "official_split") is not None
        for record in [*train, *dev, *protected_eval]
        if record["source"] == "procedural"
    ):
        manifest["procedural_grouping_limit"] = (
            "official partitions and state decontamination only; no parent-family key"
        )
    return layer_train, eval_rows, manifest


def _plan_row_builder(cfg: Any, tokenizer: Any, spec: Any):
    """Lazily build the training row of one draw of a record (see DecisionBatchPlan)."""
    decision = _value(cfg, "diffusion_decision")
    weights = _value(_value(decision, "mixture"), "loss_weight", {}) or {}
    vocab_size = _value(model_config_overrides(cfg), "vocab_size")
    if vocab_size is None:
        vocab_size = len(tokenizer)
    slot_plan, thought_open_ids, thought_close_ids = _resolve_slot_plan(
        tokenizer, cfg, spec, int(vocab_size)
    )

    def build(
        source: str, record: Mapping[str, Any], draw: tuple[int, int]
    ) -> dict[str, Any] | None:
        try:
            return _canvas_row(
                tokenizer,
                dict(record),
                cfg,
                float(weights.get(source, 1.0)),
                spec=spec,
                vocab_size=int(vocab_size),
                slot_plan=slot_plan,
                thought_open_ids=thought_open_ids,
                thought_close_ids=thought_close_ids,
                train=True,
                draw=draw,
            )
        except SchemaError as error:
            # Records outside the prepared subset were never canvasized; one that
            # cannot be (not only overflow) is replaced rather than ending the run.
            LOG.debug("Decision draw of %s could not be built: %s", source, error)
            return None

    return build


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
