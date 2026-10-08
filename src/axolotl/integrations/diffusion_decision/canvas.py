"""Light decision-canvas building shared by dataset loading and serving.

Everything a server needs to turn normalized, grouped records into the exact
canvases training builds (slot plan, canvas row, overflow fallback, token-budget
filter, per-question allowed token IDs) without importing axolotl's training
stack: no ``axolotl.core``, peft, accelerate, ``datasets``, pandas or
``transformers.trainer`` at import time. ``datasets`` re-exports these names.
"""

from __future__ import annotations

import gc
import logging
import multiprocessing
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from typing import Any

from axolotl.model_support import DiffusionLayout, DiffusionNoise
from axolotl.model_support.native_packing import resolve_native_packing_budget

from ._util import (
    _value,
    active_control_ids,
    model_config_overrides,
    parse_token_ids,
    required_token_id,
    resolve_mask_token_id,
)
from .batch_plan import BudgetLimits, budget_reason
from .loss import DecisionLabelExample, decision_example_from_canvas
from .permutation import permute_record, permute_record_with_orders, shuffle_labels
from .preprocessing import (
    build_decision_canvas,
    canvas_head,
    decision_schema,
    template_fits,
    template_fits_many,
)
from .prompts import decision_prompt_ids
from .records import DecisionCanvas
from .segments import (
    SEGMENT_STATS,
    RecordSegments,
    SegmentFallback,
    answer_base_ids,
    assemble_canvas,
    build_segments,
    segment_suffix,
)
from .slots import SlotInit, SlotPlan
from .template import SchemaError, system_text

LOG = logging.getLogger(__name__)


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
    if plan.placement == "thought" and canvas_prefix_ids(cfg):
        raise ValueError(
            "diffusion_decision.canvas_prefix_ids cannot be combined with thought-placed "
            "latent slots; the slot thought block already opens the canvas"
        )
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
    # canvas_prefix_ids heads ``scaffold_ids`` too; only latent slots are refused here.
    if slot_plan is not None or _value(_value(decision, "latent"), "token_ids"):
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


def canvas_prefix_ids(cfg: Any) -> tuple[int, ...]:
    """Clean tokens pinned at the head of every train and read canvas."""
    return _canvas_prefix_ids(_value(cfg, "diffusion_decision"))


def _canvas_prefix_ids(decision: Any) -> tuple[int, ...]:
    return parse_token_ids(
        _value(decision, "canvas_prefix_ids", ()) or (),
        "diffusion_decision.canvas_prefix_ids",
    )


def _scaffold_ids(decision: Any, slot_plan: SlotPlan | None) -> tuple[int, ...]:
    """The canvas head before the answer template: ``canvas_prefix_ids``, then the
    latent token IDs when they are not placed through a slot plan."""
    prefix = _canvas_prefix_ids(decision)
    if slot_plan is not None:
        return prefix
    latent = _value(decision, "latent")
    return prefix + tuple(
        int(token) for token in (_value(latent, "token_ids", ()) or ())
    )


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
        _scaffold_ids(decision, slot_plan),
        slot_plan,
        thought_open_ids,
        thought_close_ids,
    )
    width = _canvas_width(cfg, spec)
    codebook = _value(_value(decision, "labels"), "codebook", "vendored26")
    return _GroupingFits(
        tokenizer,
        head=head,
        pad_slots=len(pad_slots),
        width=width,
        answer_format=answer_format,
        codebook=codebook,
        workers=int(_value(cfg, "dataset_num_proc", 1) or 1),
    )


# Smallest batch measured in worker processes: forking a preprocessing process
# that holds every loaded record costs about a second per pool, more than a
# smaller batch takes serially.
_FITS_MIN_BATCH = 65536
# Fewest canvases per worker task; a batch is split into about four tasks per
# worker.
_FITS_CHUNK_SIZE = 1024
# (fits, batch) of the running ``_GroupingFits.many``: forked workers inherit it,
# so the records are never pickled to them; tasks carry only index ranges.
_FITS_WORKER_STATE: tuple[Any, Sequence[Any]] | None = None


def _init_fits_worker() -> None:
    # Keep the cyclic collector off the heap inherited from the parent: a full
    # collection there walks (and copies the pages of) every loaded record.
    gc.freeze()


def _fits_range(bounds: tuple[int, int]) -> list[bool | None]:
    if _FITS_WORKER_STATE is None:
        raise RuntimeError("grouping fits worker has no batch")
    fits, batch = _FITS_WORKER_STATE
    start, stop = bounds
    return fits.many_serial(batch[start:stop])


class _GroupingFits:
    """``template_fits`` bound to one config; ``many`` measures a batch of canvases.

    ``group_records`` calls ``many`` on every record that is alone in its group
    (most records) and calls the instance itself for incremental merges.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        head: Sequence[int],
        pad_slots: int,
        width: int,
        answer_format: str,
        codebook: str,
        workers: int = 1,
    ) -> None:
        self.tokenizer = tokenizer
        self.head = head
        self.pad_slots = pad_slots
        self.width = width
        self.answer_format = answer_format
        self.codebook = codebook
        self.workers = workers

    def __call__(self, questions: Mapping[str, Mapping[str, Any]]) -> bool:
        return template_fits(
            self.tokenizer,
            questions,
            head=self.head,
            pad_slots=self.pad_slots,
            width=self.width,
            answer_format=self.answer_format,
            codebook=self.codebook,
        )

    def many_serial(
        self, batch: Sequence[Mapping[str, Mapping[str, Any]]]
    ) -> list[bool | None]:
        return template_fits_many(
            self.tokenizer,
            batch,
            head=self.head,
            pad_slots=self.pad_slots,
            width=self.width,
            answer_format=self.answer_format,
            codebook=self.codebook,
        )

    def many(
        self, batch: Sequence[Mapping[str, Mapping[str, Any]]]
    ) -> list[bool | None]:
        """``many_serial`` of the batch, split over ``workers`` forked processes.

        Each worker measures contiguous index ranges of the batch it inherited
        and results are joined in range order, so the output is the serial one.
        Without fork (or if the pool fails) the batch is measured serially.
        """
        workers = min(self.workers, len(batch) // _FITS_CHUNK_SIZE)
        if (
            workers <= 1
            or len(batch) < _FITS_MIN_BATCH
            or "fork" not in multiprocessing.get_all_start_methods()
        ):
            return self.many_serial(batch)
        global _FITS_WORKER_STATE
        size = max(_FITS_CHUNK_SIZE, -(-len(batch) // (4 * workers)))
        ranges = [
            (start, min(start + size, len(batch)))
            for start in range(0, len(batch), size)
        ]
        LOG.info(
            "Measuring %s grouping canvases with %s worker processes",
            len(batch),
            workers,
        )
        _FITS_WORKER_STATE = (self, batch)
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=multiprocessing.get_context("fork"),
                initializer=_init_fits_worker,
            ) as executor:
                results: list[bool | None] = []
                for part in executor.map(_fits_range, ranges):
                    results.extend(part)
        except Exception:  # noqa: BLE001  (a lost worker must not fail the build)
            LOG.warning(
                "Grouping fits worker pool failed; measuring serially", exc_info=True
            )
            return self.many_serial(batch)
        finally:
            _FITS_WORKER_STATE = None
        return results


def _canvas_settings(
    tokenizer: Any,
    cfg: Any,
    spec,
    vocab_size: int | None,
    slot_plan: SlotPlan | None,
) -> dict[str, Any]:
    """The build_decision_canvas keywords (and permutation settings) of one config."""
    diffusion = _value(cfg, "diffusion_lm")
    decision = _value(cfg, "diffusion_decision")
    model_config = model_config_overrides(cfg)
    labels = _value(decision, "labels")
    if vocab_size is None:
        vocab_size = _value(model_config, "vocab_size")
    if vocab_size is None:
        vocab_size = len(tokenizer)
    return {
        "seed": int(_value(cfg, "seed", 0)),
        "codebook": _value(labels, "codebook", "vendored26"),
        "shuffle_prob": float(_value(labels, "label_shuffle_prob", 0.0) or 0.0),
        "answer_format": str(_value(decision, "answer_format", "auto") or "auto"),
        "width": _canvas_width(cfg, spec),
        "scaffold_ids": _scaffold_ids(decision, slot_plan),
        "turn_close_id": required_token_id(
            _value(
                model_config, "turn_close_token_id", _value(tokenizer, "eos_token_id")
            ),
            "turn_close_token_id",
        ),
        "pad_id": required_token_id(
            _value(tokenizer, "pad_token_id"), "tokenizer.pad_token_id"
        ),
        "vocab_size": int(vocab_size),
        "steps": int(_value(_value(diffusion, "unroll"), "k_max", 1)),
        "noise_kind": "absorbing"
        if spec.noise is DiffusionNoise.ABSORBING
        else "uniform",
        "mask_token_id": resolve_mask_token_id(diffusion, model_config),
    }


def _canvas_row(
    tokenizer,
    record: Mapping[str, Any],
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
    segments: RecordSegments | None = None,
    cached_prompt: Sequence[int] | None = None,
    settings: Mapping[str, Any] | None = None,
):
    """One canvas row; a training draw with ``segments`` is assembled from cached
    token spans (``cached_prompt``: a prepared record's cached prompt, which holds
    the chat suffix), falling back to full tokenization where they cannot.
    ``settings``: ``_canvas_settings`` of these arguments, when the caller keeps them."""
    decision = _value(cfg, "diffusion_decision")
    if settings is None:
        settings = _canvas_settings(tokenizer, cfg, spec, vocab_size, slot_plan)
    if segments is not None and train:
        try:
            record_out, canvas = assemble_canvas(
                tokenizer,
                record,
                segments,
                None
                if segments.suffix_stored
                else segment_suffix(segments, cached_prompt),
                draw=draw,
                slot_plan=slot_plan,
                thought_open_ids=thought_open_ids,
                thought_close_ids=thought_close_ids,
                **settings,
            )
        except SegmentFallback as error:
            SEGMENT_STATS[f"fallback:{error}"] += 1
        except Exception as error:  # noqa: BLE001 - the full path decides
            SEGMENT_STATS[f"fallback:{type(error).__name__}"] += 1
        else:
            SEGMENT_STATS["assembled"] += 1
            return _row(
                canvas, record_out, source_weight, decision, slot_plan, settings
            )
    seed, codebook = settings["seed"], settings["codebook"]
    record = permute_record(record, seed=seed, codebook=codebook, draw=draw)
    if train and settings["shuffle_prob"] > 0.0:
        record = shuffle_labels(
            record,
            seed=seed,
            probability=settings["shuffle_prob"],
            codebook=codebook,
            draw=draw,
        )
    width = settings["width"]
    scaffold_ids = settings["scaffold_ids"]

    def build(canvas_width: int):
        return build_decision_canvas(
            tokenizer,
            record,
            scaffold_ids=scaffold_ids,
            turn_close_id=settings["turn_close_id"],
            pad_id=settings["pad_id"],
            vocab_size=settings["vocab_size"],
            width=canvas_width,
            seed=seed,
            steps=settings["steps"],
            noise_kind=settings["noise_kind"],
            mask_token_id=settings["mask_token_id"],
            slot_plan=slot_plan,
            thought_open_ids=thought_open_ids,
            thought_close_ids=thought_close_ids,
            codebook=codebook,
            prevalidated_record=True,
            max_image_size=int(_value(decision, "max_image_size", 1400)),
            answer_format=settings["answer_format"],
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
    return _row(canvas, record, source_weight, decision, slot_plan, settings)


def _row(
    canvas: DecisionCanvas,
    record: dict[str, Any],
    source_weight: float,
    decision: Any,
    slot_plan: SlotPlan | None,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
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
        if _value(_value(decision, "latent"), "sample_num_slots", False):
            row["slot_sampling"] = {
                "seed": settings["seed"],
                "max_slots": len(slot_plan.ids),
                "pad_token_id": settings["pad_id"],
                "padding_pinned": True,
            }
    return row


def _segments_supported(slot_plan: SlotPlan | None) -> bool:
    # A prompt-placed plan prefixes the prompt; segments split the plain prompt.
    return slot_plan is None or slot_plan.placement != "prompt"


def _prepared_segments(
    tokenizer: Any,
    record: Mapping[str, Any],
    row: Mapping[str, Any],
    cfg: Any,
    spec,
    *,
    vocab_size: int | None,
    slot_plan: SlotPlan | None,
    thought_open_ids: Sequence[int],
    thought_close_ids: Sequence[int],
) -> tuple[RecordSegments | None, str | None]:
    """Segments of a prepared training record, checked against its built canvas.

    Returns (segments without the chat suffix, None) or (None, why not).
    """
    if not _segments_supported(slot_plan):
        return None, "prompt-placed slots"
    canvas = row["canvas"]
    try:
        settings = _canvas_settings(tokenizer, cfg, spec, vocab_size, slot_plan)
        _, orders = permute_record_with_orders(
            record, seed=settings["seed"], codebook=settings["codebook"]
        )
        segments = build_segments(
            tokenizer,
            record,
            codebook=settings["codebook"],
            answer_format=settings["answer_format"],
            permuted=row["record"],
            orders=orders,
            prompt_ids=canvas.prompt_ids,
        )
        _, assembled = assemble_canvas(
            tokenizer,
            record,
            segments,
            None,
            draw=None,
            slot_plan=slot_plan,
            thought_open_ids=thought_open_ids,
            thought_close_ids=thought_close_ids,
            **settings,
        )
    except SegmentFallback as error:
        return None, str(error)
    except Exception as error:  # noqa: BLE001 - such a record keeps full tokenization
        return None, type(error).__name__
    if assembled != canvas:
        return None, "assembled first draw differs from its canvas"
    return segments.without_suffix(), None


def _reserve_segments(
    tokenizer: Any,
    record: Mapping[str, Any],
    cfg: Any,
    spec,
    *,
    vocab_size: int | None,
    slot_plan: SlotPlan | None,
    thought_open_ids: Sequence[int],
    thought_close_ids: Sequence[int],
) -> tuple[RecordSegments | None, str | None]:
    """Segments (with the chat suffix) of a training record not built at preprocess.

    Its first draw's prompt is tokenized in full to place the chat suffix, and its
    answer template base is checked against a full encode.
    """
    if not _segments_supported(slot_plan):
        return None, "prompt-placed slots"
    try:
        settings = _canvas_settings(tokenizer, cfg, spec, vocab_size, slot_plan)
        seed, codebook = settings["seed"], settings["codebook"]
        permuted, orders = permute_record_with_orders(
            record, seed=seed, codebook=codebook
        )
        if settings["shuffle_prob"] > 0.0:
            permuted = shuffle_labels(
                permuted,
                seed=seed,
                probability=settings["shuffle_prob"],
                codebook=codebook,
            )
        schema = decision_schema(
            permuted, codebook=codebook, answer_format=settings["answer_format"]
        )
        prompt_ids = decision_prompt_ids(
            tokenizer, system_text(schema), permuted["state"]
        )
        segments = build_segments(
            tokenizer,
            record,
            codebook=codebook,
            answer_format=settings["answer_format"],
            permuted=permuted,
            orders=orders,
            prompt_ids=prompt_ids,
        )
        _, assembled = assemble_canvas(
            tokenizer,
            record,
            segments,
            None,
            draw=None,
            slot_plan=slot_plan,
            thought_open_ids=thought_open_ids,
            thought_close_ids=thought_close_ids,
            **settings,
        )
        head, _ = canvas_head(
            settings["scaffold_ids"], slot_plan, thought_open_ids, thought_close_ids
        )
        base = answer_base_ids(tokenizer, schema, head)
    except SegmentFallback as error:
        return None, str(error)
    except Exception as error:  # noqa: BLE001 - such a record keeps full tokenization
        return None, type(error).__name__
    actual = list(assembled.canvas_ids[: assembled.template_length])
    for position, label_ids in zip(
        assembled.label_positions, assembled.allowed_ids, strict=True
    ):
        actual[position] = label_ids[0]
    if tuple(assembled.prompt_ids) != tuple(prompt_ids) or actual != base:
        return None, "assembled first draw differs from full tokenization"
    return segments, None


def _is_canvas_overflow(error: SchemaError) -> bool:
    message = str(error)
    return (
        message.startswith("answer template is ") and " canvas holds " in message
    ) or (
        message.startswith("question ")
        and " alone needs " in message
        and message.endswith(" canvas rows")
    )


def _dict_default(cfg: Mapping[str, Any]) -> Any:
    # addict, imported only on the packed-budget path.
    from axolotl.utils.dict import DictDefault

    return DictDefault(cfg)


def _budget_limits(cfg: Any, *, is_eval: bool = False) -> BudgetLimits:
    """Logical (sequence_len) and packed physical token limits for one split."""
    logical_limit = _value(cfg, "sequence_len")
    sample_packed = bool(_value(cfg, "sample_packing", False)) and (
        not is_eval or _value(cfg, "eval_sample_packing") is not False
    )
    packed = bool(_value(cfg, "batch_flattening", False)) or sample_packed
    resolved = (
        resolve_native_packing_budget(
            _dict_default(cfg) if isinstance(cfg, Mapping) else cfg,
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


def _decision_example(
    canvas: DecisionCanvas, row: Mapping[str, Any]
) -> DecisionLabelExample:
    source = row.get("decision_example")
    if source is None:
        source = decision_example_from_canvas(
            canvas, source_weight=float(row.get("source_weight", 1.0))
        )
    if not isinstance(source, DecisionLabelExample):
        raise TypeError("decision_example must be a DecisionLabelExample")
    return replace(
        source,
        questions=tuple(
            replace(question, position=index)
            for index, question in enumerate(source.questions)
        ),
    )
