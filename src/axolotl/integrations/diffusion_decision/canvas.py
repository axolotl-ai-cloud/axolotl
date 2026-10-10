"""Light decision-canvas building shared by dataset loading and serving.

Everything a server needs to turn normalized, grouped records into the exact
canvases training builds (slot plan, canvas row, overflow fallback, token-budget
filter, per-question allowed token IDs) without importing axolotl's training
stack: no ``axolotl.core``, peft, accelerate, ``datasets``, pandas or
``transformers.trainer`` at import time. ``datasets`` re-exports these names.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
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
from .permutation import permute_record, shuffle_labels
from .preprocessing import build_decision_canvas, canvas_head, template_fits
from .records import DecisionCanvas
from .slots import SlotInit, SlotPlan
from .template import SchemaError


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
