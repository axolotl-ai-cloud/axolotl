"""Model-agnostic batches for typed decision diffusion training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import torch

from axolotl.integrations.decision.loss import (
    DecisionLabelExample,
    decision_example_from_canvas,
)
from axolotl.integrations.decision.records import DecisionCanvas
from axolotl.model_support import (
    DiffusionLayout,
    DiffusionSpec,
    get_model_support_for_cfg,
    resolve_model_support,
)
from axolotl.utils.collators.multimodal import collate_image_inputs


def _value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


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


class DecisionTrainingCollator:
    """Emit layout-specific token tensors and a shared decision-loss contract."""

    def __init__(
        self,
        tokenizer: Any | DiffusionSpec | None = None,
        *,
        spec: DiffusionSpec | None = None,
        pad_token_id: int | None = None,
        padding: bool | str = True,
        max_length: int | None = None,
        pad_to_multiple_of: int | None = None,
        label_pad_token_id: int = -100,
        return_tensors: str = "pt",
        physical_payload_capacity: int | None = None,
    ) -> None:
        if isinstance(tokenizer, DiffusionSpec):
            if spec is not None:
                raise TypeError("spec was provided twice")
            spec = tokenizer
            tokenizer = None
        if not isinstance(spec, DiffusionSpec):
            raise TypeError("DecisionTrainingCollator requires a DiffusionSpec")
        if padding not in {True, "longest"}:
            raise ValueError("DecisionTrainingCollator requires longest padding")
        if label_pad_token_id != -100:
            raise ValueError("DecisionTrainingCollator uses native decision masks")
        if return_tensors != "pt":
            raise ValueError("DecisionTrainingCollator requires return_tensors='pt'")
        self.spec = spec
        self.pad_token_id = int(
            pad_token_id
            if pad_token_id is not None
            else _value(tokenizer, "pad_token_id", 0) or 0
        )
        self.max_length = max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        self.physical_payload_capacity = physical_payload_capacity

    def __call__(self, rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        rows = self._flatten_packed_rows(rows)
        if not rows:
            raise ValueError("empty decision batch")
        canvases = tuple(self._canvas(row) for row in rows)
        examples = tuple(
            _decision_example(canvas, row)
            for canvas, row in zip(canvases, rows, strict=True)
        )
        metadata = _decision_metadata(canvases, examples)
        metadata["decision_sources"] = tuple(self._source(row) for row in rows)
        if self.spec.layout is not DiffusionLayout.FULL_SEQUENCE:
            raise ValueError("decision supports full-sequence Nemotron only")
        return self._full_sequence(canvases, examples, metadata)

    @staticmethod
    def _flatten_packed_rows(
        rows: Sequence[Mapping[str, Any] | Sequence[Mapping[str, Any]]],
    ) -> list[Mapping[str, Any]]:
        """Accept the nested logical examples emitted by MultipackBatchSampler."""

        flattened: list[Mapping[str, Any]] = []
        for row in rows:
            if isinstance(row, Mapping):
                flattened.append(row)
            elif isinstance(row, Sequence) and not isinstance(row, (str, bytes)):
                if not all(isinstance(item, Mapping) for item in row):
                    raise TypeError("packed decision batches must contain row mappings")
                flattened.extend(row)
            else:
                raise TypeError("decision batches must contain row mappings")
        return flattened

    @staticmethod
    def _canvas(row: Mapping[str, Any]) -> DecisionCanvas:
        canvas = row.get("canvas")
        if not isinstance(canvas, DecisionCanvas):
            raise TypeError(
                "decision dataset rows require a DecisionCanvas under 'canvas'"
            )
        return canvas

    @staticmethod
    def _source(row: Mapping[str, Any]) -> str:
        source = row.get("source")
        if not isinstance(source, str) or not source:
            raise ValueError("decision dataset rows require a nonempty source")
        return source

    def _full_sequence(
        self,
        canvases: Sequence[DecisionCanvas],
        examples: tuple[DecisionLabelExample, ...],
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        ids: list[int] = []
        documents: list[int] = []
        positions: list[int] = []
        loss: list[bool] = []
        corruptible: list[bool] = []
        pinned: list[bool] = []
        label_positions: list[list[int]] = []
        offset = 0
        for document, canvas in enumerate(canvases):
            prompt = tuple(int(token) for token in canvas.prompt_ids)
            output = tuple(int(token) for token in canvas.canvas_ids)
            ids.extend((*prompt, *output))
            documents.extend([document] * (len(prompt) + len(output)))
            positions.extend(range(len(prompt) + len(output)))
            label_set = set(int(position) for position in canvas.label_positions)
            output_loss = [index in label_set for index in range(len(output))]
            loss.extend([False] * len(prompt) + output_loss)
            corruptible.extend([False] * len(prompt) + output_loss)
            pinned.extend([True] * len(prompt) + list(canvas.pinned_mask))
            label_positions.append(
                [offset + len(prompt) + position for position in canvas.label_positions]
            )
            offset += len(prompt) + len(output)
        if (
            self.physical_payload_capacity is not None
            and len(ids) > self.physical_payload_capacity
        ):
            raise ValueError(
                "decision full-sequence microbatch exceeds native physical payload capacity"
            )
        input_ids = torch.tensor(ids, dtype=torch.long)[None]
        metadata.update(
            {
                "input_ids": input_ids,
                "document_ids": torch.tensor(documents, dtype=torch.long)[None],
                "semantic_validity": torch.ones_like(input_ids, dtype=torch.bool),
                "position_ids": torch.tensor(positions, dtype=torch.long)[None],
                "canvas_loss_mask": torch.tensor(loss, dtype=torch.bool)[None],
                "canvas_corruptible_mask": torch.tensor(corruptible, dtype=torch.bool)[
                    None
                ],
                "canvas_input_pinned_mask": torch.tensor(pinned, dtype=torch.bool)[
                    None
                ],
                "canvas_update_mask": torch.tensor(corruptible, dtype=torch.bool)[None]
                & ~torch.tensor(pinned, dtype=torch.bool)[None],
                "decision_label_rows": _label_rows(label_positions),
                "decision_label_positions": _label_positions(label_positions),
            }
        )
        metadata["decision_examples"] = examples
        media = collate_image_inputs([canvas.model_inputs for canvas in canvases])
        if media:
            metadata["model_inputs"] = media
        return metadata


def _decision_metadata(
    canvases: Sequence[DecisionCanvas], examples: tuple[DecisionLabelExample, ...]
) -> dict[str, Any]:
    width = max(len(canvas.question_ids) for canvas in canvases)
    batch = len(canvases)
    question_mask = torch.zeros((batch, width), dtype=torch.bool)
    logical_rows = torch.full((batch, width), -1, dtype=torch.long)
    positions = torch.full((batch, width), -1, dtype=torch.long)
    for row, canvas in enumerate(canvases):
        count = len(canvas.question_ids)
        question_mask[row, :count] = True
        logical_rows[row, :count] = row
        positions[row, :count] = torch.tensor(canvas.label_positions, dtype=torch.long)
    return {
        "decision_examples": examples,
        "decision_question_mask": question_mask,
        "decision_supervision_mask": question_mask.clone(),
        "decision_logical_rows": logical_rows,
        "decision_label_rows": logical_rows.clone(),
        "decision_label_positions": positions,
    }


def _label_rows(label_positions: Sequence[Sequence[int]]) -> torch.Tensor:
    result = torch.full(
        (len(label_positions), max(map(len, label_positions))), -1, dtype=torch.long
    )
    for row, values in enumerate(label_positions):
        result[row, : len(values)] = 0
    return result


def _label_positions(label_positions: Sequence[Sequence[int]]) -> torch.Tensor:
    result = torch.full(
        (len(label_positions), max(map(len, label_positions))), -1, dtype=torch.long
    )
    for row, values in enumerate(label_positions):
        result[row, : len(values)] = torch.tensor(values, dtype=torch.long)
    return result


def decision_collator_for_config(cfg: Any, is_eval: bool = False):
    from axolotl.utils.dict import DictDefault

    if isinstance(cfg, Mapping):
        cfg = DictDefault(cfg)
    profile = resolve_model_support(get_model_support_for_cfg(cfg))
    spec = None if profile is None else profile.diffusion
    if not isinstance(spec, DiffusionSpec):
        raise ValueError("decision requires a resolved DiffusionSpec")
    tokenizer = _value(cfg, "tokenizer")
    pad_token_id = int(_value(tokenizer, "pad_token_id", 0) or 0)
    from axolotl.integrations.diffusion.lm.sampling import (
        resolve_native_packing_budget,
    )

    packed = bool(_value(cfg, "sample_packing", False)) and (
        not is_eval or _value(cfg, "eval_sample_packing", False) is not False
    )
    fixed_logical_batch = bool(_value(cfg, "batch_flattening", False))
    budget = resolve_native_packing_budget(
        cfg,
        packed=packed or fixed_logical_batch,
        batch_size=(
            _value(cfg, "micro_batch_size")
            if packed
            else _value(cfg, "eval_batch_size")
            if is_eval
            else _value(cfg, "micro_batch_size")
        ),
    )
    return DecisionTrainingCollator, {
        "spec": spec,
        "pad_token_id": pad_token_id,
        "physical_payload_capacity": None
        if budget is None
        else budget.payload_capacity,
    }
