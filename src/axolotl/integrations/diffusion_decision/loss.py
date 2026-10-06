"""Typed label objectives for decision canvases."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence, TypeAlias

import torch
import torch.nn.functional as F

from .records import DecisionCanvas

LabelSoftmax = Literal["restricted", "full", "both"]
FullCEWeighting = Literal["ce", "dft"]


@dataclass(frozen=True)
class HardLabel:
    """A single correct index among a question's allowed token IDs."""

    gold_index: int
    smoothing: float | None = None


@dataclass(frozen=True)
class DistributionLabel:
    """A dense or sparse probability target for a question's allowed token IDs.

    ``candidate_indices`` makes ``probabilities`` sparse.  ``other_probability``
    is the target mass for every vocabulary token outside those candidates and is
    only meaningful with the full-vocabulary objective.
    """

    probabilities: tuple[float, ...]
    candidate_indices: tuple[int, ...] | None = None
    other_probability: float = 0.0
    candidate_ids: tuple[str, ...] | None = None


@dataclass(frozen=True)
class SetLabel:
    """Indices of every correct alternative among the allowed token IDs."""

    allowed_indices: tuple[int, ...]


DecisionLabelTarget: TypeAlias = HardLabel | DistributionLabel | SetLabel


@dataclass(frozen=True)
class DecisionLabelQuestion:
    """One supervised canvas position and its typed label target."""

    position: int
    allowed_token_ids: tuple[int, ...]
    target: DecisionLabelTarget
    weight: float = 1.0


@dataclass(frozen=True)
class DecisionLabelExample:
    """Questions from one canvas, weighted as one training example."""

    questions: tuple[DecisionLabelQuestion, ...]
    source_weight: float = 1.0


@dataclass(frozen=True)
class DecisionLossResult:
    """Source-weighted question/example averages for logging and optimization."""

    loss: torch.Tensor
    restricted_loss: torch.Tensor
    full_vocab_loss: torch.Tensor
    brier_loss: torch.Tensor
    effective_full_vocab_loss: torch.Tensor | None = None
    full_vocab_dft_hard_weight_sum: torch.Tensor | None = None
    full_vocab_dft_hard_count: torch.Tensor | None = None
    per_example_loss: tuple[torch.Tensor, ...] | None = None
    per_example_restricted_loss: tuple[torch.Tensor, ...] | None = None
    per_example_full_vocab_loss: tuple[torch.Tensor, ...] | None = None
    per_example_effective_full_vocab_loss: tuple[torch.Tensor, ...] | None = None
    per_example_full_vocab_dft_hard_weight_sum: tuple[torch.Tensor, ...] | None = None
    per_example_full_vocab_dft_hard_count: tuple[torch.Tensor, ...] | None = None
    per_example_brier_loss: tuple[torch.Tensor, ...] | None = None


def _optional_smoothing(value: object) -> float | None:
    if value is None:
        return None
    if (
        isinstance(value, bool)
        or not isinstance(value, (float, int))
        or not math.isfinite(value)
        or not 0.0 <= value < 1.0
    ):
        raise ValueError("hard target smoothing must be finite and in [0, 1)")
    return float(value)


def label_target_from_mapping(target: Mapping[str, object]) -> DecisionLabelTarget:
    """Convert a normalized-record target into the typed loss contract."""
    kind = target.get("kind")
    if kind == "hard":
        return HardLabel(
            _index(target.get("gold_idx"), "gold_idx"),
            _optional_smoothing(target.get("smoothing")),
        )
    if kind == "dist":
        values = target.get("probs")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("dist target requires a nonempty probs sequence")
        probabilities = tuple(_probability(value) for value in values)
        indices = target.get("candidate_indices")
        if indices is None:
            candidate_indices = None
        else:
            if not isinstance(indices, (list, tuple)) or len(indices) != len(values):
                raise ValueError("dist candidate_indices must align with probs")
            candidate_indices = tuple(
                _index(value, "candidate_indices") for value in indices
            )
        other_probability = _probability(target.get("other_probability", 0.0))
        candidate_ids_value = target.get("candidate_ids")
        if candidate_ids_value is None:
            candidate_ids = None
        elif (
            not isinstance(candidate_ids_value, (list, tuple))
            or len(candidate_ids_value) != len(probabilities)
            or any(
                not isinstance(value, str) or not value for value in candidate_ids_value
            )
        ):
            raise ValueError("dist candidate_ids must align with probs")
        else:
            candidate_ids = tuple(candidate_ids_value)
        _validate_distribution_values(
            probabilities, candidate_indices, other_probability
        )
        return DistributionLabel(
            probabilities, candidate_indices, other_probability, candidate_ids
        )
    if kind == "set":
        values = target.get("allowed_set")
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("set target requires a nonempty allowed_set sequence")
        indices = tuple(_index(value, "allowed_set") for value in values)
        if len(set(indices)) != len(indices):
            raise ValueError("set target indices must be unique")
        return SetLabel(indices)
    raise ValueError(f"unknown decision label kind {kind!r}")


def decision_example_from_canvas(
    canvas: DecisionCanvas, *, source_weight: float = 1.0
) -> DecisionLabelExample:
    """Adapt a validated decision canvas for the label-loss API."""
    questions = tuple(
        DecisionLabelQuestion(
            position=int(position),
            allowed_token_ids=tuple(int(token) for token in allowed_ids),
            target=(
                target
                if isinstance(target, (HardLabel, DistributionLabel, SetLabel))
                else label_target_from_mapping(target)
            ),
            weight=_question_weight(target),
        )
        for position, allowed_ids, target in zip(
            canvas.label_positions, canvas.allowed_ids, canvas.targets, strict=True
        )
    )
    return DecisionLabelExample(questions=questions, source_weight=source_weight)


def decision_label_loss(
    logits: torch.Tensor,
    examples: Sequence[DecisionLabelExample],
    supervision_mask: torch.Tensor,
    *,
    label_softmax: LabelSoftmax = "both",
    full_ce_weighting: FullCEWeighting = "ce",
    brier_weight: float = 0.1,
    hard_label_smoothing: float = 0.0,
) -> DecisionLossResult:
    """Compute typed label losses using only supervised label positions.

    ``both`` applies restricted and full-vocabulary CE to every label kind.
    Set labels use a detached highest-scoring member of the correct set for
    full-vocabulary CE while retaining their set-mass objective.
    """
    _validate_inputs(
        logits,
        examples,
        supervision_mask,
        label_softmax,
        full_ce_weighting,
        brier_weight,
        hard_label_smoothing,
    )
    restricted_examples = []
    full_vocab_examples = []
    effective_full_vocab_examples = []
    full_vocab_dft_hard_weight_sum_examples = []
    full_vocab_dft_hard_count_examples = []
    brier_examples = []
    for batch_index, example in enumerate(examples):
        _validate_source_weight(example.source_weight)
        restricted_questions = []
        full_vocab_questions = []
        effective_full_vocab_questions = []
        full_vocab_dft_hard_weight_sums = []
        full_vocab_dft_hard_counts = []
        brier_questions = []
        for question in example.questions:
            query_logits = logits[batch_index, question.position].float()
            allowed_ids = _allowed_ids(
                question.allowed_token_ids, query_logits.shape[0]
            )
            allowed_ids = allowed_ids.to(query_logits.device)
            restricted_logprobs = F.log_softmax(
                query_logits.index_select(0, allowed_ids), dim=0
            )
            restricted_probs = restricted_logprobs.exp()
            if (
                isinstance(question.target, DistributionLabel)
                and question.target.other_probability > 0
                and label_softmax == "full"
            ):
                restricted = query_logits.new_zeros(())
                brier = query_logits.new_zeros(())
                full_target = allowed_ids[0]
                full_distribution = None
            else:
                restricted, brier, full_target, full_distribution = (
                    _restricted_question_loss(
                        restricted_logprobs,
                        restricted_probs,
                        allowed_ids,
                        question.target,
                        hard_label_smoothing=hard_label_smoothing,
                    )
                )
            zero = query_logits.new_zeros(())
            if label_softmax == "restricted":
                full_vocab = zero
                effective_full_vocab = zero
                dft_hard_weight_sum = zero
                dft_hard_count = zero
            else:
                full_logprobs = F.log_softmax(query_logits, dim=0)
                full_vocab = _full_vocab_question_loss(
                    full_logprobs,
                    allowed_ids,
                    question.target,
                    full_target,
                    full_distribution,
                )
                dft_hard = full_ce_weighting == "dft" and isinstance(
                    question.target, HardLabel
                )
                dft_weight = (-full_vocab.detach()).exp() if dft_hard else zero
                effective_full_vocab = (
                    full_vocab * dft_weight if dft_hard else full_vocab
                )
                dft_hard_weight_sum = dft_weight
                dft_hard_count = zero.new_ones(()) if dft_hard else zero
                if label_softmax == "full":
                    restricted = zero
            restricted_questions.append(restricted)
            full_vocab_questions.append(full_vocab)
            effective_full_vocab_questions.append(effective_full_vocab)
            full_vocab_dft_hard_weight_sums.append(dft_hard_weight_sum)
            full_vocab_dft_hard_counts.append(dft_hard_count)
            brier_questions.append(brier)
        weight = example.source_weight
        qw = [question.weight for question in example.questions]
        restricted_examples.append(_weighted_mean(restricted_questions, qw) * weight)
        full_vocab_examples.append(_weighted_mean(full_vocab_questions, qw) * weight)
        effective_full_vocab_examples.append(
            _weighted_mean(effective_full_vocab_questions, qw) * weight
        )
        full_vocab_dft_hard_weight_sum_examples.append(
            torch.stack(full_vocab_dft_hard_weight_sums).sum()
        )
        full_vocab_dft_hard_count_examples.append(
            torch.stack(full_vocab_dft_hard_counts).sum()
        )
        brier_examples.append(_weighted_mean(brier_questions, qw) * weight)
    restricted_loss = torch.stack(restricted_examples).mean()
    full_vocab_loss = torch.stack(full_vocab_examples).mean()
    effective_full_vocab_loss = torch.stack(effective_full_vocab_examples).mean()
    full_vocab_dft_hard_weight_sum = torch.stack(
        full_vocab_dft_hard_weight_sum_examples
    ).sum()
    full_vocab_dft_hard_count = torch.stack(full_vocab_dft_hard_count_examples).sum()
    brier_loss = torch.stack(brier_examples).mean()
    metric_restricted = (
        restricted_examples
        if label_softmax != "full"
        else [restricted_loss.new_zeros(()) for _ in examples]
    )
    return DecisionLossResult(
        loss=restricted_loss + effective_full_vocab_loss + brier_weight * brier_loss,
        restricted_loss=restricted_loss,
        full_vocab_loss=full_vocab_loss,
        brier_loss=brier_loss,
        effective_full_vocab_loss=effective_full_vocab_loss,
        full_vocab_dft_hard_weight_sum=full_vocab_dft_hard_weight_sum,
        full_vocab_dft_hard_count=full_vocab_dft_hard_count,
        per_example_loss=tuple(
            restricted.detach()
            + effective_full.detach()
            + brier_weight * brier.detach()
            for restricted, effective_full, brier in zip(
                metric_restricted,
                effective_full_vocab_examples,
                brier_examples,
                strict=True,
            )
        ),
        per_example_restricted_loss=(
            tuple(value.detach() for value in metric_restricted)
        ),
        per_example_full_vocab_loss=tuple(
            value.detach() for value in full_vocab_examples
        ),
        per_example_effective_full_vocab_loss=tuple(
            value.detach() for value in effective_full_vocab_examples
        ),
        per_example_full_vocab_dft_hard_weight_sum=tuple(
            value.detach() for value in full_vocab_dft_hard_weight_sum_examples
        ),
        per_example_full_vocab_dft_hard_count=tuple(
            value.detach() for value in full_vocab_dft_hard_count_examples
        ),
        per_example_brier_loss=tuple(value.detach() for value in brier_examples),
    )


def decision_label_loss_from_hidden(
    hidden: torch.Tensor,
    head: torch.nn.Linear,
    examples: Sequence[DecisionLabelExample],
    supervision_mask: torch.Tensor,
    *,
    linear_token_loss: Callable[
        [torch.Tensor, torch.nn.Linear, torch.Tensor], torch.Tensor
    ],
    label_softmax: LabelSoftmax = "both",
    full_ce_weighting: FullCEWeighting = "ce",
    brier_weight: float = 0.1,
    hard_label_smoothing: float = 0.0,
) -> DecisionLossResult:
    """Compute typed loss without materializing full-vocabulary logits."""
    _validate_hidden_inputs(
        hidden,
        head,
        examples,
        supervision_mask,
        label_softmax,
        full_ce_weighting,
        brier_weight,
        hard_label_smoothing,
    )
    restricted_examples: list[torch.Tensor] = []
    brier_examples: list[torch.Tensor] = []
    full_targets: list[torch.Tensor] = []
    full_soft_corrections: list[torch.Tensor | None] = []
    selected_hidden: list[torch.Tensor] = []
    full_example_indices: list[int] = []
    full_question_weights: list[float] = []
    full_dft_hard: list[bool] = []
    residual_full_losses: list[tuple[int, torch.Tensor, float]] = []
    for batch_index, example in enumerate(examples):
        _validate_source_weight(example.source_weight)
        restricted_questions: list[torch.Tensor] = []
        brier_questions: list[torch.Tensor] = []
        for question in example.questions:
            query_hidden = hidden[batch_index, question.position]
            projection_hidden = query_hidden.to(dtype=head.weight.dtype)
            allowed_ids = _allowed_ids(question.allowed_token_ids, head.weight.shape[0])
            allowed_ids = allowed_ids.to(query_hidden.device)
            selected_bias = (
                head.bias.index_select(0, allowed_ids)
                if head.bias is not None
                else None
            )
            restricted_logits = F.linear(
                projection_hidden,
                head.weight.index_select(0, allowed_ids),
                selected_bias,
            ).float()
            restricted_logprobs = F.log_softmax(restricted_logits, dim=0)
            if (
                isinstance(question.target, DistributionLabel)
                and question.target.other_probability > 0
                and label_softmax == "full"
            ):
                restricted = query_hidden.new_zeros(())
                brier = query_hidden.new_zeros(())
                full_target = allowed_ids[0]
                full_distribution = None
            else:
                restricted, brier, full_target, full_distribution = (
                    _restricted_question_loss(
                        restricted_logprobs,
                        restricted_logprobs.exp(),
                        allowed_ids,
                        question.target,
                        hard_label_smoothing=hard_label_smoothing,
                    )
                )
            restricted_questions.append(restricted)
            brier_questions.append(brier)
            if label_softmax != "restricted":
                if (
                    isinstance(question.target, DistributionLabel)
                    and question.target.other_probability > 0
                ):
                    full_logprobs = F.log_softmax(
                        F.linear(projection_hidden, head.weight, head.bias).float(),
                        dim=0,
                    )
                    residual_full_losses.append(
                        (
                            batch_index,
                            _full_vocab_question_loss(
                                full_logprobs,
                                allowed_ids,
                                question.target,
                                full_target,
                                full_distribution,
                            ),
                            question.weight,
                        )
                    )
                    continue
                selected_hidden.append(projection_hidden)
                full_targets.append(full_target)
                full_question_weights.append(question.weight)
                full_soft_corrections.append(
                    None
                    if full_distribution is None
                    else restricted_logits[full_distribution.argmax()]
                    - (full_distribution * restricted_logits).sum()
                )
                full_example_indices.append(batch_index)
                full_dft_hard.append(
                    full_ce_weighting == "dft"
                    and isinstance(question.target, HardLabel)
                )
        weight = example.source_weight
        qw = [question.weight for question in example.questions]
        restricted_examples.append(_weighted_mean(restricted_questions, qw) * weight)
        brier_examples.append(_weighted_mean(brier_questions, qw) * weight)
    restricted_loss = torch.stack(restricted_examples).mean()
    brier_loss = torch.stack(brier_examples).mean()
    full_vocab_loss = hidden.new_zeros((), dtype=torch.float32)
    effective_full_vocab_loss = hidden.new_zeros((), dtype=torch.float32)
    full_vocab_dft_hard_weight_sum = hidden.new_zeros((), dtype=torch.float32)
    full_vocab_dft_hard_count = hidden.new_zeros((), dtype=torch.float32)
    dft_hard_weight_sum_per_example = [
        hidden.new_zeros((), dtype=torch.float32) for _ in examples
    ]
    dft_hard_count_per_example = [
        hidden.new_zeros((), dtype=torch.float32) for _ in examples
    ]
    if label_softmax != "restricted":
        if selected_hidden:
            token_losses = linear_token_loss(
                torch.stack(selected_hidden), head, torch.stack(full_targets)
            )
            if token_losses.ndim != 1 or token_losses.shape[0] != len(selected_hidden):
                raise ValueError(
                    "linear_token_loss must return one loss per selected query"
                )
            corrections = torch.stack(
                [
                    token_losses[index].new_zeros(())
                    if value is None
                    else value.to(dtype=token_losses.dtype)
                    for index, value in enumerate(full_soft_corrections)
                ]
            )
            token_losses = token_losses + corrections
        else:
            token_losses = hidden.new_empty((0,), dtype=torch.float32)
        per_example: list[list[torch.Tensor]] = [[] for _ in examples]
        per_example_weights: list[list[float]] = [[] for _ in examples]
        for loss, batch_index, question_weight in zip(
            token_losses, full_example_indices, full_question_weights, strict=True
        ):
            per_example[batch_index].append(loss)
            per_example_weights[batch_index].append(question_weight)
        for batch_index, loss, question_weight in residual_full_losses:
            per_example[batch_index].append(loss)
            per_example_weights[batch_index].append(question_weight)
        weighted = [
            _weighted_mean(losses, per_example_weights[index])
            * examples[index].source_weight
            for index, losses in enumerate(per_example)
        ]
        full_vocab_loss = torch.stack(weighted).mean()
        if full_ce_weighting == "dft":
            dft_weights = (-token_losses.detach()).exp()
            dft_hard_mask = torch.tensor(
                full_dft_hard, device=token_losses.device, dtype=torch.bool
            )
            effective_token_losses = torch.where(
                dft_hard_mask, token_losses * dft_weights, token_losses
            )
        else:
            dft_weights = torch.zeros_like(token_losses)
            effective_token_losses = token_losses
        effective_per_example: list[list[torch.Tensor]] = [[] for _ in examples]
        for loss, dft_weight, dft_hard, batch_index in zip(
            effective_token_losses,
            dft_weights,
            full_dft_hard,
            full_example_indices,
            strict=True,
        ):
            effective_per_example[batch_index].append(loss)
            if dft_hard:
                dft_hard_weight_sum_per_example[batch_index] = (
                    dft_hard_weight_sum_per_example[batch_index] + dft_weight
                )
                dft_hard_count_per_example[batch_index] = (
                    dft_hard_count_per_example[batch_index] + 1
                )
        for batch_index, loss, _question_weight_unused in residual_full_losses:
            effective_per_example[batch_index].append(loss)
        effective_weighted = [
            _weighted_mean(losses, per_example_weights[index])
            * examples[index].source_weight
            for index, losses in enumerate(effective_per_example)
        ]
        effective_full_vocab_loss = torch.stack(effective_weighted).mean()
        full_vocab_dft_hard_weight_sum = torch.stack(
            dft_hard_weight_sum_per_example
        ).sum()
        full_vocab_dft_hard_count = torch.stack(dft_hard_count_per_example).sum()
        if label_softmax == "full":
            restricted_loss = restricted_loss.new_zeros(())
    metric_restricted = (
        restricted_examples
        if label_softmax != "full"
        else [restricted_loss.new_zeros(()) for _ in examples]
    )
    metric_full = (
        weighted
        if label_softmax != "restricted"
        else [full_vocab_loss.new_zeros(()) for _ in examples]
    )
    return DecisionLossResult(
        loss=restricted_loss + effective_full_vocab_loss + brier_weight * brier_loss,
        restricted_loss=restricted_loss,
        full_vocab_loss=full_vocab_loss,
        brier_loss=brier_loss,
        effective_full_vocab_loss=effective_full_vocab_loss,
        full_vocab_dft_hard_weight_sum=full_vocab_dft_hard_weight_sum,
        full_vocab_dft_hard_count=full_vocab_dft_hard_count,
        per_example_loss=tuple(
            restricted.detach()
            + effective_full.detach()
            + brier_weight * brier.detach()
            for restricted, effective_full, brier in zip(
                metric_restricted,
                effective_weighted if label_softmax != "restricted" else metric_full,
                brier_examples,
                strict=True,
            )
        ),
        per_example_restricted_loss=tuple(
            value.detach() for value in metric_restricted
        ),
        per_example_full_vocab_loss=tuple(value.detach() for value in metric_full),
        per_example_effective_full_vocab_loss=tuple(
            value.detach()
            for value in (
                effective_weighted if label_softmax != "restricted" else metric_full
            )
        ),
        per_example_full_vocab_dft_hard_weight_sum=tuple(
            value.detach() for value in dft_hard_weight_sum_per_example
        )
        if label_softmax != "restricted"
        else tuple(hidden.new_zeros(()) for _ in examples),
        per_example_full_vocab_dft_hard_count=tuple(
            value.detach() for value in dft_hard_count_per_example
        ),
        per_example_brier_loss=tuple(value.detach() for value in brier_examples),
    )


def _restricted_question_loss(
    restricted_logprobs: torch.Tensor,
    restricted_probs: torch.Tensor,
    allowed_ids: torch.Tensor,
    target: DecisionLabelTarget,
    *,
    hard_label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    if isinstance(target, HardLabel):
        _validate_index(target.gold_index, allowed_ids.shape[0], "gold_index")
        smoothing = (
            hard_label_smoothing
            if target.smoothing is None
            else _optional_smoothing(target.smoothing)
        )
        hard_desired = F.one_hot(
            torch.tensor(target.gold_index, device=restricted_logprobs.device),
            num_classes=allowed_ids.shape[0],
        ).to(dtype=restricted_logprobs.dtype)
        if smoothing:
            desired = (
                hard_desired * (1.0 - smoothing) + smoothing / allowed_ids.shape[0]
            )
            primary = -(desired * restricted_logprobs).sum()
        else:
            desired = hard_desired
            primary = -restricted_logprobs[target.gold_index]
        full_target = allowed_ids[target.gold_index]
        full_distribution = desired if smoothing else None
    elif isinstance(target, DistributionLabel):
        indices, probabilities = _distribution_candidates(target, allowed_ids.shape[0])
        if target.other_probability > 0:
            raise ValueError(
                "a sparse dist target with other_probability requires "
                "label_softmax=full"
            )
        desired = torch.tensor(
            probabilities,
            dtype=restricted_logprobs.dtype,
            device=restricted_logprobs.device,
        )
        if len(indices) != allowed_ids.shape[0]:
            dense_desired = torch.zeros_like(restricted_logprobs)
            dense_desired[torch.tensor(indices, device=allowed_ids.device)] = desired
            desired = dense_desired
        ce_desired = desired
        if hard_label_smoothing and _is_one_hot_distribution(
            target, allowed_ids.shape[0]
        ):
            ce_desired = (
                desired * (1.0 - hard_label_smoothing)
                + hard_label_smoothing / allowed_ids.shape[0]
            )
        positive = ce_desired > 0
        primary = torch.where(
            positive,
            ce_desired * (ce_desired.log() - restricted_logprobs),
            torch.zeros_like(ce_desired),
        ).sum()
        full_target = allowed_ids[desired.argmax()]
        full_distribution = ce_desired
    elif isinstance(target, SetLabel):
        _validate_set(target.allowed_indices, allowed_ids.shape[0])
        indices = torch.tensor(
            target.allowed_indices, dtype=torch.long, device=restricted_logprobs.device
        )
        set_logprobs = restricted_logprobs.index_select(0, indices)
        primary = -torch.logsumexp(set_logprobs, dim=0)
        set_mass = set_logprobs.exp().sum()
        full_target = allowed_ids[indices[set_logprobs.detach().argmax()]]
        brier = (1.0 - set_mass).square()
        full_distribution = None
    else:
        raise TypeError("unsupported decision label target")

    if not isinstance(target, SetLabel):
        brier_target = hard_desired if isinstance(target, HardLabel) else desired
        brier = torch.sum((restricted_probs - brier_target).square())
    return primary, brier, full_target, full_distribution


def _full_vocab_question_loss(
    full_logprobs: torch.Tensor,
    allowed_ids: torch.Tensor,
    target: DecisionLabelTarget,
    full_target: torch.Tensor,
    full_distribution: torch.Tensor | None,
) -> torch.Tensor:
    if isinstance(target, DistributionLabel) and target.other_probability > 0:
        indices, probabilities = _distribution_candidates(target, allowed_ids.shape[0])
        candidate_ids = allowed_ids[torch.tensor(indices, device=allowed_ids.device)]
        desired = torch.tensor(
            probabilities, dtype=full_logprobs.dtype, device=full_logprobs.device
        )
        loss = -(desired * full_logprobs.index_select(0, candidate_ids)).sum()
        other_logprob = full_logprobs.index_fill(
            0, candidate_ids, -torch.inf
        ).logsumexp(0)
        return loss - target.other_probability * other_logprob
    if full_distribution is None:
        if not isinstance(target, DistributionLabel):
            return -full_logprobs[full_target]
        indices, probabilities = _distribution_candidates(target, allowed_ids.shape[0])
        candidate_ids = allowed_ids[torch.tensor(indices, device=allowed_ids.device)]
        desired = torch.tensor(
            probabilities, dtype=full_logprobs.dtype, device=full_logprobs.device
        )
    else:
        candidate_ids = allowed_ids
        desired = full_distribution
    return -(desired * full_logprobs.index_select(0, candidate_ids)).sum()


def _validate_inputs(
    logits: torch.Tensor,
    examples: Sequence[DecisionLabelExample],
    supervision_mask: torch.Tensor,
    label_softmax: LabelSoftmax,
    full_ce_weighting: FullCEWeighting,
    brier_weight: float,
    hard_label_smoothing: float,
) -> None:
    if logits.ndim != 3:
        raise ValueError("logits must be [batch, sequence, vocabulary]")
    if supervision_mask.shape != logits.shape[:2]:
        raise ValueError("supervision_mask must be [batch, sequence]")
    if supervision_mask.dtype is not torch.bool:
        raise TypeError("supervision_mask must be bool")
    if supervision_mask.device != logits.device:
        raise ValueError("supervision_mask and logits must share a device")
    if len(examples) != logits.shape[0] or not examples:
        raise ValueError("examples must contain one nonempty entry per batch item")
    if label_softmax not in {"restricted", "full", "both"}:
        raise ValueError("label_softmax must be restricted, full, or both")
    if full_ce_weighting not in {"ce", "dft"}:
        raise ValueError("full_ce_weighting must be ce or dft")
    if (
        isinstance(brier_weight, bool)
        or not isinstance(brier_weight, (float, int))
        or not math.isfinite(brier_weight)
    ):
        raise ValueError("brier_weight must be finite")
    if brier_weight < 0:
        raise ValueError("brier_weight must be nonnegative")
    if (
        isinstance(hard_label_smoothing, bool)
        or not isinstance(hard_label_smoothing, (float, int))
        or not math.isfinite(hard_label_smoothing)
        or not 0.0 <= hard_label_smoothing < 1.0
    ):
        raise ValueError("hard_label_smoothing must be finite and in [0, 1)")
    for batch_index, example in enumerate(examples):
        if not example.questions:
            raise ValueError("each example must have at least one question")
        positions: set[int] = set()
        for question in example.questions:
            if (
                isinstance(question.position, bool)
                or not isinstance(question.position, int)
                or not 0 <= question.position < logits.shape[1]
            ):
                raise ValueError("question position is outside the canvas logits")
            if question.position in positions:
                raise ValueError("each example requires distinct question positions")
            positions.add(question.position)
            torch._assert_async(
                supervision_mask[batch_index, question.position],
                "decision loss requires supervised label positions",
            )


def _validate_hidden_inputs(
    hidden: torch.Tensor,
    head: torch.nn.Linear,
    examples: Sequence[DecisionLabelExample],
    supervision_mask: torch.Tensor,
    label_softmax: LabelSoftmax,
    full_ce_weighting: FullCEWeighting,
    brier_weight: float,
    hard_label_smoothing: float,
) -> None:
    _validate_inputs(
        hidden,
        examples,
        supervision_mask,
        label_softmax,
        full_ce_weighting,
        brier_weight,
        hard_label_smoothing,
    )
    if not isinstance(head, torch.nn.Linear) or head.weight.ndim != 2:
        raise TypeError("head must be a linear vocabulary projection")
    if head.weight.shape[1] != hidden.shape[2]:
        raise ValueError("head input features must match hidden states")
    if head.bias is not None and head.bias.shape != (head.weight.shape[0],):
        raise ValueError("head bias must align with vocabulary rows")


def _allowed_ids(token_ids: tuple[int, ...], vocab_size: int) -> torch.Tensor:
    if not token_ids:
        raise ValueError("a question requires at least one allowed token ID")
    if len(set(token_ids)) != len(token_ids):
        raise ValueError("allowed token IDs must be unique")
    if any(
        isinstance(token, bool)
        or not isinstance(token, int)
        or not 0 <= token < vocab_size
        for token in token_ids
    ):
        raise ValueError("allowed token ID is outside the vocabulary")
    return torch.tensor(token_ids, dtype=torch.long)


def _question_weight(target: object) -> float:
    """Optional per-question ``weight`` on a normalized label mapping (default 1)."""
    if not isinstance(target, Mapping):
        return 1.0
    weight = target.get("weight", 1.0)
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (int, float))
        or not math.isfinite(weight)
        or weight <= 0
    ):
        raise ValueError("label weight must be a positive finite number")
    return float(weight)


def _weighted_mean(
    losses: Sequence[torch.Tensor], weights: Sequence[float]
) -> torch.Tensor:
    """Question average with per-question weights; weight 1 everywhere is the plain mean."""
    stacked = torch.stack(list(losses))
    scale = torch.tensor(list(weights), device=stacked.device, dtype=stacked.dtype)
    return (stacked * scale).sum() / stacked.shape[0]


def _validate_source_weight(weight: float) -> None:
    if (
        isinstance(weight, bool)
        or not isinstance(weight, (float, int))
        or not math.isfinite(weight)
        or weight <= 0
    ):
        raise ValueError("source_weight must be a positive finite number")


def _validate_index(index: int, count: int, name: str) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
        raise ValueError(f"{name} must identify an allowed label")


def _distribution_candidates(
    target: DistributionLabel, count: int
) -> tuple[tuple[int, ...], tuple[float, ...]]:
    indices = (
        tuple(range(count))
        if target.candidate_indices is None
        else target.candidate_indices
    )
    _validate_distribution_values(
        target.probabilities, indices, target.other_probability, count
    )
    return indices, target.probabilities


def _validate_distribution_values(
    probabilities: tuple[float, ...],
    candidate_indices: tuple[int, ...] | None,
    other_probability: float,
    count: int | None = None,
) -> None:
    if not probabilities:
        raise ValueError("dist target requires at least one candidate")
    if candidate_indices is None:
        if count is not None and len(probabilities) != count:
            raise ValueError("dist target probabilities must align with allowed labels")
    elif len(candidate_indices) != len(probabilities):
        raise ValueError("dist candidate_indices must align with probs")
    elif len(set(candidate_indices)) != len(candidate_indices):
        raise ValueError("dist candidate_indices must be unique")
    elif count is not None:
        for index in candidate_indices:
            _validate_index(index, count, "dist candidate index")
    if any(
        isinstance(value, bool) or not math.isfinite(value) or value < 0
        for value in probabilities
    ):
        raise ValueError("dist target probabilities must be finite and nonnegative")
    if not math.isfinite(other_probability) or other_probability < 0:
        raise ValueError("dist other_probability must be finite and nonnegative")
    if not math.isclose(
        sum(probabilities) + other_probability, 1.0, rel_tol=0.0, abs_tol=1e-6
    ):
        raise ValueError(
            "dist target probabilities plus other_probability must sum to one"
        )


def _validate_set(indices: tuple[int, ...], count: int) -> None:
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("set target indices must be nonempty and unique")
    for index in indices:
        _validate_index(index, count, "set target index")


def _is_one_hot_distribution(target: DistributionLabel, count: int) -> bool:
    return (
        target.other_probability == 0.0
        and (target.candidate_indices is not None or len(target.probabilities) == count)
        and sum(value == 1.0 for value in target.probabilities) == 1
        and all(value in {0.0, 1.0} for value in target.probabilities)
    )


def _index(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _probability(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError("dist target probabilities must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError("dist target probabilities must be finite and nonnegative")
    return result
