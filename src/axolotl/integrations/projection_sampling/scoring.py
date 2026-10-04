"""Compare final replies using only their causally predicted loss labels."""

import math

from .backend import SamplingBackend


def evaluate_logprob_margin(
    backend: SamplingBackend,
    original: dict,
    rewritten: dict,
    margin: float,
    *,
    starts: tuple[int, int],
) -> dict:
    """Score labeled reply spans while retaining masked tokens as conditioning."""
    contexts: list[list[int]] = []
    continuations: list[list[int]] = []
    groups: list[tuple[int, int, int]] = []
    for example, start in zip((original, rewritten), starts, strict=True):
        ids, labels = example["input_ids"], example["labels"]
        if len(ids) != len(labels):
            raise ValueError("Reply token IDs and labels must align")
        first = len(continuations)
        count = 0
        index = max(1, start)
        while index < len(ids):
            if labels[index] == -100:
                index += 1
                continue
            end = index + 1
            while end < len(ids) and labels[end] != -100:
                end += 1
            contexts.append(ids[:index])
            continuations.append(ids[index:end])
            count += end - index
            index = end
        groups.append((first, len(continuations), count))
    scores = (
        backend.target_logprob_batch(contexts, continuations) if continuations else []
    )
    if len(scores) != len(continuations) or not all(
        math.isfinite(score) for score in scores
    ):
        raise ValueError("Backend returned invalid labeled-reply log probabilities")
    means = [
        math.fsum(scores[first:end]) / count if count else None
        for first, end, count in groups
    ]
    improvement = (
        means[1] - means[0] if means[0] is not None and means[1] is not None else None
    )
    return {
        "min_logprob_improvement": margin,
        "original_mean_logprob": means[0],
        "rewritten_mean_logprob": means[1],
        "original_labeled_tokens": groups[0][2],
        "rewritten_labeled_tokens": groups[1][2],
        "logprob_improvement": improvement,
        "logprob_margin_passed": improvement is not None and improvement > margin,
    }


def evaluate_proposal_kl(
    backend: SamplingBackend,
    target_context: list[int],
    proposal_context: list[int],
    tokens: list[int],
    tokenized: dict,
    ceiling: float,
) -> dict:
    """Gate the final rewrite using matching response-prefix conditionals."""
    offset = len(target_context)
    if (
        tokenized["input_ids"][:offset] != target_context
        or tokenized["input_ids"][offset : offset + len(tokens)] != tokens
        or len(tokenized["input_ids"]) != len(tokenized["labels"])
    ):
        raise ValueError("KL reply IDs and parser labels must retain the sampled trace")
    positions = [
        position
        for position in range(len(tokens))
        if tokenized["labels"][offset + position] != -100
    ]
    scores = (
        backend.proposal_kl(target_context, proposal_context, tokens, positions)
        if positions
        else []
    )
    if len(scores) != len(positions) or not all(
        math.isfinite(value) and value >= 0 for value in scores
    ):
        raise ValueError("Backend returned invalid conditional proposal KL")
    mean = math.fsum(scores) / len(scores) if scores else None
    return {
        "max_proposal_kl": ceiling,
        "proposal_to_base_mean_kl": mean,
        "proposal_kl_labeled_tokens": len(scores),
        "proposal_kl_gate_passed": mean is not None and mean <= ceiling,
    }
