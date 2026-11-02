"""Stochastic label dropout for configured token ids (``label_keep_prob``).

Each trained label position whose token id is configured keeps its label with
probability ``keep_prob`` and is otherwise set to -100, drawn independently per
position on every optimizer step. Inputs are untouched; only the loss target is
masked (``shift_labels`` when present, else ``labels``), so every loss path that
reads them (model CE, CCE, Liger) applies it.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping

import numpy as np
import torch

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)
IGNORE_INDEX = -100


class LabelDropout:
    def __init__(self, keep_probs: Mapping[int, float], seed: int = 0):
        items = sorted((int(k), float(v)) for k, v in keep_probs.items())
        if not items:
            raise ValueError("`label_keep_prob` must configure at least one token id.")
        self.token_ids = torch.tensor([k for k, _ in items], dtype=torch.long)
        self.keep_probs = torch.tensor([v for _, v in items], dtype=torch.float32)
        self.seed = int(seed)

    def generator(self, step: int, rank: int = 0, micro_step: int = 0):
        # SeedSequence mixes the coordinates so neighbouring steps/ranks get unrelated streams.
        state = np.random.SeedSequence(
            [self.seed, int(step), int(rank), int(micro_step)]
        ).generate_state(2, dtype=np.uint32)
        return torch.Generator().manual_seed(int(state[0]) << 32 | int(state[1]))

    def apply(self, batch: dict, generator: torch.Generator) -> dict:
        """Mask dropped positions of the loss target in place.

        The target is ``shift_labels`` when the batch carries it (e.g. context-parallel
        shards cut in the collator), since loss paths prefer it over ``labels``.
        """
        labels = batch.get("shift_labels")
        if labels is None:
            labels = batch.get("labels")
        if labels is None:
            return batch
        token_ids = self.token_ids.to(labels.device)
        matched = torch.isin(labels, token_ids)
        positions = matched.nonzero(as_tuple=True)
        if positions[0].numel() == 0:
            return batch
        values = labels[positions]
        keep_probs = self.keep_probs.to(labels.device)[
            torch.searchsorted(token_ids, values)
        ]
        draws = torch.rand(values.shape, generator=generator).to(labels.device)
        dropped = draws >= keep_probs
        labels[tuple(p[dropped] for p in positions)] = IGNORE_INDEX
        return batch

    def iter_applied(
        self, batches: Iterator, num_batches: int, step: int, rank: int
    ) -> Iterator:
        for micro_step in range(num_batches):
            try:
                batch = next(batches)
            except StopIteration:
                return
            yield self.apply(batch, self.generator(step, rank, micro_step))


def check_label_keep_prob_ids(keep_probs: Mapping[int, float], tokenizer) -> None:
    """Reject ids outside the tokenizer vocab and log what each configured id decodes to."""
    vocab_size = len(tokenizer)
    out_of_range = sorted(int(t) for t in keep_probs if int(t) >= vocab_size)
    if out_of_range:
        raise ValueError(
            f"`label_keep_prob` token ids {out_of_range} are outside the tokenizer "
            f"vocabulary (size {vocab_size}). Use the exact ids the chat template emits, "
            "e.g. from `tokenizer.convert_tokens_to_ids(...)` or `axolotl preprocess --debug`."
        )
    for token_id, keep_prob in sorted(keep_probs.items()):
        LOG.info(
            f"label_keep_prob: token id {token_id} "
            f"{tokenizer.convert_ids_to_tokens(int(token_id))!r} "
            f"(decodes to {tokenizer.decode([int(token_id)])!r}) keep_prob={keep_prob}"
        )
