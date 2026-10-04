"""Independent chains, nested proposal batches, seed stability and cancellation."""

import hashlib
import random
import threading

import pytest
from test_projection_sampling import TinyTokenizer

from axolotl.integrations.projection_sampling.args import ProjectionSamplingConfig
from axolotl.integrations.projection_sampling.backend import SamplingBackend
from axolotl.integrations.projection_sampling.batching import (
    row_seed,
    run_ordered_batch,
)
from axolotl.integrations.projection_sampling.sampler import ProjectionSampler


class BatchBackend(SamplingBackend):
    def __init__(self):
        self.tokenizer = TinyTokenizer()
        self.eos_token_ids = {7}
        self.owner = threading.get_ident()
        self.generation = []
        self.scoring = []
        self.kl = []
        self.closed = False

    @classmethod
    def from_config(cls, cfg, config):
        return cls()

    def sample(self, context, max_tokens):
        raise AssertionError("Concurrent sampling must use explicit seeds")

    def sample_batch_seeded(self, contexts, max_tokens, seeds):
        assert threading.get_ident() == self.owner
        self.generation.append((contexts, max_tokens, seeds))
        return [
            [random.Random(seed).randrange(2, 7)] * (budget - 1) + [7]
            for budget, seed in zip(max_tokens, seeds, strict=True)
        ]

    def target_logprob(self, context, tokens):
        assert threading.get_ident() == self.owner
        return -sum(8 - token for token in tokens) / 10

    def target_logprob_batch(self, contexts, tokens):
        self.scoring.append(("target", contexts, tokens))
        return super().target_logprob_batch(contexts, tokens)

    def proposal_logprob(self, context, tokens):
        assert threading.get_ident() == self.owner
        return -len(tokens)

    def proposal_logprob_batch(self, contexts, tokens):
        self.scoring.append(("proposal", contexts, tokens))
        return super().proposal_logprob_batch(contexts, tokens)

    def proposal_kl(self, target_context, proposal_context, tokens, positions):
        assert threading.get_ident() == self.owner
        self.kl.append((target_context, proposal_context, tokens, positions))
        return [0.1] * len(positions)

    def close(self):
        self.closed = True


def jobs(config, seed, count, offset=0):
    functions = []
    for index in range(offset, offset + count):

        def run(proxy, _index=index):
            sampler = ProjectionSampler(proxy, config, seed=row_seed(seed, _index))
            return sampler.sample(f"question {_index}", "expert")

        functions.append(run)
    return functions


@pytest.mark.parametrize("acceptance", ["logprob_improvement", "metropolis_hastings"])
def test_two_proposals_per_row_share_one_native_generation_batch(acceptance):
    config = ProjectionSamplingConfig(
        block_size=2,
        max_new_tokens=2,
        mcmc_steps=1,
        proposal_batch_size=2,
        acceptance=acceptance,
    )
    backend = BatchBackend()
    results = run_ordered_batch(backend, jobs(config, 42, 3), seed=42)
    assert len(results) == 3
    assert [len(contexts) for contexts, _, _ in backend.generation] == [3, 6]
    contexts, _, seeds = backend.generation[1]
    assert all(contexts[index] == contexts[index + 1] for index in range(0, 6, 2))
    assert seeds == [
        int.from_bytes(
            hashlib.sha256(f"42:{index}:{call}".encode()).digest()[:4], "big"
        )
        for index in range(3)
        for call in (1, 2)
    ]
    assert [
        len(contexts) for kind, contexts, _ in backend.scoring if kind == "target"
    ] == [3, 6]
    if acceptance == "metropolis_hastings":
        assert [
            len(contexts) for kind, contexts, _ in backend.scoring if kind == "proposal"
        ] == [9]
    else:
        assert not any(kind == "proposal" for kind, _, _ in backend.scoring)
    assert all(result.attempts == 1 for result in results)


def test_row_seeds_and_outputs_do_not_depend_on_dataset_batch_boundaries():
    config = ProjectionSamplingConfig(
        block_size=2, max_new_tokens=4, mcmc_steps=2, proposal_batch_size=2
    )
    full_backend = BatchBackend()
    full = run_ordered_batch(full_backend, jobs(config, 0, 3), seed=0)
    split_backend = BatchBackend()
    split = run_ordered_batch(split_backend, jobs(config, 0, 2), seed=0)
    split += run_ordered_batch(
        split_backend, jobs(config, 0, 1, offset=2), seed=0, row_offset=2
    )
    assert full == split
    changed = run_ordered_batch(BatchBackend(), jobs(config, 17, 3), seed=17)
    assert changed != full


def test_mixed_operations_ragged_batches_kl_and_early_completed_rows():
    backend = BatchBackend()

    def first(proxy):
        generated = proxy.sample_batch([[1], [2]], [1, 2])
        return proxy.target_logprob_batch([[1], [2]], generated)

    def second(proxy):
        assert proxy.sample_batch([], []) == []
        assert proxy.target_logprob_batch([], []) == []
        assert proxy.proposal_logprob_batch([], []) == []
        assert proxy.proposal_kl([1], [2, 3], [4, 5], [1]) == [0.1]
        return proxy.sample([3], 1)

    completed = []
    result = run_ordered_batch(
        backend,
        [first, second, lambda proxy: "cached"],
        seed=42,
        on_result=lambda index, row: completed.append(index),
    )
    assert result[2] == "cached"
    assert completed[0] == 2
    assert result[0] == [-0.1, -0.4]
    assert result[1] == [7]
    assert backend.kl == [([1], [2, 3], [4, 5], [1])]


@pytest.mark.parametrize("failure", ["worker", "backend", "alignment"])
def test_failures_release_waiting_workers_before_backend_closes(failure):
    backend = BatchBackend()

    def fails(proxy):
        if failure == "worker":
            raise ValueError("worker failed")
        return proxy.sample([1], 1)

    if failure == "backend":
        backend.sample_batch_seeded = lambda *args: (_ for _ in ()).throw(
            ValueError("backend failed")
        )
    elif failure == "alignment":
        backend.sample_batch_seeded = lambda *args: []
    with pytest.raises(ValueError):
        run_ordered_batch(backend, [lambda proxy: proxy.sample([2], 1), fails], seed=42)
    assert not any(
        thread.name.startswith("projection-sampling")
        for thread in threading.enumerate()
    )
    assert not backend.closed


def test_batch_mismatch_fails_before_submitting_request():
    backend = BatchBackend()

    def fails(proxy):
        return proxy.sample_batch([[1]], [])

    with pytest.raises(ValueError, match="align"):
        run_ordered_batch(backend, [fails], seed=42)
    assert not backend.generation
