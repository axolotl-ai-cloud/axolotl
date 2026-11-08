"""Axolotl's DataLoader worker seeding matches transformers'."""

import random

import numpy as np
import pytest
import torch

from axolotl.utils.worker_seed import seed_worker


@pytest.fixture(autouse=True)
def _restore_rng_state():
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    yield
    random.setstate(state[0])
    np.random.set_state(state[1])
    torch.set_rng_state(state[2])


def _draws(fn, num_workers, rank):
    torch.manual_seed(1234)
    fn(0, num_workers=num_workers, rank=rank)
    return random.random(), float(np.random.rand()), float(torch.rand(1))


@pytest.mark.parametrize("num_workers, rank", [(1, 0), (4, 0), (4, 3)])
def test_matches_transformers_seed_worker(num_workers, rank):
    from transformers.trainer_utils import seed_worker as hf_seed_worker

    assert _draws(seed_worker, num_workers, rank) == _draws(
        hf_seed_worker, num_workers, rank
    )


def test_ranks_get_different_streams():
    assert _draws(seed_worker, 4, 0) != _draws(seed_worker, 4, 1)
