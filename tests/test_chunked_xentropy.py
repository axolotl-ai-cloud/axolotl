"""
test suite for chunked cross entropy
"""

import pytest
import torch
from torch import nn

from axolotl.monkeypatch.loss.chunked import get_causal_lm_loss


@pytest.fixture
def chunked_fixtures():
    model_dim = 512
    vocab_size = 1024 * 256
    seq_len = 2048
    batch_size = 1

    lm_head = nn.Linear(model_dim, vocab_size)
    hidden_state = torch.randn(batch_size, seq_len, model_dim)
    labels = torch.randint(low=0, high=vocab_size, size=(batch_size, seq_len))
    return lm_head, hidden_state, labels, vocab_size


def test_chunked_forward(chunked_fixtures):
    lm_head, hidden_state, labels, vocab_size = chunked_fixtures
    lm_loss = get_causal_lm_loss()

    logits = lm_head(hidden_state)

    chunked_lm_loss = lm_loss(logits, labels)

    logits_flattened = logits.view(-1, vocab_size)
    labels_flattened = labels.view(-1)

    loss = nn.functional.cross_entropy(
        logits_flattened.float(), labels_flattened, reduction="mean"
    )

    assert torch.allclose(chunked_lm_loss, loss, atol=1e-2, rtol=1e-2)
