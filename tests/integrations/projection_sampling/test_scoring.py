"""Final reply comparisons exclude masked tokens while retaining their context."""

import pytest
import torch
from test_projection_sampling import ScriptedBackend, TinyTokenizer
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
)

from axolotl.integrations.projection_sampling.args import ProjectionSamplingConfig
from axolotl.integrations.projection_sampling.backends.transformers import (
    TransformersBackend,
)
from axolotl.integrations.projection_sampling.scoring import (
    evaluate_logprob_margin,
    evaluate_proposal_kl,
)


def test_labeled_spans_match_causally_shifted_model_scores():
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=16, n_positions=32, n_embd=8, n_layer=1, n_head=1)
    ).eval()
    backend = TransformersBackend(model, TinyTokenizer(), ProjectionSamplingConfig())
    original = {
        "input_ids": [1, 2, 3, 4, 5, 6, 7],
        "labels": [1, 2, 3, 4, -100, 6, -100],
    }
    rewritten = {
        "input_ids": [1, 2, 3, 8, 5, 9, 7],
        "labels": [1, 2, 3, 8, -100, 9, -100],
    }
    result = evaluate_logprob_margin(backend, original, rewritten, 0, starts=(3, 3))
    for name, example in (("original", original), ("rewritten", rewritten)):
        ids = torch.tensor([example["input_ids"]])
        with torch.inference_mode():
            logits = model(ids).logits[0].float().log_softmax(-1)
        expected = torch.stack([logits[2, ids[0, 3]], logits[4, ids[0, 5]]]).mean()
        assert result[name + "_mean_logprob"] == pytest.approx(expected.item())
        assert result[name + "_labeled_tokens"] == 2


def test_margin_without_response_labels_fails_closed():
    backend = ScriptedBackend([])
    example = {"input_ids": [1, 2], "labels": [-100, -100]}
    result = evaluate_logprob_margin(backend, example, example, 0, starts=(1, 1))
    assert result["logprob_margin_passed"] is False
    assert result["logprob_improvement"] is None
    assert not backend.calls


@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_backend_scores_raise(invalid):
    backend = ScriptedBackend([], target=lambda tokens: invalid)
    example = {"input_ids": [1, 2], "labels": [-100, 2]}
    with pytest.raises(ValueError, match="invalid labeled-reply"):
        evaluate_logprob_margin(backend, example, example, 0, starts=(1, 1))


def test_transformers_kl_matches_independent_processed_prefix_forwards():
    model = GPT2LMHeadModel(
        GPT2Config(vocab_size=16, n_positions=32, n_embd=8, n_layer=1, n_head=1)
    ).eval()
    config = ProjectionSamplingConfig(temperature=0.6, repetition_penalty=1.1)
    backend = TransformersBackend(model, TinyTokenizer(), config)
    target, proposal, tokens, positions = [1, 2], [1, 3, 4], [3, 5, 2], [2, 0]
    expected = []
    with torch.inference_mode():
        for position in positions:
            base_ids = torch.tensor([target + tokens[:position]])
            proposal_ids = torch.tensor([proposal + tokens[:position]])
            base = model(base_ids).logits[:, -1].float().log_softmax(-1)
            logits = model(proposal_ids).logits[:, -1].float()
            logits = RepetitionPenaltyLogitsProcessor(1.1)(proposal_ids, logits)
            logits = TemperatureLogitsWarper(0.6)(proposal_ids, logits)
            logq = logits.log_softmax(-1)
            expected.append((logq.exp() * (logq - base)).sum().item())
    assert backend.proposal_kl(target, proposal, tokens, positions) == pytest.approx(
        expected, abs=1e-6
    )
    assert backend.proposal_kl(target, proposal, tokens, []) == []


@pytest.mark.parametrize("ceiling,passed", [(0.25, True), (0.249, False)])
def test_kl_gate_selects_parser_labels_and_retains_masked_context(ceiling, passed):
    from unittest.mock import Mock

    backend = ScriptedBackend([])
    backend.proposal_kl = Mock(return_value=[0.1, 0.4])
    example = {"input_ids": [1, 2, 3, 4, 5], "labels": [1, 2, 3, -100, 5]}
    metadata = evaluate_proposal_kl(
        backend, [1, 2], [6, 7], [3, 4, 5], example, ceiling
    )
    backend.proposal_kl.assert_called_once_with([1, 2], [6, 7], [3, 4, 5], [0, 2])
    assert metadata["proposal_to_base_mean_kl"] == 0.25
    assert metadata["proposal_kl_labeled_tokens"] == 2
    assert metadata["proposal_kl_gate_passed"] is passed


def test_kl_gate_without_labeled_continuation_fails_closed():
    from unittest.mock import Mock

    backend = ScriptedBackend([])
    backend.proposal_kl = Mock(side_effect=AssertionError("no labeled positions"))
    example = {"input_ids": [1, 2, 3], "labels": [1, 2, -100]}
    result = evaluate_proposal_kl(backend, [1, 2], [6], [3], example, 1)
    assert result["proposal_kl_gate_passed"] is False
    assert result["proposal_to_base_mean_kl"] is None


@pytest.mark.parametrize("scores", [[float("nan")], [float("inf")], [-0.1], []])
def test_kl_gate_invalid_backend_values_raise(scores):
    from unittest.mock import Mock

    backend = ScriptedBackend([])
    backend.proposal_kl = Mock(return_value=scores)
    example = {"input_ids": [1, 2], "labels": [-100, 2]}
    with pytest.raises(ValueError, match="invalid conditional"):
        evaluate_proposal_kl(backend, [1], [3], [2], example, 1)
