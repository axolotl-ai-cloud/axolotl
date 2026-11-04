import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from axolotl.integrations.causal_decision.data import (
    CausalDecisionCollator,
    decision_examples,
    letter_token_ids,
)
from axolotl.integrations.causal_decision.head import (
    DecisionHead,
    decision_logits,
    fit_temperatures,
    soft_cross_entropy,
)

from ..diffusion_decision.helpers import ChatCharacterTokenizer

LETTER_IDS = [ord(letter) for letter in "ABCD"]


def test_head_starts_at_lm_letter_logits():
    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    model = LlamaForCausalLM(config).eval()
    head = DecisionHead.from_lm_head(model, LETTER_IDS, rank=4)
    assert torch.equal(head.weight, model.lm_head.weight[LETTER_IDS])
    assert not head.up.any()

    batch = CausalDecisionCollator(ChatCharacterTokenizer())(
        [
            {
                "input_ids": [5, 6, 7],
                "decision_targets": [1, 0, 0, 0],
                "num_options": 2,
            },
            {"input_ids": [8, 9], "decision_targets": [0, 1, 0, 0], "num_options": 4},
        ]
    )
    with torch.no_grad():
        logits, outputs = decision_logits(
            model,
            head,
            model.lm_head,
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
    torch.testing.assert_close(logits, outputs.logits[:, -1, LETTER_IDS])


def test_soft_cross_entropy_ignores_masked_options():
    logits = torch.tensor([[1.0, 2.0, 3.0, 50.0]])
    targets = torch.tensor([[0.2, 0.3, 0.5, 0.0]])
    mask = torch.tensor([[True, True, True, False]])
    expected = -(targets[0, :3] * logits[0, :3].log_softmax(-1)).sum()
    torch.testing.assert_close(soft_cross_entropy(logits, targets, mask), expected)


def test_fit_temperatures_recovers_known_temperatures():
    torch.manual_seed(0)
    logits = 3 * torch.randn(400, 4)
    mask = torch.ones_like(logits, dtype=torch.bool)
    true = torch.tensor([2.0] * 200 + [0.5] * 200)[:, None]
    targets = (logits / true).softmax(-1)
    fitted = fit_temperatures(logits, targets, mask, ["a"] * 200 + ["b"] * 200)
    assert fitted == pytest.approx({"a": 2.0, "b": 0.5}, rel=1e-3)


def test_decision_examples_read_each_question_through_the_letter_head():
    tokenizer = ChatCharacterTokenizer()
    assert letter_token_ids(tokenizer, 4) == LETTER_IDS
    record = {
        "state": "s",
        "questions": {
            "act": {"type": "choice", "options": ["x", "y", "z"]},
            "ok": {"type": "noul"},
        },
        "labels": {
            "act": {"kind": "dist", "probs": [0.2, 0.0, 0.8]},
            "ok": {"kind": "set", "allowed_set": [1]},
        },
    }
    act, ok = decision_examples(tokenizer, record, LETTER_IDS)
    assert act["input_ids"] == [99, *map(ord, "act: ")]
    assert act["decision_targets"] == [0.2, 0.0, 0.8, 0.0]
    assert act["num_options"] == 3
    assert ok["decision_targets"] == [0.0, 1.0, 0.0, 0.0]
    assert ok["num_options"] == 2
