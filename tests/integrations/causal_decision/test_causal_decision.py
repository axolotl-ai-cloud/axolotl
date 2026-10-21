import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, LlamaForCausalLM

from axolotl.core.training_args import AxolotlTrainingArguments
from axolotl.integrations.causal_decision.args import CausalDecisionConfig
from axolotl.integrations.causal_decision.data import (
    CausalDecisionCollator,
    decision_examples,
    letter_token_ids,
)
from axolotl.integrations.causal_decision.head import (
    HEAD_FILENAME,
    DecisionHead,
    decision_logits,
    fit_temperatures,
    soft_cross_entropy,
)
from axolotl.integrations.causal_decision.trainer import CausalDecisionTrainer
from axolotl.utils.dict import DictDefault

from ..diffusion_decision.helpers import ChatCharacterTokenizer

LETTER_IDS = [ord(letter) for letter in "ABCD"]


class Tokenizer(ChatCharacterTokenizer):
    def save_pretrained(self, *args, **kwargs):
        pass


def tiny_llama():
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
    )
    return LlamaForCausalLM(config)


def decision_trainer(output_dir, **kwargs):
    rows = [
        {
            "input_ids": [5, 6, 7][:length],
            "decision_targets": [0, 1, 0, 0],
            "num_options": 2,
        }
        for length in (1, 2, 3, 3)
    ]
    collator = CausalDecisionCollator(Tokenizer())
    trainer = CausalDecisionTrainer(
        model=tiny_llama(),
        args=AxolotlTrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=2,
            learning_rate=1e-2,
            eval_strategy="steps",
            logging_steps=1,
            report_to=[],
            use_cpu=True,
            **kwargs,
        ),
        train_dataset=Dataset.from_list(rows),
        eval_dataset=Dataset.from_list(rows),
        data_collator=collator,
        eval_data_collator=collator,
        processing_class=Tokenizer(),
    )
    trainer.axolotl_cfg = DictDefault(
        causal_decision=CausalDecisionConfig(num_options=4, rank=2)
    )
    trainer.post_set_axolotl_cfg()
    return trainer


def test_head_starts_at_lm_letter_logits():
    torch.manual_seed(0)
    model = tiny_llama().eval()
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


def test_left_padding_keeps_absolute_positions():
    torch.manual_seed(0)
    config = GPT2Config(
        vocab_size=128, n_embd=32, n_layer=1, n_head=4, bos_token_id=1, eos_token_id=11
    )
    model = GPT2LMHeadModel(config).eval()
    head = DecisionHead.from_lm_head(model, LETTER_IDS, rank=4)
    collator = CausalDecisionCollator(ChatCharacterTokenizer())

    def logits(*rows):
        batch = collator(
            [
                {"input_ids": ids, "decision_targets": [1, 0, 0, 0], "num_options": 4}
                for ids in rows
            ]
        )
        del batch["decision_targets"], batch["decision_mask"]
        with torch.no_grad():
            return decision_logits(model, head, model.lm_head, **batch)[0]

    torch.testing.assert_close(logits([5, 6, 7], [8, 9])[1], logits([8, 9])[0])


def test_trainer_saves_head_outside_model_weights_and_resumes_it(tmp_path):
    torch.manual_seed(0)
    trainer = decision_trainer(tmp_path, max_steps=2, save_steps=1, eval_steps=2)
    trainer.train()

    head = trainer.model.decision_head.state_dict()
    final = torch.load(tmp_path / HEAD_FILENAME)
    assert all(torch.equal(final[name], value) for name, value in head.items())
    assert not any("ppl" in key for log in trainer.state.log_history for key in log)
    assert not trainer.model.lm_head.weight.requires_grad
    weights = load_file(tmp_path / "checkpoint-1" / "model.safetensors")
    assert not any(key.startswith("decision_head.") for key in weights)

    saved = torch.load(tmp_path / "checkpoint-1" / HEAD_FILENAME)
    resumed = decision_trainer(tmp_path)
    resumed._load_from_checkpoint(str(tmp_path / "checkpoint-1"))
    for name, value in resumed.model.decision_head.state_dict().items():
        assert torch.equal(value, saved[name])


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
