"""Loss masks from `chat_template.multiturn` reach TRL's DPO log-prob sums."""

import copy

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from trl.trainer.dpo_trainer import DataCollatorForPreference

from axolotl.core.trainers.dpo.args import AxolotlDPOConfig
from axolotl.core.trainers.dpo.trainer import AxolotlDPOTrainer
from axolotl.prompt_strategies.dpo.chat_template import multiturn
from axolotl.utils.collators.preference import AxolotlDataCollatorForPreference
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline

PREFIX = [{"role": "user", "content": "Find X."}]


def _trajectory(query, tool_result, trailing_result):
    return [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": {"q": query}},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": tool_result},
        {"role": "assistant", "content": f"Answer about {query}."},
        {"role": "tool", "tool_call_id": "call_1", "content": trailing_result},
    ]


def _rows(tokenizer, chosen_trailing="done", rejected_trailing="failed"):
    transform, _ = multiturn(
        DictDefault(
            {
                "chat_template": "tokenizer_default",
                "sequence_len": 1024,
                "datasets": [{"type": "chat_template.multiturn"}],
            }
        )
    )
    rows = []
    for query in ("X", "Y"):
        sample = {
            "messages": PREFIX,
            "chosen": _trajectory(query, "found it", chosen_trailing),
            "rejected": _trajectory(f"{query}?", "nothing", rejected_trailing),
        }
        rows.append(transform(copy.deepcopy(sample), tokenizer=tokenizer))
    return rows


def _tiny_model(tokenizer, seed):
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=len(tokenizer),
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=1024,
        pad_token_id=tokenizer.pad_token_id,
    )
    return Qwen3ForCausalLM(config).float()


def _trainer(tokenizer, tmp_path, rows, loss_type, data_collator=None, **args):
    training_args = AxolotlDPOConfig(
        output_dir=str(tmp_path),
        per_device_train_batch_size=2,
        max_steps=1,
        learning_rate=1e-3,
        loss_type=loss_type,
        beta=0.5,
        max_length=1024,
        report_to="none",
        save_strategy="no",
        use_cpu=True,
        bf16=False,
        gradient_checkpointing=False,
        **args,
    )
    return AxolotlDPOTrainer(
        model=_tiny_model(tokenizer, seed=0),
        ref_model=_tiny_model(tokenizer, seed=1),
        args=training_args,
        train_dataset=Dataset.from_list(rows),
        processing_class=tokenizer,
        data_collator=data_collator,
    )


def _loss(trainer, rows):
    batch = trainer.data_collator(rows)
    with torch.no_grad():
        return trainer.compute_loss(trainer.model, batch).item()


@pytest.fixture(name="tokenizer", scope="module")
@enable_hf_offline
def fixture_tokenizer(download_qwen3_half_billion_model):
    return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")


class TestLossMaskCollator:
    def test_completion_mask_is_prompt_zeros_plus_loss_mask(self, tokenizer):
        rows = _rows(tokenizer)
        batch = AxolotlDataCollatorForPreference(pad_token_id=0)(rows)
        for i, (side, row) in enumerate(
            [("chosen", r) for r in rows] + [("rejected", r) for r in rows]
        ):
            expected = [0] * len(row["prompt_ids"]) + row[f"{side}_loss_mask"]
            got = batch["completion_mask"][i, : len(expected)].tolist()
            assert got == expected
            assert batch["completion_mask"][i, len(expected) :].sum() == 0

    def test_without_masks_matches_trl(self):
        rows = [
            {"prompt_ids": [1, 2, 3], "chosen_ids": [4, 5], "rejected_ids": [6]},
            {"prompt_ids": [7, 8], "chosen_ids": [9], "rejected_ids": [10, 11]},
        ]
        for kwargs in ({}, {"max_length": 4}, {"pad_to_multiple_of": 8}):
            ours = AxolotlDataCollatorForPreference(pad_token_id=0, **kwargs)(rows)
            theirs = DataCollatorForPreference(pad_token_id=0, **kwargs)(rows)
            assert ours.keys() == theirs.keys()
            for key in ours:
                assert torch.equal(ours[key], theirs[key])

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"max_length": 6},
            {"max_length": 6, "truncation_mode": "keep_end"},
            {"max_length": 6, "pad_to_multiple_of": 8},
        ],
    )
    def test_truncation_keeps_masks_aligned(self, kwargs):
        row = {
            "prompt_ids": [1, 2, 3],
            "chosen_ids": [10, 11, 12, 13, 14],
            "rejected_ids": [20, 21, 22, 23],
            "chosen_loss_mask": [1, 0, 1, 1, 0],
            "rejected_loss_mask": [0, 1, 0, 1],
        }
        batch = AxolotlDataCollatorForPreference(pad_token_id=0, **kwargs)([row])
        sl = (
            slice(-6, None)
            if kwargs.get("truncation_mode") == "keep_end"
            else slice(None, 6)
        )
        for i, side in enumerate(("chosen", "rejected")):
            ids = (row["prompt_ids"] + row[f"{side}_ids"])[sl]
            mask = ([0, 0, 0] + row[f"{side}_loss_mask"])[sl]
            assert batch["input_ids"][i, : len(ids)].tolist() == ids
            assert batch["completion_mask"][i, : len(mask)].tolist() == mask
            assert batch["completion_mask"][i, len(mask) :].sum() == 0
        if "pad_to_multiple_of" in kwargs:
            assert batch["completion_mask"].shape[1] == 8

    def test_length_mismatch_raises(self):
        row = {
            "prompt_ids": [1],
            "chosen_ids": [2, 3],
            "rejected_ids": [4],
            "chosen_loss_mask": [1],
            "rejected_loss_mask": [1],
        }
        with pytest.raises(ValueError, match="chosen_loss_mask"):
            AxolotlDataCollatorForPreference(pad_token_id=0)([row])


class TestDPOTrainerLossMask:
    def test_masked_dataset_uses_loss_mask_collator(self, tokenizer, tmp_path):
        trainer = _trainer(tokenizer, tmp_path, _rows(tokenizer), ["sigmoid"])
        assert isinstance(trainer.data_collator, AxolotlDataCollatorForPreference)
        trainer._set_signature_columns_if_needed()
        assert "chosen_loss_mask" in trainer._signature_columns
        assert "rejected_loss_mask" in trainer._signature_columns

    def test_rejects_collator_that_ignores_masks(self, tokenizer, tmp_path):
        with pytest.raises(ValueError, match="ignore the masks"):
            _trainer(
                tokenizer,
                tmp_path,
                _rows(tokenizer),
                ["sigmoid"],
                data_collator=DataCollatorForPreference(pad_token_id=0),
            )

    def test_unmasked_dataset_keeps_trl_collator(self, tokenizer, tmp_path):
        rows = [{"prompt": "Hi", "chosen": " there", "rejected": " no"}] * 2
        trainer = _trainer(tokenizer, tmp_path, rows, ["sigmoid"])
        assert type(trainer.data_collator) is DataCollatorForPreference

    @pytest.mark.parametrize(
        "loss_type", [["sigmoid"], ["apo_zero"], ["apo_down"], ["sigmoid", "sft"]]
    )
    def test_masked_tokens_do_not_affect_loss(self, tokenizer, tmp_path, loss_type):
        rows = _rows(tokenizer)
        perturbed = _rows(
            tokenizer,
            chosen_trailing="something entirely different and longer",
            rejected_trailing="ok",
        )
        assert rows[0]["chosen_ids"] != perturbed[0]["chosen_ids"]

        trainer = _trainer(tokenizer, tmp_path, rows, loss_type)
        loss = _loss(trainer, rows)
        assert loss == pytest.approx(_loss(trainer, perturbed), abs=1e-6)

        # control: the same perturbation moves the loss when masks are dropped
        unmasked = [
            [{k: v for k, v in r.items() if not k.endswith("_loss_mask")} for r in rs]
            for rs in (rows, perturbed)
        ]
        assert _loss(trainer, unmasked[0]) != pytest.approx(
            _loss(trainer, unmasked[1]), abs=1e-6
        )

    def test_sigmoid_loss_matches_masked_logp_sums(self, tokenizer, tmp_path):
        rows = _rows(tokenizer)
        trainer = _trainer(tokenizer, tmp_path, rows, ["sigmoid"])
        batch = trainer.data_collator(rows)

        def masked_logps(model):
            with torch.no_grad():
                logits = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                ).logits[:, :-1]
            logps = torch.gather(
                F.log_softmax(logits.float(), dim=-1),
                2,
                batch["input_ids"][:, 1:].unsqueeze(-1),
            ).squeeze(-1)
            return (logps * batch["completion_mask"][:, 1:]).sum(dim=1).chunk(2)

        chosen, rejected = masked_logps(trainer.model)
        ref_chosen, ref_rejected = masked_logps(trainer.ref_model)
        delta = (chosen - ref_chosen) - (rejected - ref_rejected)
        expected = -F.logsigmoid(trainer.beta * delta).mean().item()
        assert _loss(trainer, rows) == pytest.approx(expected, abs=1e-5)

    @pytest.mark.parametrize("loss_type", [["sigmoid"], ["apo_zero"]])
    def test_train_step(self, tokenizer, tmp_path, loss_type):
        trainer = _trainer(tokenizer, tmp_path, _rows(tokenizer), loss_type)
        result = trainer.train()
        assert torch.isfinite(torch.tensor(result.training_loss))
