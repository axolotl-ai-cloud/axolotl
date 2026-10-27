"""Module for testing prompt tokenizers."""

import json
from pathlib import Path

from axolotl.prompt_strategies.alpaca_chat import NoSystemPrompter
from axolotl.prompt_strategies.alpaca_w_system import (
    InstructionWSystemPromptTokenizingStrategy,
    SystemDataPrompter,
)
from axolotl.prompt_strategies.bradley_terry.chat_template import load as bt_load
from axolotl.prompt_strategies.llama2_chat import (
    Llama2ChatPrompter,
    LLama2ChatTokenizingStrategy,
)
from axolotl.prompt_strategies.orpo.chat_template import (
    ORPODatasetParsingStrategy,
    load,
)
from axolotl.prompt_strategies.user_defined import (
    UserDefinedDatasetConfig,
    load as user_defined_load,
)
from axolotl.prompt_tokenizers import AlpacaPromptTokenizingStrategy
from axolotl.prompters import AlpacaPrompter, PromptStyle
from axolotl.utils.dict import DictDefault

from tests.hf_offline_utils import enable_hf_offline

test_data = {
    "multi_turn_sys": {
        "conversations": [
            {"from": "system", "value": "lorem"},
            {"from": "human", "value": "abc"},
            {"from": "gpt", "value": "ipsum"},
            {"from": "human", "value": "123"},
            {"from": "gpt", "value": "sit"},
        ]
    },
    "single_turn_sys": {
        "conversations": [
            {"from": "system", "value": "lorem"},
            {"from": "human", "value": "abc"},
            {"from": "gpt", "value": "ipsum"},
        ]
    },
    "single_turn_no_sys": {
        "conversations": [
            {"from": "human", "value": "abc"},
            {"from": "gpt", "value": "ipsum"},
        ]
    },
    "multi_turn_no_sys": {
        "conversations": [
            {"from": "human", "value": "abc"},
            {"from": "gpt", "value": "ipsum"},
            {"from": "human", "value": "123"},
            {"from": "gpt", "value": "sit"},
        ]
    },
}


class TestPromptTokenizationStrategies:
    """
    Test class for prompt tokenization strategies.
    """

    @enable_hf_offline
    def test_no_sys_prompt(self, tokenizer_huggyllama_w_special_tokens):
        """
        tests the interface between the user and assistant parts
        """
        prompter = NoSystemPrompter()

        strat = AlpacaPromptTokenizingStrategy(
            prompter,
            tokenizer_huggyllama_w_special_tokens,
            False,
            2048,
        )
        sample = {
            "instruction": "hello cruel. lorem ipsum dolor sit amet.",
            "output": "world!",
        }
        example = strat.tokenize_prompt(sample)
        world_idx = example["input_ids"].index(3186)
        assert example["labels"][world_idx] == 3186
        assert example["labels"][world_idx - 1] == -100

    @enable_hf_offline
    def test_alpaca(self, tokenizer_huggyllama_w_special_tokens):
        """
        tests the interface between the user and assistant parts
        """

        prompter = AlpacaPrompter()
        strat = AlpacaPromptTokenizingStrategy(
            prompter,
            tokenizer_huggyllama_w_special_tokens,
            False,
            2048,
        )
        sample = {"instruction": "hello!", "output": "Hi! How can I help?"}
        example = strat.tokenize_prompt(sample)
        world_idx = example["input_ids"].index(6324)
        assert example["labels"][world_idx] == 6324
        assert example["labels"][world_idx - 1] == -100


class TestInstructionWSystemPromptTokenizingStrategy:
    """
    Test class for prompt tokenization strategies with sys prompt from the dataset
    """

    @enable_hf_offline
    def test_system_alpaca(self, tokenizer_huggyllama_w_special_tokens):
        prompter = SystemDataPrompter(PromptStyle.CHAT.value)
        strat = InstructionWSystemPromptTokenizingStrategy(
            prompter,
            tokenizer_huggyllama_w_special_tokens,
            False,
            2048,
        )
        sample = {
            "system": "use cot",
            "instruction": "hello!",
            "output": "Hi! How can I help?",
        }
        example = strat.tokenize_prompt(sample)
        assert example["input_ids"][0:5] == [
            1,
            28962,
            1254,
            12665,
            29901,
        ]  # "<s>SYSTEM:"
        assert example["input_ids"][5:7] == [671, 20118]  # " use cot"
        assert example["input_ids"][8] == 11889  # USER

    @enable_hf_offline
    def test_user_defined_null_system_uses_system_prompt(
        self, tokenizer_huggyllama_w_special_tokens
    ):
        strat = user_defined_load(
            tokenizer_huggyllama_w_special_tokens,
            DictDefault({"train_on_inputs": False, "sequence_len": 2048}),
            UserDefinedDatasetConfig(
                system_prompt="use cot", system_format="SYSTEM: {system}\n"
            ),
        )
        sample = {"system": None, "instruction": "hello!", "output": "Hi!"}
        example = strat.tokenize_prompt(sample)
        decoded = tokenizer_huggyllama_w_special_tokens.decode(example["input_ids"])
        assert "SYSTEM: use cot" in decoded

        sample = {"system": "be brief", "instruction": "hello!", "output": "Hi!"}
        example = strat.tokenize_prompt(sample)
        decoded = tokenizer_huggyllama_w_special_tokens.decode(example["input_ids"])
        assert "SYSTEM: be brief" in decoded
        assert "use cot" not in decoded


class TestLlama2ChatTokenization:
    """
    Test class for prompt tokenization strategies with sys prompt from the dataset
    """

    @enable_hf_offline
    def test_llama2_chat_integration(self, tokenizer_llama2_7b):
        with open(
            Path(__file__).parent / "fixtures/conversation.json", encoding="utf-8"
        ) as fin:
            data = fin.read()
            conversation = json.loads(data)
        with open(
            Path(__file__).parent / "fixtures/conversation.tokenized_llama2chat.json",
            encoding="utf-8",
        ) as fin:
            data = fin.read()
            tokenized_conversation = json.loads(data)
        prompter = Llama2ChatPrompter()
        strat = LLama2ChatTokenizingStrategy(
            prompter,
            tokenizer_llama2_7b,
            False,
            4096,
        )
        example = strat.tokenize_prompt(conversation)
        for fields in ["input_ids", "attention_mask", "labels"]:
            # pytest assert equals

            assert len(example[fields]) == len(tokenized_conversation[fields])
            assert example[fields] == tokenized_conversation[fields]


class TestOrpoTokenization:
    """test case for the ORPO tokenization"""

    @enable_hf_offline
    def test_orpo_integration(
        self,
        tokenizer_mistral_7b_instruct_chatml,
        dataset_argilla_ultrafeedback_binarized_preferences_cleaned,
    ):
        ds = dataset_argilla_ultrafeedback_binarized_preferences_cleaned.select([0])
        strat = load(
            tokenizer_mistral_7b_instruct_chatml,
            DictDefault({"train_on_inputs": False}),
            DictDefault({"chat_template": "chatml"}),
        )
        res = strat.tokenize_prompt(ds[0])
        assert "rejected_ids" in res
        assert "rejected_labels" in res
        assert "input_ids" in res
        assert "labels" in res
        assert "prompt_attention_mask" in res

        assert len(res["rejected_ids"]) == len(res["rejected_labels"])
        assert len(res["input_ids"]) == len(res["labels"])
        assert len(res["input_ids"]) == len(res["prompt_attention_mask"])

        assert res["rejected_labels"][0] == -100
        assert res["rejected_ids"][-1] == res["rejected_labels"][-1]

        assert res["labels"][0] == -100
        assert res["input_ids"][-1] == res["labels"][-1]

        assert res["prompt_attention_mask"][0] == 1
        assert res["prompt_attention_mask"][-1] == 0

    def test_orpo_null_prompt_uses_chosen_turn(self):
        sample = {
            "prompt": None,
            "chosen": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "good"},
            ],
            "rejected": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "bad"},
            ],
        }
        messages = ORPODatasetParsingStrategy().get_prompt(sample).messages
        assert messages[-1].content == "q"


class TestBradleyTerryTokenization:
    """test case for the Bradley-Terry chat_template tokenization"""

    @enable_hf_offline
    def test_bt_null_system(self, tokenizer_huggyllama_w_special_tokens):
        strat = bt_load(
            tokenizer_huggyllama_w_special_tokens,
            DictDefault(
                {
                    "train_on_inputs": False,
                    "sequence_len": 2048,
                    "chat_template": "chatml",
                    "reward_model": True,
                }
            ),
        )
        sample = {"system": None, "input": "hi", "chosen": "a", "rejected": "b"}
        res = strat.tokenize_prompt(sample)
        assert res["chosen_ids"] and res["rejected_ids"]
