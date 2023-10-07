"""Module for testing prompt tokenizers."""
import json
import logging
import unittest
from pathlib import Path
from typing import Optional

import pytest
from transformers import AutoTokenizer, LlamaTokenizer

from axolotl.prompt_strategies.alpaca_chat import NoSystemPrompter
from axolotl.prompt_strategies.alpaca_w_system import (
    InstructionWSystemPromptTokenizingStrategy,
    SystemDataPrompter,
)
from axolotl.prompt_strategies.completion import (
    CompletionPrompter,
    CompletionPromptTokenizingStrategy,
)
from axolotl.prompt_strategies.llama2_chat import (
    Llama2ChatPrompter,
    LLama2ChatTokenizingStrategy,
)
from axolotl.prompt_tokenizers import (
    AlpacaPromptTokenizingStrategy,
    ShareGPTPromptTokenizingStrategy,
)
from axolotl.prompters import AlpacaPrompter, PromptStyle, ShareGPTPrompterV2

LOG = logging.getLogger("axolotl")


class TestPromptTokenizationStrategies(unittest.TestCase):
    """
    Test class for prompt tokenization strategies.
    """

    _caplog: Optional[pytest.LogCaptureFixture] = None

    @pytest.fixture(autouse=True)
    def inject_fixtures(self, caplog):
        self._caplog = caplog

    def setUp(self) -> None:
        # pylint: disable=duplicate-code
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<pad>",
            }
        )

    def test_sharegpt_integration(self):
        with open(
            Path(__file__).parent / "fixtures/conversation.json", encoding="utf-8"
        ) as fin:
            data = fin.read()
            conversation = json.loads(data)
        with open(
            Path(__file__).parent / "fixtures/conversation.tokenized.json",
            encoding="utf-8",
        ) as fin:
            data = fin.read()
            tokenized_conversation = json.loads(data)
        prompter = ShareGPTPrompterV2()
        strat = ShareGPTPromptTokenizingStrategy(
            prompter,
            self.tokenizer,
            False,
            2048,
        )
        example = strat.tokenize_prompt(conversation)
        for fields in ["input_ids", "attention_mask", "labels"]:
            self.assertEqual(len(example[fields]), len(tokenized_conversation[fields]))
            self.assertEqual(example[fields], tokenized_conversation[fields])

    def test_sharegpt_warnings_integration(self):
        with open(
            Path(__file__).parent / "fixtures/conversation.missingturns.json",
            encoding="utf-8",
        ) as fin:
            data = fin.read()
            conversation = json.loads(data)
        prompter = ShareGPTPrompterV2()
        strat = ShareGPTPromptTokenizingStrategy(
            prompter,
            self.tokenizer,
            False,
            2048,
        )
        with self._caplog.at_level(logging.WARNING):
            strat.tokenize_prompt(conversation)
            assert "assistant turn has empty text" in self._caplog.records[1].message

    def test_no_sys_prompt(self):
        """
        tests the interface between the user and assistant parts
        """
        prompter = NoSystemPrompter()
        # pylint: disable=duplicate-code
        strat = AlpacaPromptTokenizingStrategy(
            prompter,
            self.tokenizer,
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

    def test_alpaca(self):
        """
        tests the interface between the user and assistant parts
        """
        # pylint: disable=duplicate-code
        prompter = AlpacaPrompter()
        strat = AlpacaPromptTokenizingStrategy(
            prompter,
            self.tokenizer,
            False,
            2048,
        )
        sample = {"instruction": "hello!", "output": "Hi! How can I help?"}
        example = strat.tokenize_prompt(sample)
        world_idx = example["input_ids"].index(6324)
        assert example["labels"][world_idx] == 6324
        assert example["labels"][world_idx - 1] == -100

    def test_completion_strategy(self):
        """
        tests the completion prompt tokenization strategy
        """
        # pylint: disable=duplicate-code
        self.tokenizer.padding_side = "left"
        for text in [
            # ['▁Once', '▁upon', '▁a', '▁time', ',', '▁there', '▁was', '▁a', '▁dog', '.'] [10]
            "Once upon a time, there was a dog.",  # fits in one sample at 12 ctxlen
            # ['▁Once', '▁upon', '▁a', '▁time', ',', '▁there', '▁was', '▁a', '▁dog', '.', '▁The', '▁dog', '▁was', '▁very', '▁happy', '.'] [16]
            "Once upon a time, there was a dog. The dog was very happy.",  # fits in two samples
            # ['▁Once', '▁upon', '▁a', '▁time', ',', '▁there', '▁was', '▁a', '▁dog', '.', '▁The', '▁dog', '▁was', '▁very', '▁happy', '.', '▁It', '▁was', '▁in', '▁fact', '▁so', '▁happy', '▁that', '▁it', '▁emb', 'ark', 'ed', '▁upon', '▁a', '▁cr', 'us', 'ade', '▁to', '▁save', '▁human', 'ity', '▁from', '▁the', '▁ev', 'ils', '▁of', '▁man', 'kind', '.'] [44]
            "Once upon a time, there was a dog. The dog was very happy. It was in fact so happy that it embarked upon a crusade to save humanity from the evils of mankind.",  # requires 4 samples
        ]:
            prompt_sample = {"text": [text]}
            tokenized = self.tokenizer.tokenize(text)

            strat = CompletionPromptTokenizingStrategy(
                CompletionPrompter(),
                self.tokenizer,
                False,
                12,
                max_length=12 * 64,
                align_samples=self.tokenizer.padding_side == "left",
            )

            example = strat.tokenize_prompt(prompt_sample)
            # The first sample should have 0+ padding followed by the start of the text
            # All padding should also have attention mask 0
            is_padding = True
            did_end = False
            tokenized_idx = 0
            for sample_idx, sample in enumerate(example["input_ids"]):
                attention_mask = example["attention_mask"][sample_idx]
                comp_tokens = self.tokenizer.convert_ids_to_tokens(sample)
                for idx, token in enumerate(sample):
                    if tokenized_idx == len(tokenized):
                        # We must have reached the end of the tokenized text
                        assert token == self.tokenizer.eos_token_id
                        assert idx + 1 == len(sample)
                        assert not did_end
                        did_end = True
                        continue
                    if is_padding:
                        if token != self.tokenizer.pad_token_id:
                            # Must be the BOS token
                            assert token == self.tokenizer.bos_token_id
                            assert attention_mask[idx] == 1
                            is_padding = False
                            continue
                        assert attention_mask[idx] == 0
                    else:
                        comp_token = comp_tokens[idx]
                        assert tokenized[tokenized_idx] == comp_token
                        tokenized_idx += 1
            # We must have reached the end of the tokenized text
            assert tokenized_idx == len(tokenized)
            assert did_end


class InstructionWSystemPromptTokenizingStrategyTest(unittest.TestCase):
    """
    Test class for prompt tokenization strategies with sys prompt from the dataset
    """

    def setUp(self) -> None:
        # pylint: disable=duplicate-code
        self.tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
        self.tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
            }
        )

    def test_system_alpaca(self):
        prompter = SystemDataPrompter(PromptStyle.CHAT.value)
        strat = InstructionWSystemPromptTokenizingStrategy(
            prompter,
            self.tokenizer,
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


class Llama2ChatTokenizationTest(unittest.TestCase):
    """
    Test class for prompt tokenization strategies with sys prompt from the dataset
    """

    def setUp(self) -> None:
        # pylint: disable=duplicate-code
        self.tokenizer = LlamaTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
        # woraround because official Meta repos are not open

    def test_llama2_chat_integration(self):
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
            self.tokenizer,
            False,
            4096,
        )
        example = strat.tokenize_prompt(conversation)
        for fields in ["input_ids", "attention_mask", "labels"]:
            self.assertEqual(len(example[fields]), len(tokenized_conversation[fields]))
            self.assertEqual(example[fields], tokenized_conversation[fields])

    def compare_with_transformers_integration(self):
        # this needs transformers >= v4.31.0
        from transformers.models.llama.tokenization_llama import B_SYS, E_SYS
        from transformers.pipelines.conversational import Conversation

        # from transformers.models.llama.tokenization_llama import DEFAULT_SYSTEM_PROMPT
        # broken as of 23/7/20
        # see https://github.com/huggingface/transformers/pull/24935
        # pylint: disable=C0103
        DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""
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

        user_input = []
        answers = []
        for msg in conversation["conversations"]:
            if msg["from"] == "human":
                user_input.append(msg["value"])
            else:
                answers.append(msg["value"])
        hf_conf = Conversation(
            text=user_input[-1],
            past_user_inputs=[B_SYS + DEFAULT_SYSTEM_PROMPT + E_SYS + user_input[0]]
            + user_input[1:-1],
            generated_responses=answers,
        )
        # pylint: disable=W0212
        hf_tokens = self.tokenizer._build_conversation_input_ids(hf_conf)

        self.assertEqual(
            hf_tokens, tokenized_conversation["input_ids"][: len(hf_tokens)]
        )


if __name__ == "__main__":
    unittest.main()
