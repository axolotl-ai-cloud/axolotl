"""pretraining prompt strategies"""

from typing import Generator

from transformers import BatchEncoding

from axolotl.prompt_tokenizers import PromptTokenizingStrategy


class PretrainTokenizer:
    """basic tokenization class for pretraining"""

    def build_prompt(self, prompt) -> Generator[str, None, None]:
        yield prompt


class PretrainTokenizationStrategy(PromptTokenizingStrategy):
    """handles tokenization for pretraining with strides"""

    @property
    def supports_batched(self):
        return True

    def __init__(self, *args, max_length=None, text_column="text", **kwargs):
        super().__init__(*args, **kwargs)
        if max_length:
            self.max_length = max_length
        self.text_column = text_column

    def _tokenize(
        self, prompt: str, add_eos_token: bool = True, strip_bos_token: bool = False
    ) -> BatchEncoding:
        # keep the overlap below the window size so small sequence_len values don't
        # violate the tokenizer's `stride < effective max_length` constraint
        stride = min(256, (self.max_length - 1) // 2)
        res = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length - 1,
            add_special_tokens=True,
            return_overflowing_tokens=True,
            stride=stride,
        )
        res["input_ids"] = [
            seq + [self.tokenizer.eos_token_id] for seq in res["input_ids"]
        ]
        res["attention_mask"] = [seq + [1] for seq in res["attention_mask"]]

        return res

    def tokenize_prompt(self, prompt):
        return self._tokenize(prompt[self.text_column])


def load(tokenizer, cfg):
    strat = PretrainTokenizationStrategy(
        PretrainTokenizer(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        text_column=cfg.pretraining_dataset[0]["text_column"] or "text",
        # windows; larger values produce windows that the downstream
        # `filter_sequences_by_length` drops wholesale
        max_length=cfg.sequence_len,
    )
    return strat
