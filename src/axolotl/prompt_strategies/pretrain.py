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
        # Tokenise the full document first (large max_length mirrors completion.py),
        # then split into non-overlapping chunks.  This ensures BOS appears only at
        # the start of the first chunk and EOS only at the end of the last chunk,
        # matching the token distribution produced by `type: completion`.
        result = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length * 64,
            padding=False,
            return_tensors=None,
        )
        input_ids = result["input_ids"]
        attention_mask = result["attention_mask"]

        if (
            add_eos_token
            and len(input_ids) > 0
            and input_ids[-1] != self.tokenizer.eos_token_id
            and len(input_ids) < self.max_length * 64
        ):
            input_ids.append(self.tokenizer.eos_token_id)
            attention_mask.append(1)

        chunked_input_ids = [
            input_ids[i : i + self.max_length]
            for i in range(0, len(input_ids), self.max_length)
        ]
        chunked_attention_mask = [
            attention_mask[i : i + self.max_length]
            for i in range(0, len(attention_mask), self.max_length)
        ]

        return BatchEncoding(
            data={
                "input_ids": chunked_input_ids,
                "attention_mask": chunked_attention_mask,
            }
        )

    def tokenize_prompt(self, prompt):
        return self._tokenize(prompt[self.text_column])


def load(tokenizer, cfg):
    strat = PretrainTokenizationStrategy(
        PretrainTokenizer(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        text_column=cfg.pretraining_dataset[0]["text_column"] or "text",
        max_length=cfg.sequence_len,
    )
    return strat
