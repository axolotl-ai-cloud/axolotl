from typing import Dict, List, Generator, Tuple, Optional, Any

from axolotl.utils.dict import DictDefault
from axolotl.prompters import IGNORE_TOKEN_ID
from axolotl.prompt_tokenizers import PromptTokenizingStrategy, parse_tokenized_to_result, tokenize_prompt_default

def load(tokenizer, cfg: DictDefault, _: Optional[Dict[str, Any]] = None):
    return SummarizationPromptTokenizingStrategy(
        SummarizationPrompter(),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )

class SummarizationPrompter:
    """
    A prompter that generates prompts for summarization
    """
    input = "input"
    output = "output"

    def build_prompt(self, data: Dict[str, str]) -> Generator(Tuple[str, str], None, None):
        summary = data["content"]
        name = data["placeName"]
        reviews = "\n".join(data["reviews"])
        instruction = f"""<|im_start|>system
You are a travel expert.
Your task is to abstractively summarize the reviews of an accommodation in a paragraph limited to {len(summary)} characters.
1. The summary should be purely factual and avoid personal opinion.
2. Do not compare the accommodation with any other.
3. Respond with the summary only, without any additional comments.<|im_end|>
<|im_start|>placename
{name}<|im_end|>
<|im_start|>reviews
{reviews}<|im_end|>
<|im_start|>summary
"""
        yield self.input, instruction
        yield self.output, summary


class SummarizationPromptTokenizingStrategy(PromptTokenizingStrategy):
    """
    Tokenizing strategy for prompts with functions
    """

    def tokenize_prompt(self, prompt):
        prompter: SummarizationPrompter = self.prompter
        result, current_len = tokenize_prompt_default()
        for inout, prompt in prompter.build_prompt(prompt):
            assistant_generated = inout == "output"
            tokenized = self._tokenize(prompt, add_eos_token=assistant_generated, strip_bos_token=True)
            if inout == "input" and not self.train_on_inputs:
                tokenized["labels"] = [IGNORE_TOKEN_ID] * len(
                    tokenized["input_ids"]
                )
            result, current_len = parse_tokenized_to_result(
                result, current_len,
                tokenized, tokenized["labels"], pad_token_id=self.tokenizer.pad_token_id,
            )
        return result
