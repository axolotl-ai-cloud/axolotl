"""Preserve sampled token IDs and mask the unprivileged question for SFT."""

from axolotl.prompt_tokenizers import PromptTokenizingStrategy


class ProjectionTokenizationStrategy(PromptTokenizingStrategy):
    """Read the exact target context and completion from the sampling cache."""

    def tokenize_prompt(self, prompt):
        context = prompt["prompt_token_ids"]
        response = prompt["response_token_ids"]
        ids = context + response
        labels = (
            ids.copy() if self.train_on_inputs else [-100] * len(context) + response
        )
        return {"input_ids": ids, "labels": labels, "attention_mask": [1] * len(ids)}


def load(tokenizer, cfg):
    """Create the cached-token strategy using standard training options."""
    return ProjectionTokenizationStrategy(
        None, tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
