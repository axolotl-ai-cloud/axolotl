"""Module for Jokes prompts using sharegpt style """
from axolotl.prompt_tokenizers import ShareGPTPromptTokenizingStrategy
from axolotl.prompters import PromptStyle, ShareGPTPrompter


def load(tokenizer, cfg):
    return SimpleJokesShareGPTPromptTokenizingStrategy(
        ShareGPTPrompter(PromptStyle.CHAT.value),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
    )


class SimpleJokesShareGPTPromptTokenizingStrategy(ShareGPTPromptTokenizingStrategy):
    """
    Tokenization strategy for asking bot to tell a joke and then explain why its funny
    """

    # title, text, explanation
    def get_conversation_thread(self, prompt):
        title = "" if not prompt["title"] else prompt["title"] + " "
        return [
            {"from": "human", "value": "Tell me a joke."},
            {"from": "gpt", "value": title + prompt["text"]},
            {"from": "human", "value": "Why is that joke funny?"},
            {"from": "gpt", "value": prompt["explanation"]},
        ]
