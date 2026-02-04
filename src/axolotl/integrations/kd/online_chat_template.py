from axolotl.prompt_strategies.chat_template import ChatTemplateStrategy, StrategyLoader
from axolotl.prompters import IGNORE_TOKEN_ID
from axolotl.utils.logging import get_logger

# Configure the logger
LOG = get_logger(__name__)
LOG.setLevel("INFO")


class ChatTemplateStrategyWithOnlineKD(ChatTemplateStrategy):
    @property
    def supports_batched(self) -> bool:
        # batching doesn't work well for logprob data
        return False

    def _get_messages(self, prompt):
        input_prompt = prompt.get("problem")
        return [
            {"role": "user", "content": input_prompt},
        ]

    def _tokenize_single_prompt(self, prompt):
        turns = self.get_conversation_thread(prompt)
        tools = self._get_tools(prompt)
        input_ids = self.prompter.build_prompt(
            turns, tools=tools, add_generation_prompt=True
        )  # type: ignore
        labels = [IGNORE_TOKEN_ID] * len(input_ids)

        return {
            "input_ids": input_ids,
            "prompts": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(input_ids),
        }


class OnlineKDStrategyLoader(StrategyLoader):
    """
    Load ChatTemplateStrategy with KD support using StrategyLoader.
    """

    def _get_strategy_cls(self, cfg):
        return ChatTemplateStrategyWithOnlineKD


load = OnlineKDStrategyLoader()
