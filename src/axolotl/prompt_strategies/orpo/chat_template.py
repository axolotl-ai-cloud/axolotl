"""chatml prompt tokenization strategy for ORPO"""

from typing import Any, Dict, Generator, List, Optional, Tuple

from pydantic import BaseModel

from axolotl.prompt_tokenizers import IGNORE_INDEX, PromptTokenizingStrategy
from axolotl.prompters import Prompter
from axolotl.utils.chat_templates import get_chat_template_from_config


class Message(BaseModel):
    """message/turn"""

    role: str
    content: str
    label: Optional[bool] = None


class MessageList(BaseModel):
    """conversation"""

    messages: List[Message]


def load(
    tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None, **kwargs
):  # pylint: disable=possibly-unused-variable,unused-argument
    """
    chatml transforms for datasets with system, input, chosen, rejected
    """
    chat_template_string = get_chat_template_from_config(
        cfg=cfg, ds_cfg=ds_cfg, tokenizer=tokenizer
    )
    tokenizer.chat_template = chat_template_string

    return ORPOTokenizingStrategy(
        ORPOPrompter(chat_template_string, tokenizer),
        tokenizer,
        cfg.train_on_inputs,
        cfg.sequence_len,
        dataset_parser=ORPODatasetParsingStrategy(),
    )


class ORPODatasetParsingStrategy:
    """Strategy to parse chosen rejected dataset into messagelist"""

    def get_chosen_conversation_thread(self, prompt) -> MessageList:
        """Dataset structure mappings"""

        messages: List[Message] = []
        if system := prompt.get("system", None):
            messages.append(Message(role="system", content=system, label=False))
        messages.append(
            Message(role="user", content=prompt["chosen"][0]["content"], label=False)
        )
        messages.append(
            Message(
                role="assistant", content=prompt["chosen"][1]["content"], label=True
            )
        )
        return MessageList(messages=messages)

    def get_rejected_conversation_thread(self, prompt) -> MessageList:
        """Dataset structure mappings"""

        messages: List[Message] = []
        if system := prompt.get("system", None):
            messages.append(Message(role="system", content=system, label=False))
        messages.append(
            Message(role="user", content=prompt["rejected"][0]["content"], label=False)
        )
        messages.append(
            Message(
                role="assistant", content=prompt["rejected"][1]["content"], label=True
            )
        )
        return MessageList(messages=messages)

    def get_prompt(self, prompt) -> MessageList:
        """Map the data to extract everything up to the last turn"""
        total_msg_len = len(prompt["chosen"])
        total_msg_turns, remainder = divmod(total_msg_len, 2)
        assert remainder == 0, "invalid number of turns"

        messages: List[Message] = []
        if system := prompt.get("system", None):
            messages.append(Message(role="system", content=system, label=False))
        for i in range(total_msg_turns):
            if "prompt" in prompt:
                messages.append(
                    Message(role="user", content=prompt["prompt"], label=False)
                )
            else:
                messages.append(
                    Message(
                        role="user",
                        content=prompt["chosen"][i * 2]["content"],
                        label=False,
                    )
                )
            if i < total_msg_turns - 1:
                messages.append(
                    Message(
                        role="assistant",
                        content=prompt["chosen"][i * 2 + 1]["content"],
                        label=False,
                    )
                )

        return MessageList(messages=messages)

    def get_chosen(self, prompt) -> MessageList:
        res = self.get_prompt(prompt)
        res.messages.append(
            Message(
                role="assistant", content=prompt["chosen"][-1]["content"], label=True
            )
        )
        return res

    def get_rejected(self, prompt) -> MessageList:
        res = self.get_prompt(prompt)
        res.messages.append(
            Message(
                role="assistant", content=prompt["rejected"][-1]["content"], label=True
            )
        )
        return res


class ORPOTokenizingStrategy(PromptTokenizingStrategy):
    """
    rejected_input_ids
    input_ids
    rejected_attention_mask
    attention_mask
    rejected_labels
    labels
    """

    def __init__(
        self,
        *args,
        dataset_parser=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.dataset_parser = dataset_parser

    def tokenize_prompt(self, prompt):
        # pass the rejected prompt/row to the Prompter to get the formatted prompt
        prompt_len = 0
        rejected_message_list: MessageList = (
            self.dataset_parser.get_rejected_conversation_thread(prompt)
        )
        input_ids = []
        labels = []
        for _, (part, label) in enumerate(
            self.prompter.build_prompt(rejected_message_list)
        ):
            if not part:
                continue
            _input_ids = self.tokenizer.encode(part, add_special_tokens=False)
            prev_idx = len(input_ids)
            input_ids += _input_ids[prev_idx:]
            if label:
                labels += input_ids[prev_idx:]
            else:
                labels += [IGNORE_INDEX] * (len(input_ids) - prev_idx)
                prompt_len = len(input_ids)
        # remap the input_ids, attention_mask and labels
        rejected_input_ids = input_ids
        rejected_labels = labels
        # pass the chosen prompt/row to the Prompter to get the formatted prompt
        chosen_message_list: MessageList = (
            self.dataset_parser.get_chosen_conversation_thread(prompt)
        )
        input_ids = []
        labels = []
        for _, (part, label) in enumerate(
            self.prompter.build_prompt(chosen_message_list)
        ):
            if not part:
                continue
            _input_ids = self.tokenizer.encode(part, add_special_tokens=False)
            prev_idx = len(input_ids)
            input_ids += _input_ids[prev_idx:]
            if label:
                labels += input_ids[prev_idx:]
            else:
                labels += [IGNORE_INDEX] * (len(input_ids) - prev_idx)

        return {
            "rejected_input_ids": rejected_input_ids,
            "rejected_labels": rejected_labels,
            "rejected_attention_mask": [1] * len(rejected_labels),
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": [1] * len(labels),
            "prompt_attention_mask": [1] * prompt_len
            + [0] * (len(labels) - prompt_len),
        }


class ORPOPrompter(Prompter):
    """Single Turn prompter for ORPO"""

    def __init__(self, chat_template, tokenizer):
        self.chat_template = chat_template
        self.tokenizer = tokenizer

    def build_prompt(
        self,
        message_list: MessageList,
    ) -> Generator[Tuple[str, bool], None, None]:
        conversation = []
        for message in message_list.messages:
            conversation.append(message.model_dump())
            if message.role == "system":
                yield self.tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=False,
                    chat_template=self.chat_template,
                    tokenize=False,
                ), False
            if message.role == "user":
                yield self.tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=True,
                    chat_template=self.chat_template,
                    tokenize=False,
                ), False
            if message.role == "assistant":
                yield self.tokenizer.apply_chat_template(
                    conversation,
                    add_generation_prompt=False,
                    chat_template=self.chat_template,
                    tokenize=False,
                ), True


def argilla(cfg, **kwargs):  # pylint: disable=possibly-unused-variable,unused-argument
    dataset_parser = ORPODatasetParsingStrategy()

    def transform_fn(sample, tokenizer=None):
        res = {}

        chat_template_string = get_chat_template_from_config(
            cfg=cfg, tokenizer=tokenizer
        )

        res["prompt"] = tokenizer.apply_chat_template(
            [msg.model_dump() for msg in dataset_parser.get_prompt(sample).messages],
            add_generation_prompt=True,
            chat_template=chat_template_string,
            tokenize=False,
        )
        prompt_str_len = len(res["prompt"])
        res["chosen"] = tokenizer.apply_chat_template(
            [msg.model_dump() for msg in dataset_parser.get_chosen(sample).messages],
            add_generation_prompt=False,
            chat_template=chat_template_string,
            tokenize=False,
        )[prompt_str_len:]
        res["rejected"] = tokenizer.apply_chat_template(
            [msg.model_dump() for msg in dataset_parser.get_rejected(sample).messages],
            add_generation_prompt=False,
            chat_template=chat_template_string,
            tokenize=False,
        )[prompt_str_len:]

        return res

    return transform_fn
