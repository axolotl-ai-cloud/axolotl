from typing import Optional, Dict, Any, Callable

from axolotl.core.datasets.chat import TokenizedChatDataset
from axolotl.core.datasets.transforms.chat_builder import chat_message_transform_builder
from axolotl.prompt_tokenizers import DatasetWrappingStrategy


class ChatMessageDatasetWrappingStrategy(DatasetWrappingStrategy):
    def __init__(self, processor, message_transform=None, formatter=None, **kwargs):
        """
        :param processor: tokenizer or image processor
        :param kwargs:
        """
        self.processor = processor
        self.dataset = None
        self.message_transform = message_transform
        self.formatter = formatter

    def wrap_dataset(
        self,
        dataset,
        process_count: Optional[int] = None,
        keep_in_memory: Optional[bool] = False,
        **kwargs,
    ):
        self.dataset = TokenizedChatDataset(
            dataset,
            message_transform=self.message_transform,
            model_transform=self.processor,
            formatter=self.formatter,
            process_count=process_count,
            keep_in_memory=keep_in_memory,
        )
        return self.dataset


def load(tokenizer, cfg, ds_cfg: Optional[Dict[str, Any]] = None):
    ds_cfg = ds_cfg or {}

    field_messages = ds_cfg.get("field_messages")
    message_field_role = ds_cfg.get("message_field_role")
    message_field_content = ds_cfg.get("message_field_content")
    message_field_training = ds_cfg.get("message_field_training")

    builder_kwargs = {}
    if field_messages:
        builder_kwargs["conversations_field"] = field_messages
    if message_field_role:
        builder_kwargs["message_field_role"] = message_field_role
    if message_field_content:
        builder_kwargs["message_field_content"] = message_field_content
    if message_field_training:
        builder_kwargs["message_field_training"] = message_field_training

    chat_template = ds_cfg.get("chat_template", cfg.get("chat_template", "chatml"))
    format_message = lambda x: x
    if chat_template == "chatml":
        from axolotl.core.chat.format.chatml import format_message
    if chat_template.startswith("llama3"):
        from axolotl.core.chat.format.llama3x import format_message
    message_transform: Callable = chat_message_transform_builder(
        train_on_inputs=ds_cfg.get("train_on_inputs", False),
        **builder_kwargs,
    )
    strategy = ChatMessageDatasetWrappingStrategy(tokenizer, message_transform=message_transform, formatter=format_message)

    return strategy
