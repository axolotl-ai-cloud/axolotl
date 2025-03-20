"""
chat dataset module
"""

import os
from typing import Callable, Optional, Union

from datasets import Dataset
from transformers import PreTrainedTokenizer

from axolotl.core.chat.messages import ChatFormattedChats


class TokenizedChatDataset(Dataset):
    """
    Tokenized chat dataset
    """

    def __init__(
        self,
        data: Dataset,
        model_transform: Union[PreTrainedTokenizer, Callable],
        *args,
        message_transform: Optional[Callable] = None,
        formatter=None,
        process_count: Optional[int] = None,
        keep_in_memory: Optional[bool] = False,
        **kwargs,
    ):
        def map_fn(ex):
            if message_transform is not None:
                ex = message_transform(ex)
            if formatter is not None:
                ex = ChatFormattedChats(
                    formatter=formatter,
                    **ex,
                )
            else:
                ex = ChatFormattedChats(
                    **ex,
                )
            return ex.tokenized(model_transform)

        process_or_cpu_count: int = (
            process_count or os.cpu_count()  # type: ignore[assignment]
        )
        num_proc = min(32, process_or_cpu_count)
        features = data.features.keys()
        tokenized_data = data.map(
            map_fn,
            num_proc=num_proc,
            keep_in_memory=keep_in_memory,
            remove_columns=features,
            desc="Tokenizing Chats",
        )
        super().__init__(tokenized_data.data, *args, **kwargs)
