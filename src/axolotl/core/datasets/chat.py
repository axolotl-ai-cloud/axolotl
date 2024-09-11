import os
from typing import Callable, Optional, Union

from dacite import from_dict
from datasets import Dataset
from transformers import PreTrainedTokenizer

from axolotl.core.chat.messages import ChatFormattedChats, Chats


class TokenizedChatDataset(Dataset):
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
                ex = from_dict(
                    data_class=ChatFormattedChats,
                    data={**ex, "formatter": formatter},
                )
            else:
                ex = from_dict(
                    data_class=ChatFormattedChats,
                    data=ex,
                )
            return ex.tokenized(model_transform)

        num_proc = min(64, process_count if process_count else os.cpu_count())
        tokenized_data = data.map(
            map_fn,
            num_proc=num_proc,
            keep_in_memory=keep_in_memory,
        )
        super().__init__(tokenized_data.data, *args, **kwargs)
