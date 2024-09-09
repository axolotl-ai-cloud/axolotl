from typing import Callable

from attr.validators import is_callable
from dacite import from_dict
from torch.utils.data import Dataset

from axolotl.core.chat.messages import ChatFormattedChats, Chats


class ChatDataset(Dataset):
    def __init__(
        self,
        data: Dataset,
        *args,
        message_transform: Callable = None,
        model_transform=None,
        formatter=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._data = data
        if message_transform is not None and not is_callable(message_transform):
            raise ValueError("message_transform must be a callable function")
        self.message_transform = message_transform
        if model_transform is not None and not is_callable(model_transform):
            raise ValueError("model_transform must be a callable function")
        self.model_transform = model_transform
        self.formatter = formatter

    def __getitem__(self, idx):
        sample = self._data[idx]
        if self.message_transform is not None:
            sample = self.message_transform(sample)
        if self.formatter is not None:
            return from_dict(
                data_class=ChatFormattedChats,
                data={**sample, "formatter": self.formatter},
            )
        return from_dict(data_class=Chats, data=sample)

    def __len__(self):
        return len(self._data)

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __repr__(self):
        return f"ChatDataset({len(self)} examples)"
