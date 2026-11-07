"""DataLoader worker seeding that stays off transformers.trainer_utils.

Workers unpickle their ``worker_init_fn``. ``transformers.trainer_utils`` imports peft at
module level, which imports bitsandbytes and, through transformers' torchao quantizer,
initializes CUDA, so seeding from there did that in every worker (or in the forkserver
that preloads it).
"""

import random

import numpy as np
import torch
from transformers.utils import (
    is_torch_hpu_available,
    is_torch_mlu_available,
    is_torch_musa_available,
    is_torch_npu_available,
    is_torch_xpu_available,
)

_EXTRA_DEVICES = (
    (is_torch_mlu_available, "mlu"),
    (is_torch_musa_available, "musa"),
    (is_torch_npu_available, "npu"),
    (is_torch_hpu_available, "hpu"),
    (is_torch_xpu_available, "xpu"),
)


def seed_worker(worker_id: int, num_workers: int, rank: int) -> None:
    """Seed a DataLoader worker exactly as ``transformers.trainer_utils.seed_worker``."""
    seed = num_workers * rank + torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    for available, device in _EXTRA_DEVICES:
        if available():
            getattr(torch, device).manual_seed_all(seed)
