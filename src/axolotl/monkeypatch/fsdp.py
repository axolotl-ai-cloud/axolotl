"""
Monkeypatch to fix fsdp set state when no previous state was set
"""

import contextlib
from typing import Generator, Optional

import torch
from torch import nn
from torch.distributed.fsdp.api import (
    OptimStateDictConfig,
    StateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel


@staticmethod
@contextlib.contextmanager
def state_dict_type_patch(
    module: nn.Module,
    state_dict_type: StateDictType,
    state_dict_config: Optional[StateDictConfig] = None,
    optim_state_dict_config: Optional[OptimStateDictConfig] = None,
) -> Generator:
    prev_state_dict_settings = FullyShardedDataParallel.set_state_dict_type(
        module,
        state_dict_type,
        state_dict_config,
        optim_state_dict_config,
    )
    yield
    if prev_state_dict_settings.state_dict_type:
        FullyShardedDataParallel.set_state_dict_type(
            module,
            prev_state_dict_settings.state_dict_type,
            prev_state_dict_settings.state_dict_config,
            prev_state_dict_settings.optim_state_dict_config,
        )


def replace_fsdp_state_dict_type():
    torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel.state_dict_type = (
        state_dict_type_patch
    )
