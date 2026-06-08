# Copyright 2026 Axolotl AI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pydantic args for the ringmaster context-parallel plugin."""

from typing import Literal, Optional

from pydantic import BaseModel, model_validator

from axolotl.utils.logging import get_logger

LOG = get_logger(__name__)


class ContextParallelConfig(BaseModel):
    """Nested ``context_parallel:`` config block for the ringmaster plugin."""

    size: int = 1
    """Total context-parallel degree. 1 = disabled."""

    backend: Literal["auto", "ulysses", "ring", "usp"] = "auto"

    ulysses_size: Optional[int] = None
    """All-to-all (head) degree. None = auto-select."""

    ring_size: Optional[int] = None
    """Ring degree. None = auto-select. ulysses_size * ring_size must == size."""

    rotate_method: Literal["allgather", "alltoall"] = "allgather"
    load_balance: Literal["none", "head_tail", "distflash", "per_document", "ptrr"] = "head_tail"
    ring_impl: Literal["auto", "torch_native", "hf_kernels"] = "auto"
    """Ring block-kernel provider. auto -> hf_kernels (FA2/3/4 via HF kernels) when
    the model uses a flash kernel, else torch_native (SDPA/flex)."""

    # NOTE: memory optimizations (tiled MLP, cut_cross_entropy, activation
    # checkpointing/offload) are axolotl-native and compose with CP — configure them
    # through their existing axolotl options, not here.

    @model_validator(mode="after")
    def _validate(self):
        if self.size < 1:
            raise ValueError(f"context_parallel.size must be >= 1 (got {self.size})")
        if self.ulysses_size and self.ring_size:
            if self.ulysses_size * self.ring_size != self.size:
                raise ValueError(
                    f"ulysses_size({self.ulysses_size}) * ring_size({self.ring_size}) "
                    f"!= context_parallel.size({self.size})"
                )
        return self


class ContextParallelArgs(BaseModel):
    """Input args contributed by the ringmaster context-parallel plugin."""

    context_parallel: Optional[ContextParallelConfig] = None
